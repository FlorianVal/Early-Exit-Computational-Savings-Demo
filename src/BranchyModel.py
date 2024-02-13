from typing import Dict, List, Optional
from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F
from transformers import PreTrainedModel
from transformers.cache_utils import Cache, DynamicCache
from transformers.modeling_attn_mask_utils import _prepare_4d_causal_attention_mask
from transformers.utils import ModelOutput


@dataclass
class CausalBranchyLLMOutputWithPast(ModelOutput):
    loss: Optional[torch.Tensor] = None
    lm_loss: Optional[torch.Tensor] = None
    head_loss: Optional[torch.Tensor] = None
    logits: torch.Tensor = None
    head_outputs: Optional[torch.Tensor] = None
    past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None

class Branch(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.layernorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=True)
    
    def forward(self, x):
        x = self.layernorm(x)
        x = self.lm_head(x)
        return x

class BranchyModel(PreTrainedModel):
    """
    This class is a wrapper for transformer models with added functionality for branchy networks.
    It uses BranchyConfig to initialize a model and later will be extended to add branches.

    Args:
        branch_locations (List[int]): The locations of the branches in the model.
        starts indexing from 0. Branch 0 is after layer 0.
        model (PreTrainedModel): The underlying transformer model to wrap.

    Returns:
        A model instance with the given configuration.
    """

    def __init__(self, branch_locations, model, loss_type="kl_div", penality_weight=None):
        super().__init__(model.config)
        # Initialize the base transformer model
        self.model = model
        self.branch_locations = branch_locations
        self.loss_type = loss_type
        self.penality_weight = penality_weight
        if self.loss_type == "penalized_cross_entropy":
            assert self.penality_weight is not None, "penality_weight must be provided for penalized_cross_entropy loss"
        # Get details on layering inside the model
        if hasattr(self.model.config, "n_layer") or hasattr(
            self.model.config, "num_hidden_layers"
        ):  # If there is no n_layer in the config, there might be ways to get it from the model itself
            self.num_layers = (
                self.model.config.n_layer
                if hasattr(self.model.config, "n_layer")
                else self.model.config.num_hidden_layers
            )
        else:
            raise ValueError("cannot find n_layer in config")
        # if no branch locations are specified, branch at every layer
        if self.branch_locations is None:
            self.branch_locations = list(range(self.num_layers - 1))
            
        assert self.num_layers > 0, "The number of layers must be greater than 0"
        assert (
            len(self.branch_locations) < self.num_layers
        ), "The number of branches must be less than the number of layers"
        assert all(
            [0 <= i < self.num_layers for i in self.branch_locations]
        ), "The branch locations must be between 0 and num_layers"


        # Make sure the base model is frozen
        for param in self.model.parameters():
            param.requires_grad = False

        # Instantiate heads. Default: heads are copies of the lm_head
        self.model.heads = torch.nn.ModuleList(
            [
                Branch(self.model.config) for _ in range(len(self.branch_locations))
            ]
        )

        # initialize heads
        for head in self.model.heads:
            head.apply(self.model._init_weights)
            # Make them trainable
            for param in head.parameters():
                param.requires_grad = True

        self.post_init()

    # Copied from transformers.models.llama.modeling_llama.LlamaForCausalLM.prepare_inputs_for_generation
    def prepare_inputs_for_generation(
        self,
        input_ids,
        past_key_values=None,
        attention_mask=None,
        inputs_embeds=None,
        **kwargs,
    ):
        if past_key_values is not None:
            if isinstance(past_key_values, Cache):
                cache_length = past_key_values.get_seq_length()
                past_length = past_key_values.seen_tokens
                max_cache_length = past_key_values.get_max_length()
            else:
                cache_length = past_length = past_key_values[0][0].shape[2]
                max_cache_length = None

            # Keep only the unprocessed tokens:
            # 1 - If the length of the attention_mask exceeds the length of input_ids, then we are in a setting where
            # some of the inputs are exclusivelly passed as part of the cache (e.g. when passing input_embeds as
            # input)
            if (
                attention_mask is not None
                and attention_mask.shape[1] > input_ids.shape[1]
            ):
                input_ids = input_ids[:, -(attention_mask.shape[1] - past_length) :]
            # 2 - If the past_length is smaller than input_ids', then input_ids holds all input tokens. We can discard
            # input_ids based on the past_length.
            elif past_length < input_ids.shape[1]:
                input_ids = input_ids[:, past_length:]
            # 3 - Otherwise (past_length >= input_ids.shape[1]), let's assume input_ids only has unprocessed tokens.

            # If we are about to go beyond the maximum cache length, we need to crop the input attention mask.
            if (
                max_cache_length is not None
                and attention_mask is not None
                and cache_length + input_ids.shape[1] > max_cache_length
            ):
                attention_mask = attention_mask[:, -max_cache_length:]

        position_ids = kwargs.get("position_ids", None)
        if attention_mask is not None and position_ids is None:
            # create position_ids on the fly for batch generation
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            if past_key_values:
                position_ids = position_ids[:, -input_ids.shape[1] :]

        # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids}

        model_inputs.update(
            {
                "position_ids": position_ids,
                "past_key_values": past_key_values,
                "use_cache": kwargs.get("use_cache"),
                "attention_mask": attention_mask,
                "fixed_output_head": kwargs.get("fixed_output_head", None),
            }
        )
        return model_inputs

    def compute_self_supervision_loss(
        self,
        aux_logits: torch.Tensor,
        lm_logits: torch.Tensor,
        return_per_head: bool = False,
    ) -> Dict[str, torch.Tensor]:
        last_aux_logits = aux_logits[..., -1, :]
        last_lm_logits = lm_logits[..., -1, :]

        repeated_last_lm_logits = last_lm_logits.repeat(
            last_aux_logits.shape[0], 1, 1, 1
        )
        losses = []
        # Can be useful to have detailed loss per head for comparison of performance
        if return_per_head:
            for head_logit in last_aux_logits:
                if self.loss_type == "kl_div":
                    losses.append(
                        nn.KLDivLoss(reduction="batchmean")(
                            F.log_softmax(head_logit, dim=-1),
                            F.softmax(last_lm_logits, dim=-1),
                        )
                    )
                elif self.loss_type == "cross_entropy":
                    losses.append(
                        nn.CrossEntropyLoss(reduction="mean")(
                            head_logit, torch.argmax(last_lm_logits, dim=-1)
                        )
                    )
                elif self.loss_type == "penalized_cross_entropy":
                    ce_loss = nn.CrossEntropyLoss(reduction="mean")(
                        head_logit, torch.argmax(last_lm_logits, dim=-1)
                    )
                    probas = F.softmax(head_logit, dim=-1)
                    entropy = torch.mean(-torch.sum(probas * torch.log(probas + 1e-8), dim=-1))
                    #losses.append(ce_loss - self.penality_weight * (1.0 / (1.0 + entropy)))
                    losses.append(ce_loss - self.penality_weight * entropy)
                else:
                    raise ValueError(
                        "The loss type must be either kl_div or cross_entropy"
                    )
            loss = torch.stack(losses, dim=0).mean(dim=-1)
        else:
            # Compute the KL divergence between the last auxiliary head and the last LM head
            if self.loss_type == "kl_div":
                loss = nn.KLDivLoss(reduction="batchmean")(
                    F.log_softmax(last_aux_logits.view(-1, self.config.vocab_size), dim=-1),
                    F.softmax(
                        repeated_last_lm_logits.view(-1, self.config.vocab_size), dim=-1
                    ),
                )
            elif self.loss_type == "cross_entropy":
                loss = nn.CrossEntropyLoss(reduction="mean")(
                    last_aux_logits.view(-1, self.config.vocab_size),
                    torch.argmax(
                        repeated_last_lm_logits.view(-1, self.config.vocab_size), dim=-1
                    ),
                )
            elif self.loss_type == "penalized_cross_entropy":
                ce_loss = nn.CrossEntropyLoss(reduction="mean")(
                    last_aux_logits.view(-1, self.config.vocab_size), 
                    torch.argmax(
                        repeated_last_lm_logits.view(-1, self.config.vocab_size), dim=-1
                    ),
                )
                probas = F.softmax(
                    last_aux_logits.view(-1, self.config.vocab_size), dim=-1
                )
                entropy = torch.mean(-torch.sum(probas * torch.log(probas + 1e-8), dim=-1))
                loss = ce_loss + self.penality_weight * entropy
            else:
                raise ValueError(
                    "The loss type must be either kl_div or cross_entropy"
                )
        if return_per_head:
            return {"loss": loss, "aux_loss": torch.stack(losses)}
        else:
            return {"loss": loss, "aux_loss": None}

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        self_supervision: Optional[bool] = None,
        fixed_output_head: Optional[int] = None,
    ):
        output_attentions = (
            output_attentions
            if output_attentions is not None
            else self.config.output_attentions
        )
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache

        if self_supervision:
            output_hidden_states = True
            return self.forward_for_training(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                inputs_embeds=inputs_embeds,
                labels=labels,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
        else:
            return self.forward_for_inference(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                inputs_embeds=inputs_embeds,
                use_cache=use_cache,
                return_dict=return_dict,
                fixed_output_head=fixed_output_head,
            )

    def forward_for_inference(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        fixed_output_head: Optional[int] = None,
    ):
        if fixed_output_head not in self.branch_locations and fixed_output_head is not None and fixed_output_head != -1:
            raise ValueError(
                "The fixed output head must be one of the branch locations"
            )
        # retrieve input_ids and inputs_embeds
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both decoder_input_ids and decoder_inputs_embeds at the same time")
        elif input_ids is not None:
            batch_size, seq_length = input_ids.shape
        elif inputs_embeds is not None:
            batch_size, seq_length, _ = inputs_embeds.shape
        else:
            raise ValueError("You have to specify either decoder_input_ids or decoder_inputs_embeds")
        
        past_key_values_length = 0
        
        if use_cache:
            use_legacy_cache = not isinstance(past_key_values, Cache)
            if use_legacy_cache:
                past_key_values = DynamicCache.from_legacy_cache(past_key_values)
            past_key_values_length = past_key_values.get_usable_length(seq_length)
        
        if position_ids is None:
            device = input_ids.device if input_ids is not None else inputs_embeds.device
            position_ids = torch.arange(
                past_key_values_length, seq_length + past_key_values_length, dtype=torch.long, device=device
            )
            position_ids = position_ids.unsqueeze(0)

        if inputs_embeds is None:
            inputs_embeds = self.model.model.embed_tokens(input_ids)

        inputs_embeds = self.model.model.embed_dropout(inputs_embeds)
        
        # Attention mask.
        if self.model.model._use_flash_attention_2:
            # 2d mask is passed through the layers
            attention_mask = attention_mask if (attention_mask is not None and 0 in attention_mask) else None
        else:
            # 4d mask is passed through the layers
            attention_mask = _prepare_4d_causal_attention_mask(
                attention_mask, (batch_size, seq_length), inputs_embeds, past_key_values_length
            )
        all_head_logits = []
        hidden_states = inputs_embeds
        is_early_exited = False
        for layer_idx, decoder_layer in enumerate(self.model.model.layers):
            layer_outputs = decoder_layer(
                    hidden_states,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    past_key_value=past_key_values,
                    use_cache=use_cache,
                )
            
            hidden_states = layer_outputs[0]

            if use_cache:
                next_decoder_cache = layer_outputs[1]
                
            if fixed_output_head is not None and layer_idx == fixed_output_head:
                # find postion of layer idx in branch_locations
                branch_idx = self.branch_locations.index(layer_idx)
                logits = self.model.heads[branch_idx](hidden_states)
                is_early_exited = True
                break
            elif fixed_output_head == -1 and layer_idx in self.branch_locations:
                # -1 means output all heads
                branch_idx = self.branch_locations.index(layer_idx)
                logits = self.model.heads[branch_idx](hidden_states)
                all_head_logits.append(logits)
            
        if not is_early_exited:
            hidden_states = self.model.model.final_layernorm(hidden_states)
            logits = self.model.lm_head(hidden_states)
            if fixed_output_head == -1:
                all_head_logits.append(logits)
                all_head_logits = torch.stack(all_head_logits, dim=0)
        next_cache = None
        if use_cache:
            next_cache = next_decoder_cache.to_legacy_cache() if use_legacy_cache else next_decoder_cache
        if not return_dict:
            return tuple(v for v in [logits, next_cache] if v is not None)
        
        return CausalBranchyLLMOutputWithPast(
            logits=logits,
            head_outputs=all_head_logits,
            past_key_values=next_cache,
        )
        
    def forward_for_training(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ):

        if not output_hidden_states:
            raise ValueError("output_hidden_states must be True for BranchyLLM")
        if labels is not None:
            raise NotImplementedError("BranchyLLM only supports self-supervision")
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        if not hasattr(outputs, "hidden_states") or outputs.hidden_states is None:
            raise ValueError("The model must return hidden states")
        hidden_states = outputs.hidden_states


        heads_logits = []
        for i, branch in enumerate(self.branch_locations):
            heads_logits.append(
                self.model.heads[i](
                    hidden_states[branch]
                )
            )
        lm_logits = self.model.lm_head(hidden_states[-1])

        heads_logits = torch.stack(heads_logits, dim=0).float()
        lm_logits = lm_logits.float()
        logits = torch.cat([heads_logits, lm_logits.unsqueeze(0)], dim=0)

        loss = None
        lm_loss = None
        aux_loss = None

        losses = self.compute_self_supervision_loss(
            heads_logits, lm_logits, return_per_head=True
        )
        loss = losses["loss"]
        if losses["aux_loss"] is not None:
            aux_loss = losses["aux_loss"]

        if not return_dict:
            output = (logits,) + outputs[1:]
            return ((loss, aux_loss, lm_loss) + output) if loss is not None else output

        return CausalBranchyLLMOutputWithPast(
            loss=loss,
            lm_loss=lm_loss,
            head_loss=aux_loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )