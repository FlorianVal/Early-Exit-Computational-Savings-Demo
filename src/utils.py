import torch

def generate_next_token(model, tokenizer, input, method='greedy'):
    """
    Generate the next token of a sequence using the given model and tokenizer.
    Specific for multi branched models.
    Only output token from last head.

    Args:
        model (torch.nn.Module): The model to use for generation.
        tokenizer (transformers.PreTrainedTokenizer): The tokenizer to use for generation.
        input (str): The input text to generate from.

    Returns:
        token (str): The next token in the sequence.
        logits (torch.Tensor): The logits of the next token. of shape[Head, vocab_size]
        new_sequence (str): The new sequence after adding the next token.
    """
    device = model.device
    input_ids = tokenizer.encode(input, return_tensors="pt").to(device)
    model.eval()
    logits = model(input_ids, fixed_output_head=-1).head_outputs[..., -1, :].squeeze(1) # squeeze batch dimension as it is 1 new shape is (head_count, vocab_size)
    if logits == []:
        raise ValueError("Model does not have head_outputs")
    if method == 'greedy':
        head_tokens = torch.argmax(logits, dim=-1)
    elif method == 'sample':
        head_tokens = torch.multinomial(torch.nn.functional.softmax(logits, dim=-1), num_samples=1)
    elif method == 'top_k':
        k = 5
        top_k = torch.topk(logits, k, dim=-1)
        top_k_logits, top_k_indices = top_k.values, top_k.indices
        top_k_probs = torch.nn.functional.softmax(top_k_logits, dim=-1)
        head_tokens = top_k_indices[torch.arange(top_k_probs.shape[0]), torch.multinomial(top_k_probs, num_samples=1).squeeze()]
    elif method == 'top_p':
        # logits is of shape [batch, vocab_size]
        p = 0.9
        probs = torch.nn.functional.softmax(logits, dim=-1)
        sorted_probs, sorted_indices = torch.sort(probs, descending=True, dim=-1)
        cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
        sorted_indices_to_remove = cumulative_probs > p
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0
        indices_to_remove = sorted_indices[sorted_indices_to_remove]
        tmp_logits = logits.clone()
        for i in range(logits.shape[0]):
            tmp_logits[i, indices_to_remove[i]] = float('-inf')
        head_tokens = torch.multinomial(torch.nn.functional.softmax(tmp_logits, dim=-1), num_samples=1).squeeze()
    else:
        raise ValueError(f"Unknown method: {method}")
    head_tokens = tokenizer.batch_decode(head_tokens) # Treat head dim as batch dim
    new_sequence = input + head_tokens[-1]
    return head_tokens[-1], logits, new_sequence, head_tokens
    

def breaking_ties(tensor):
    return torch.sub(torch.topk(tensor, 2, dim=-1).values[..., 0], torch.topk(tensor, 2, dim=-1).values[..., 1]).squeeze()
