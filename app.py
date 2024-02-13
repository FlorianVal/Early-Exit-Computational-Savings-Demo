# Save this as app.py and run with `streamlit run app.py`
import streamlit as st
import torch
import pandas as pd

from transformers import AutoModelForCausalLM, AutoTokenizer

from src.utils import generate_next_token, breaking_ties
from src.BranchyModel import BranchyModel

st.title("Multi-Head LLM Demo")

def add_and_run(token, head):
    # Update pd with Head and mean of previous heads and actual head
    head_list = st.session_state["computation_pd"]["Head"].to_list() + [head]
    mean = sum(head_list) / len(head_list)
    st.session_state["computation_pd"] = pd.concat([st.session_state["computation_pd"], pd.DataFrame({"Head": [head], "Mean": [mean], "Base model consumption": [st.session_state['head_number']]})], ignore_index=True)
    
    st.session_state['current_sentence'] += token
    _, st.session_state['logits'], _, st.session_state['head_tokens'] = generate_next_token(st.session_state.model, st.session_state.tokenizer, st.session_state['current_sentence'])

def reset():
    st.session_state['computation_pd'] = pd.DataFrame(columns=["Head", "Mean", "Base model consumption"])
    st.session_state['current_sentence'] = "The climate in"
    _, st.session_state['logits'], _, st.session_state['head_tokens'] = generate_next_token(st.session_state.model, st.session_state.tokenizer, st.session_state['current_sentence'])

@st.cache_resource
def load_model(penalty_alpha):
    penalty_map = {0.1:"model_20240118-144039.bin", 
               0.5:"model_20240118-192548.bin", 
               2:"model_20240118-211943.bin", 
               5:"model_20240118-231333.bin",
               10:"model_20240119-010725.bin", 
               20:"model_20240119-030115.bin", 
               0:"model_20240119-135506.bin", 
               1:"model_20240119-154900.bin",
               -20: "model_20240208-072350.bin",
               -10: "model_20240208-052958.bin",
               -5: "model_20240208-033606.bin",
               -2: "model_20240208-014211.bin",
               -1: "model_20240207-234817.bin",
               -0.5: "model_20240207-215423.bin",
               -0.1: "model_20240207-200020.bin"}
    
    model_str = "susnato/phi-1_5_dev"
    model = AutoModelForCausalLM.from_pretrained(model_str).to("cuda:1")
    tokenizer = AutoTokenizer.from_pretrained(model_str)

    branch_locations = list(range(0, 23, 5))
    model = BranchyModel(branch_locations= branch_locations, model= model).to("cuda:1")

    # Load the specific model based on penalty_alpha
    model_path = penalty_map.get(penalty_alpha)
    if model_path:
        model.load_state_dict(torch.load(model_path, map_location="cuda:1"))
    else:
        print("Invalid penalty_alpha. Using default model weights.")

    return model, tokenizer


if "model" not in st.session_state or "tokenizer" not in st.session_state:
    print("Loading model...")
    st.session_state.model, st.session_state.tokenizer = load_model(penalty_alpha=-2)  # Example penalty_alpha
    st.session_state["head_number"] = len(st.session_state.model.branch_locations) + 1
    print(f"Head number: {st.session_state['head_number']}")
# Session state to store the current sentence
if 'current_sentence' not in st.session_state:
    reset()

# Create a container to hold the buttons
cols = st.columns(len(st.session_state.head_tokens))  # Create a column for each token

# Iterate through each head token and create a button in a separate column
for i, (col, token) in enumerate(zip(cols, st.session_state.head_tokens)):
    col.button(f"{st.session_state['head_tokens'][i]}",
                key=f"head_{i}",
                use_container_width=True,
                on_click=add_and_run,
                args=(st.session_state['head_tokens'][i], i))


# Display the current sentence
st.markdown(f"{st.session_state['current_sentence']}")

# Reset button to start over
st.button('Reset', on_click=reset)

if 'computation_pd' in st.session_state:
    st.line_chart(st.session_state['computation_pd'])
    # get last element from a pd
    saved_budget = 100 - ((st.session_state["computation_pd"]["Mean"].iloc[-1] * 100) / st.session_state["computation_pd"]["Base model consumption"].iloc[-1])
    st.markdown(f"You saved **{saved_budget:.2f}%** of the base model consumption.")
    #st.write(st.session_state['computation_pd'])
