# Save this as app.py and run with `streamlit run app.py`
import time
import streamlit as st
import torch
import pandas as pd
import plotly.graph_objects as go
import numpy as np

from plotly.subplots import make_subplots
from transformers import AutoModelForCausalLM, AutoTokenizer
from typer import clear
from annotated_text import annotated_text

st.title("Multi-Head LLM Demo")
st.markdown("""This is a demo of a multi-head language model with early exit capabilities. 
            The model is based on the Phi-2 architecture and model is available here : https://huggingface.co/valcore/Branchy-Phi-2.
            \nThe model has four heads, each of which can be exited early based on a threshold. The graph show the depth of early exit for each token (the deeper being the faster) and the time taken to generate each token.
            Early exited tokens are annotated with the depth of early exit (with a float smaller than 1, 1 being the deepest)
            """)

def annotated_to_normal(text):
    result = ""
    for elem in text:
        if isinstance(elem, tuple):
            result += elem[0]
        else:
            result += elem
    return result

def generate_next_token(device="cpu"):
    print(f"Generating next token from {st.session_state.messages}")
    inputs = ""
    for message in st.session_state.messages:
        inputs += message["role"] + ": " + annotated_to_normal(message["content"]) + "\n"
    inputs += "Assistant:"
    print(f"Inputs: {inputs}")
    inputs = st.session_state.tokenizer.encode(inputs, return_tensors="pt").to(device)
    for i in range(50):
        start = time.time()
        outputs = st.session_state.model(inputs)
        stop = time.time()
        next_token_logits = outputs.logits[:, -1, :].squeeze()
        next_token_probs = torch.softmax(next_token_logits, dim=-1)
        next_token_id = torch.argmax(next_token_probs, dim=-1)
        if next_token_id == 50256:
            break
        print(inputs.shape, next_token_id.shape)
        inputs = torch.cat([inputs, next_token_id.unsqueeze(0).unsqueeze(-1)], dim=-1)
        next_token = st.session_state.tokenizer.decode(next_token_id, return_tensors="pt")
        time_taken = stop - start
        branch_locations = st.session_state.model.config.branch_locations
        print(outputs.head_indices)
        if outputs.head_indices in branch_locations:
            print(sorted(branch_locations, reverse=True))
            early_exit = (branch_locations.index(outputs.head_indices) + 1) / len(branch_locations)
        else:
            early_exit = 1.25
        # Add data to dataframe
        new_row = pd.DataFrame({"Time taken (in ms)": [time_taken], "Early exit depth": [early_exit]})
        st.session_state.data = pd.concat([st.session_state.data, new_row], ignore_index=True)
        yield next_token, early_exit

@st.cache_resource
def load_model(model_str, tokenizer_str, device="cpu"):
    model = AutoModelForCausalLM.from_pretrained(model_str, trust_remote_code=True).to(device)
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_str)
    return model, tokenizer

model_str = "valcore/Branchy-Phi-2"
tokenizer_str = "microsoft/Phi-2"
device = "cuda:0" if torch.cuda.is_available() else "cpu"

if "model" not in st.session_state or "tokenizer" not in st.session_state:
    print(f"Loading model on {device}")
    st.session_state.model, st.session_state.tokenizer = load_model(model_str, tokenizer_str, device)

# Initialize chat history and dataframe
if "messages" not in st.session_state:
    st.session_state.messages = []
st.session_state.data = pd.DataFrame(columns=["Time taken (in ms)", "Early exit depth"])

col1, col2 = st.columns([1, 4])

with col1:
    early_exit = st.checkbox("Early exit", value=False)
    if early_exit:
        st.session_state.model.head_thresholds = [2.506962537765503, 2.656052589416504, 1.924393653869629, 1.4434680938720703]
    else:
        st.session_state.model.head_thresholds = [10., 10., 10., 10.]
    clear_session = st.button("Clear session")
    if clear_session:
        print("Clearing session")
        st.session_state.messages = []
        st.session_state.data = pd.DataFrame(columns=["Time taken (in ms)", "Early exit depth"])

with col2:
    # Display chat messages from history on app rerun
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            annotated_text(message["content"])

    prompt = st.chat_input("What is up?")
    # React to user input
    if prompt:
        # Display user message in chat message container
        with st.chat_message("User"):
            st.markdown(prompt)
        # Add user message to chat history
        st.session_state.messages.append({"role": "User", "content": prompt})

        # Display assistant response in chat message container
        with st.chat_message("Assistant"):
            response = []
            with st.spinner('Running inference...'):
                for next_token, early_exit in generate_next_token(device):
                    if early_exit > 0.0 and early_exit != 1.25:
                        response.append(tuple((next_token, str(early_exit))))
                    else:
                        response.append(next_token)
                    print(response)
            annotated_text(response)

        # Add assistant response to chat history
        st.session_state.messages.append({"role": "Assistant", "content": response})

        # Assuming st.session_state.data is a pandas DataFrame
        df = st.session_state.data

        # Calculate the max time taken and add a 10% margin
        max_time = df["Time taken (in ms)"].max()
        time_axis_max = max_time * 1.1  # 10% margin


        # Create figure with secondary y-axis
        fig = make_subplots(specs=[[{"secondary_y": True}]])

        # Add traces
        fig.add_trace(
            go.Scatter(x=df.index, y=df["Time taken (in ms)"], name="Time taken (in ms)"),
            secondary_y=False,
        )

        fig.add_trace(
            go.Scatter(x=df.index, y=df["Early exit depth"], name="Early exit depth"),
            secondary_y=True,
        )

        # Set x-axis title
        fig.update_xaxes(title_text="Index")

        # Set y-axes titles
        fig.update_yaxes(
            title_text="Time taken (in ms)", 
            secondary_y=False,
            range=[0, time_axis_max],
            tickmode='linear',
            dtick=np.ceil(time_axis_max / 5 / 10) * 10  # Round to nearest 10
        )
        fig.update_yaxes(
            title_text="Early exit depth", 
            secondary_y=True, 
            range=[0, 1.25],
            tickmode='linear',
            dtick=0.25
        )
        
        fig.update_layout(
            title_text="Time Taken vs Early Exit Depth",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        # Use Streamlit to display the Plotly chart
        st.plotly_chart(fig)
        
        #st.line_chart(st.session_state.data, x=None, y=["Time taken (in ms)", "Early exit depth"])
        print(st.session_state.messages)
