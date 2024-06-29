import gradio as gr
import torch
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from transformers import AutoModelForCausalLM, AutoTokenizer
import time
import numpy as np

# Load the model and tokenizer
model_str = "valcore/Branchy-Phi-2"
tokenizer_str = "microsoft/Phi-2"
device = "cuda" if torch.cuda.is_available() else "cpu"

model = AutoModelForCausalLM.from_pretrained(model_str, trust_remote_code=True).to(device)
tokenizer = AutoTokenizer.from_pretrained(tokenizer_str)

# Initialize dataframe for storing token generation data
data = pd.DataFrame(columns=["Time taken (in ms)", "Early exit depth", "Token"])

# Define thresholds for different epsilon values
epsilon_thresholds = {
    0.4: [1.0307843685150146, 0.8693032264709473, 0.6637287139892578, 0.3111608028411865],
    0.5: [1.505380630493164, 1.5712471008300781, 1.1971790790557861, 0.6908178329467773],
    0.6: [2.0270779132843018, 1.8969502449035645, 1.4789371490478516, 0.9875392913818359],
    0.7: [2.506962537765503, 2.656052589416504, 1.924393653869629, 1.4434680938720703],
    0.8: [3.3786778450012207, 2.568857192993164, 2.5665550231933594, 2.006620407104492],
    0.9: [3.187114715576172, 3.442272663116455, 2.636230945587158, 2.460529088973999],
    1.0: [10.0, 10.0, 10.0, 10.0]  # Effectively disable early exits
}

# Global variable to control generation
stop_generation = False

def create_plot():
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    
    fig.add_trace(
        go.Scatter(
            x=data.index,
            y=data["Time taken (in ms)"],
            name="Time taken (ms)",
            text=data["Token"],
            hovertemplate="<b>Token:</b> %{text}<br><b>Time:</b> %{y:.2f} ms<extra></extra>",
        ),
        secondary_y=False,
    )
    
    fig.add_trace(
        go.Scatter(
            x=data.index,
            y=data["Early exit depth"],
            name="Early exit depth",
            text=data["Token"],
            hovertemplate="<b>Token:</b> %{text}<br><b>Depth:</b> %{y:.2f}<extra></extra>",
        ),
        secondary_y=True,
    )
    
    fig.update_layout(
        title_text="Token Generation Metrics",
        xaxis_title="Token Index",
        yaxis_title="Time (ms)",
        yaxis2_title="Exit Depth",
        hovermode="closest",
    )
    
    fig.update_yaxes(range=[0, 1.1], secondary_y=True)
    
    return fig

def truncate_context(input_ids, max_length=2048):
    if len(input_ids[0]) > max_length:
        return input_ids[:, -max_length:]
    return input_ids

def generate_response(message, chat_history, epsilon):
    global data, stop_generation
    data = pd.DataFrame(columns=["Time taken (in ms)", "Early exit depth", "Token"])
    stop_generation = False
    
    # Set model thresholds based on epsilon
    model.head_thresholds = torch.tensor(epsilon_thresholds[epsilon])
    
    full_response = ""
    chat_history = chat_history or []
    inputs = tokenizer.encode(message, return_tensors="pt").to(device)
    
    while not stop_generation:
        inputs = truncate_context(inputs)
        start = time.time()
        outputs = model(inputs)
        stop = time.time()
        
        next_token_logits = outputs.logits[:, -1, :]
        next_token_id = torch.argmax(next_token_logits, dim=-1)
        
        if next_token_id.item() == tokenizer.eos_token_id:
            break
        
        inputs = torch.cat([inputs, next_token_id.unsqueeze(0)], dim=-1)
        next_token = tokenizer.decode(next_token_id)
        full_response += next_token
        
        time_taken = (stop - start) * 1000  # Convert to milliseconds
        branch_locations = model.config.branch_locations
        early_exit = (branch_locations.index(outputs.head_indices) + 1) / len(branch_locations) if outputs.head_indices in branch_locations else 1.0
        
        new_row = pd.DataFrame({
            "Time taken (in ms)": [time_taken],
            "Early exit depth": [early_exit],
            "Token": [next_token]
        })
        data = pd.concat([data, new_row], ignore_index=True)
        
        new_history = chat_history + [(message, full_response)]
        yield new_history, new_history, gr.update(value=create_plot())

def stop_gen():
    global stop_generation
    stop_generation = True
    return gr.update(interactive=False)

with gr.Blocks() as demo:
    gr.Markdown("# Multi-Head LLM Demo with Early Exit Capabilities 🤗")
    gr.Markdown("""This is a demo of a multi-head language model with early exit capabilities. 
                The model is based on the Phi-2 architecture and is available here: https://huggingface.co/valcore/Branchy-Phi-2.
                The model has four heads, each of which can be exited early based on a threshold. The graph shows the depth of early exit for each token and the time taken to generate each token.
                Use the slider to adjust the early exit threshold. Lower values allow for more early exits, potentially speeding up generation at the cost of accuracy.
                """)
    chatbot = gr.Chatbot()
    msg = gr.Textbox(label="Message")
    epsilon = gr.Slider(minimum=0.4, maximum=1.0, value=0.7, step=0.1, label="Epsilon")
    
    with gr.Row():
        send = gr.Button("Send")
        stop = gr.Button("Stop Generation")
    
    graph = gr.Plot()
    
    send.click(generate_response, inputs=[msg, chatbot, epsilon], outputs=[chatbot, chatbot, graph])
    msg.submit(generate_response, inputs=[msg, chatbot, epsilon], outputs=[chatbot, chatbot, graph])
    stop.click(stop_gen, outputs=[stop])

demo.queue().launch()