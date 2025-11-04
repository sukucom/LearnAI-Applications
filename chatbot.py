import gradio as gr
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# Choose a better conversational model (more coherent)
model_name = "facebook/blenderbot-400M-distill"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

def chatbot_response(message, history):
    # Combine past conversation into context
    history_text = ""
    for human, bot in history:
        history_text += f"User: {human}\nAI: {bot}\n"
    # Add current user input
    prompt = history_text + f"User: {message}\nAI:"

    inputs = tokenizer(prompt, return_tensors="pt", truncation=True)
    reply_ids = model.generate(
        **inputs,
        max_new_tokens=150,
        temperature=0.7,
        top_p=0.9,
        do_sample=True,
        use_cache=False
    )
    reply = tokenizer.decode(reply_ids[0], skip_special_tokens=True)

    # Extract only AI response part
    reply = reply.split("AI:")[-1].strip()
    return reply

# Use a ChatGPT-like UI
chat_ui = gr.ChatInterface(
    fn=chatbot_response,
    title="AI Chat Assistant 🤖",
    description="Chat with an intelligent AI assistant running locally inside Docker!"
)

if __name__ == "__main__":
    chat_ui.launch(server_name="0.0.0.0", server_port=7860)
