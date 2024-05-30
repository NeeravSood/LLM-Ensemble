import gradio as gr
from transformers import AutoModelForCausalLM, AutoTokenizer
import functools
import hashlib
import logging

# Initialize logging and caching
logging.basicConfig(level=logging.INFO)
cache = {}

def get_cache_key(text):
    """Create a hash key for the text to use in caching."""
    return hashlib.md5(text.encode()).hexdigest()

def check_cache(func):
    """Decorator to check cache before processing the model inference."""
    @functools.wraps(func)
    def wrapper(input_text):
        key = get_cache_key(input_text)
        if key in cache:
            return cache[key]
        response = func(input_text)
        cache[key] = response
        return response
    return wrapper

# Dictionary to store loaded models and tokenizers
models = {}

# List of model identifiers from the Hugging Face Model Hub
model_keys = {
    "microsoft/DialoGPT-large": "dialogpt",
    "mistralai/Mistral-7B-Instruct-v0.3": "mistral",
    "baidu/ernie-code-560m": "ernie_code",
    "EleutherAI/gpt-neox-20b": "gpt_neox",
    "meta-llama/Meta-Llama-3-8B-Instruct": "llama"
}

# Define model identifiers
model_ids = list(model_keys.keys())

# Function to load a model and tokenizer
def load_model_and_tokenizer(model_id):
    try:
        model = AutoModelForCausalLM.from_pretrained(model_id)
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        return model, tokenizer
    except Exception as e:
        logging.error(f"Failed to load model {model_id}: {str(e)}")
        return None, None

# Load all models and tokenizers
for model_id in model_ids:
    model, tokenizer = load_model_and_tokenizer(model_id)
    if model and tokenizer:
        models[model_keys[model_id]] = {"model": model, "tokenizer": tokenizer}
        logging.info(f"Successfully loaded {model_id}")

@check_cache
def chatbot_response(input_text):
    input_lower = input_text.lower()
    if "code" in input_lower:
        model_key = "ernie_code"
    elif any(phrase in input_lower for phrase in ["how to", "explain", "strategy", "best practice", "guidance on"]):
        model_key = "mistral"
    elif any(phrase in input_lower for phrase in ["innovation", "future", "trend", "predict", "forecast"]):
        model_key = "gpt_neox"
    elif any(phrase in input_lower for phrase in ["instruct", "teach me", "tutorial", "lesson"]):
        model_key = "llama"
    else:
        model_key = "dialogpt"
    
    model = models[model_key]['model']
    tokenizer = models[model_key]['tokenizer']
    tokens = tokenizer.encode(input_text, return_tensors="pt")
    output = model.generate(tokens, max_length=500)
    response = tokenizer.decode(output[0], skip_special_tokens=True)
    
    return response

"""
Gradio interface setup
"""
def respond(
    message,
    history: list[tuple[str, str]],
    system_message,
    max_tokens,
    temperature,
    top_p,
):
    messages = [{"role": "system", "content": system_message}]

    for val in history:
        if val[0]:
            messages.append({"role": "user", "content": val[0]})
        if val[1]:
            messages.append({"role": "assistant", "content": val[1]})

    messages.append({"role": "user", "content": message})

    # Generate response from the ensemble
    response = chatbot_response(message)
    return response

demo = gr.ChatInterface(
    respond,
    additional_inputs=[
        gr.Textbox(value="Hello, this is an example of how to use an ensemble in a chatbot. Feel free to test!", label="System message"),
        gr.Slider(minimum=1, maximum=2048, value=512, step=1, label="Max new tokens"),
        gr.Slider(minimum=0.1, maximum=4.0, value=0.7, step=0.1, label="Temperature"),
        gr.Slider(
            minimum=0.1,
            maximum=1.0,
            value=0.95,
            step=0.05,
            label="Top-p (nucleus sampling)",
        ),
    ],
)

if __name__ == "__main__":
    demo.launch()
