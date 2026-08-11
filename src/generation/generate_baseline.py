import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

def load_baseline(model_name: str):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float32,
        device_map="cpu",
    )
    return model, tokenizer

def generate_baseline(model, tokenizer, prompt: str, max_new_tokens: int = 200):
    inputs = tokenizer(prompt, return_tensors="pt")
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=max_new_tokens)
    return tokenizer.decode(out[0], skip_special_tokens=True)