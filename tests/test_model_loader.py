from src.utils.model_loader import load_tokenizer

def test_tokenizer_load():
    tokenizer = load_tokenizer("qwen2-0.5b-lora")
    assert tokenizer is not None