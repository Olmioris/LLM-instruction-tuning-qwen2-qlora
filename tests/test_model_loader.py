from src.utils.model_loader import load_tokenizer

def test_tokenizer_load():
    tokenizer = load_tokenizer("./models/local-qwen")
    assert tokenizer is not None