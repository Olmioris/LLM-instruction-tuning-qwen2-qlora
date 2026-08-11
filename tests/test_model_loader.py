from src.utils.model_loader import load_tokenizer

def test_tokenizer_loads():
    tokenizer = load_tokenizer("Qwen/Qwen2-0.5B-Instruct")
    assert tokenizer.pad_token_id is not None