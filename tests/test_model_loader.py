from src.utils.model_loader import load_tokenizer

def test_tokenizer_load():
    tokenizer = load_tokenizer("/content/drive/MyDrive/Qwen2-0.5B-SFT-MultiDomain")
    assert tokenizer is not None