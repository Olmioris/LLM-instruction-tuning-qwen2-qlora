from src.utils.model_loader import load_tokenizer

def test_tokenizer_load():
    model_path = "/content/drive/MyDrive/Qwen2-0.5B-SFT-MultiDomain"
    tokenizer = load_tokenizer(model_path)
    assert tokenizer is not None