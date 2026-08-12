import pytest
from src.utils.model_loader import load_tokenizer
from src.training.config import WEAK_MODE

def test_tokenizer_loads():
    if WEAK_MODE:
        pytest.skip("Skipping tokenizer test in WEAK_MODE")

    tokenizer = load_tokenizer("Qwen/Qwen2-0.5B-Instruct")
    assert tokenizer.pad_token_id is not None