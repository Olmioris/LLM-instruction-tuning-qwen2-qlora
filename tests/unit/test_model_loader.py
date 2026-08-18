import logging
from src.utils.model_loader import load_tokenizer
from src.training.config import MODEL_NAME, WEAK_MODE

logger = logging.getLogger("app")

def test_load_tokenizer_normal_mode(monkeypatch):
    if WEAK_MODE:
        monkeypatch.setattr("src.training.config.WEAK_MODE", False)

    tok = load_tokenizer(MODEL_NAME)
    assert tok is not None
    assert tok.pad_token is not None