import pytest
from src.training.config import WEAK_MODE, MODEL_NAME
from src.utils.model_loader import load_tokenizer
from transformers import AutoModelForCausalLM

@pytest.mark.skipif(WEAK_MODE, reason="Weak mode: model generation skipped")
def test_model_generates_text():
    tok = load_tokenizer(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)

    inputs = tok("Привет!", return_tensors="pt")
    out = model.generate(**inputs, max_new_tokens=8)
    text = tok.decode(out[0])

    assert isinstance(text, str)
    assert len(text.strip()) > 0