import torch

def test_model_generates(base_model, tokenizer):
    inputs = tokenizer("Привет, как дела?", return_tensors="pt")
    out = base_model.generate(**inputs, max_new_tokens=10)
    text = tokenizer.decode(out[0])
    assert len(text.strip()) > 0


def test_sft_semantic_quality(base_model, tokenizer):
    prompt = "Что такое supervised learning?"
    inputs = tokenizer(prompt, return_tensors="pt")
    out = base_model.generate(**inputs, max_new_tokens=30)
    text = tokenizer.decode(out[0]).lower()
    assert "обуч" in text or "данн" in text


def test_generation_deterministic(base_model, tokenizer):
    torch.manual_seed(42)
    inputs = tokenizer("Привет", return_tensors="pt")
    out1 = base_model.generate(**inputs, max_new_tokens=5)
    out2 = base_model.generate(**inputs, max_new_tokens=5)
    assert tokenizer.decode(out1[0]) == tokenizer.decode(out2[0])


def test_long_prompt(base_model, tokenizer):
    prompt = "Привет " * 200
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True)
    out = base_model.generate(**inputs, max_new_tokens=10)
    assert out is not None