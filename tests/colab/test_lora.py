def test_lora_applies(lora_model):
    has_lora = any("lora" in n.lower() for n, _ in lora_model.named_parameters())
    assert has_lora


def test_lora_changes_output(base_model, lora_model, tokenizer):
    inputs = tokenizer("Привет", return_tensors="pt")
    out_base = base_model.generate(**inputs, max_new_tokens=5)
    out_lora = lora_model.generate(**inputs, max_new_tokens=5)

    assert tokenizer.decode(out_base[0]) != tokenizer.decode(out_lora[0])