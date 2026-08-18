def test_model_loads(base_model, tokenizer):
    assert base_model is not None
    assert tokenizer is not None


def test_tokenizer_files_exist(model_path):
    import os
    for fname in ["tokenizer.json", "config.json"]:
        assert os.path.exists(os.path.join(model_path, fname))