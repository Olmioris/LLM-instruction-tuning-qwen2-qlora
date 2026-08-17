import pytest

@pytest.mark.skip(reason="LM-Eval is not compatible with Qwen2 models")
def test_hellaswag_runner():
    pass