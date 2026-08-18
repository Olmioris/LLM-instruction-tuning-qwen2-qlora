import time
import torch

def test_generation_speed(base_model, tokenizer):
    inputs = tokenizer("Привет", return_tensors="pt")
    start = time.time()
    base_model.generate(**inputs, max_new_tokens=10)
    end = time.time()
    assert (end - start) < 3.0


def test_memory_stability(base_model, tokenizer):
    if not torch.cuda.is_available():
        import pytest
        pytest.skip("CUDA not available in Colab CPU runtime")

    inputs = tokenizer("Привет", return_tensors="pt")

    torch.cuda.reset_peak_memory_stats()
    base_model.generate(**inputs, max_new_tokens=10)
    mem1 = torch.cuda.max_memory_allocated()

    base_model.generate(**inputs, max_new_tokens=10)
    mem2 = torch.cuda.max_memory_allocated()

    assert mem2 < mem1 * 1.5