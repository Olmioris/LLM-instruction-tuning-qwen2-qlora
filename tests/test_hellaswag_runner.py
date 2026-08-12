import pytest
from src.evaluation.hellaswag_runner import run_hellaswag
from src.training.config import WEAK_MODE

def test_hellaswag_runs():
    if WEAK_MODE:
        pytest.skip("Skipping Hellaswag test in WEAK_MODE")

    score = run_hellaswag("Qwen/Qwen2-0.5B-Instruct", limit=5)
    assert score is not None