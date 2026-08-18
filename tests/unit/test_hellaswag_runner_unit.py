import pytest
from src.evaluation.hellaswag_runner import run_hellaswag
from src.training.config import MODEL_NAME, WEAK_MODE

@pytest.mark.skipif(WEAK_MODE, reason="Weak mode: Hellaswag skipped")
def test_run_hellaswag_baseline_runs():
    score = run_hellaswag(MODEL_NAME, limit=10)
    assert isinstance(score, float)