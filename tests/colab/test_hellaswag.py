from src.evaluation.hellaswag_runner import run_hellaswag

def test_hellaswag_runs_on_cpu(model_path):
    score = run_hellaswag(model_path, limit=10)
    assert isinstance(score, float)
    assert 0.0 <= score <= 1.0