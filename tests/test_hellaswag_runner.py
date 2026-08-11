from src.evaluation.hellaswag_runner import run_hellaswag

def test_hellaswag_runs():
    score = run_hellaswag("Qwen/Qwen2-0.5B-Instruct", limit=5)
    assert isinstance(score, float)