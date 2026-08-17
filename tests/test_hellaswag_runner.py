from src.evaluation.hellaswag_runner import run_hellaswag

def test_hellaswag_runner():
    result = run_hellaswag("./models/local-qwen")
    assert isinstance(result, float)