from src.evaluation.hellaswag_runner import run_hellaswag

def test_hellaswag_runner():
    model_path = "./models/local-qwen"
    result = run_hellaswag(model_path)
    assert isinstance(result, float)