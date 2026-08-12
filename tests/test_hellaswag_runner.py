from src.evaluation.hellaswag_runner import run_hellaswag

def test_hellaswag_runner():
    model_path = "/content/drive/MyDrive/Qwen2-0.5B-SFT-MultiDomain"
    result = run_hellaswag(model_path)
    assert isinstance(result, float)