from src.evaluation.hellaswag_runner import run_hellaswag

def test_hellaswag_runner():
    result = run_hellaswag(model_name="qwen2-0.5b-lora", dataset_path="data/hellaswag_sample.json")
    assert "accuracy" in result