from src.evaluation.hellaswag_runner import run_hellaswag
import json

def test_hellaswag_runner():
    dataset = json.load(open("data/hellaswag_sample.json"))
    result = run_hellaswag("/content/drive/MyDrive/Qwen2-0.5B-SFT-MultiDomain", dataset)
    assert "accuracy" in result