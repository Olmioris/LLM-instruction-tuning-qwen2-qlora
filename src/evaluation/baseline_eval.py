from lm_eval import evaluator
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

def evaluate_baseline(model_name: str, limit: int = 500):
    results = evaluator.simple_evaluate(
        model="hf",
        model_args=f"pretrained={model_name},dtype=float32",
        tasks=["hellaswag"],
        num_fewshot=0,
        limit=limit,
        batch_size=1,
    )
    return results["results"]["hellaswag"]