from lm_eval import evaluator

def evaluate_finetuned(model_name: str, adapter_path: str, limit: int = 500):
    results = evaluator.simple_evaluate(
        model="hf",
        model_args=f"pretrained={model_name},peft={adapter_path},dtype=float32",
        tasks=["hellaswag"],
        num_fewshot=0,
        limit=limit,
        batch_size=1,
    )
    return results["results"]["hellaswag"]