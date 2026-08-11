import logging
from lm_eval import evaluator

logger = logging.getLogger("app")

def run_hellaswag(model_name: str, adapter_path: str = None, limit: int = 500):
    """
    Run Hellaswag evaluation for baseline or finetuned model.
    """
    if adapter_path:
        logger.info("Running Hellaswag for finetuned model")
        model_args = f"pretrained={model_name},peft={adapter_path},dtype=float32"
    else:
        logger.info("Running Hellaswag for baseline model")
        model_args = f"pretrained={model_name},dtype=float32"

    results = evaluator.simple_evaluate(
        model="hf",
        model_args=model_args,
        tasks=["hellaswag"],
        num_fewshot=0,
        limit=limit,
        batch_size=1,
    )

    hella = results["results"]["hellaswag"]
    acc_norm_key = next(k for k in hella.keys() if "acc_norm" in k)
    score = hella[acc_norm_key]

    logger.info(f"Hellaswag acc_norm: {score:.4f}")
    return score