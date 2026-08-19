import logging
import os
from lm_eval import evaluator
from src.training.config import WEAK_MODE, MODEL_NAME

logger = logging.getLogger("app")


def run_hellaswag(model_path: str = MODEL_NAME, adapter_path: str = None, limit: int = 500):
    if WEAK_MODE:
        logger.warning("Weak laptop mode: skipping Hellaswag evaluation")
        return None

    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    if adapter_path:
        logger.info(f"Running Hellaswag for finetuned model (adapter: {adapter_path})")
        model_args = f"pretrained={model_path},peft={adapter_path},dtype=float32"
    else:
        logger.info("Running Hellaswag for baseline model")
        model_args = f"pretrained={model_path},dtype=float32"

    try:
        results = evaluator.simple_evaluate(
            model="hf",
            model_args=model_args,
            tasks=["hellaswag"],
            num_fewshot=0,
            limit=limit,
            batch_size=1,
            device="cpu",
            no_torchao=True,
        )
    except Exception as e:
        logger.error(f"LM‑Eval Hellaswag failed: {e}")
        return None

    if "results" not in results or "hellaswag" not in results["results"]:
        logger.error("Invalid LM‑Eval output: missing 'results.hellaswag'")
        return None

    hella = results["results"]["hellaswag"]
    acc_norm_key = next((k for k in hella.keys() if "acc_norm" in k), None)
    if acc_norm_key is None:
        logger.error(f"acc_norm metric not found. Keys: {list(hella.keys())}")
        return None

    score = hella[acc_norm_key]
    logger.info(f"Hellaswag acc_norm: {score:.4f}")
    return score