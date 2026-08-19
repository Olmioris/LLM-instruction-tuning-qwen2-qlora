from src.utils.logging import setup_logging
from src.utils.profiling import cpu_profile
from src.evaluation.hellaswag_runner import run_hellaswag
from src.training.config import MODEL_NAME, LORA_ADAPTER_DIR, WEAK_MODE


def main():
    logger = setup_logging()

    if WEAK_MODE:
        logger.warning("Weak mode: skipping evaluation")
        return

    with cpu_profile("eval_baseline"):
        baseline_score = run_hellaswag(MODEL_NAME)

    with cpu_profile("eval_finetuned"):
        finetuned_score = run_hellaswag(MODEL_NAME, LORA_ADAPTER_DIR)

    logger.info(f"Baseline acc_norm: {baseline_score}")
    logger.info(f"Finetuned acc_norm: {finetuned_score}")


if __name__ == "__main__":
    main()