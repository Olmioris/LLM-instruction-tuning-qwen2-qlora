from src.utils.logging import setup_logging
from src.utils.profiling import cpu_profile
from src.evaluation.hellaswag_runner import run_hellaswag
from src.training.config import MODEL_NAME, OUTPUT_DIR, WEAK_MODE


def main():
    logger = setup_logging()
    logger.info("Starting evaluation pipeline")

    if WEAK_MODE:
        logger.warning("Weak laptop mode: skipping Hellaswag evaluation")
        return

    with cpu_profile("eval_baseline"):
        baseline_score = run_hellaswag(MODEL_NAME)

    with cpu_profile("eval_finetuned"):
        finetuned_score = run_hellaswag(MODEL_NAME, OUTPUT_DIR)

    logger.info(f"Baseline Hellaswag acc_norm: {baseline_score:.4f}")
    logger.info(f"Finetuned Hellaswag acc_norm: {finetuned_score:.4f}")


if __name__ == "__main__":
    main()