from src.utils.logging import setup_logging
from src.utils.profiling import cpu_profile
from src.evaluation.hellaswag_runner import run_hellaswag
from src.training.config import MODEL_NAME, OUTPUT_DIR


def main():
    logger = setup_logging()
    logger.info("Starting evaluation pipeline")

    # Baseline evaluation
    with cpu_profile("eval_baseline"):
        baseline_score = run_hellaswag(MODEL_NAME)

    # Finetuned evaluation
    with cpu_profile("eval_finetuned"):
        finetuned_score = run_hellaswag(MODEL_NAME, OUTPUT_DIR)

    logger.info(f"Baseline Hellaswag acc_norm: {baseline_score:.4f}")
    logger.info(f"Finetuned Hellaswag acc_norm: {finetuned_score:.4f}")

    logger.info("Evaluation completed successfully")


if __name__ == "__main__":
    main()