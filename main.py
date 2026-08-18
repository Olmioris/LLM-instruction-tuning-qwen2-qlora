from src.utils.logging import setup_logging
from src.utils.model_loader import load_baseline_model, load_finetuned_model
from src.utils.profiling import cpu_profile
from src.evaluation.hellaswag_runner import run_hellaswag
from src.training.config import MODEL_NAME, LORA_ADAPTER_DIR, WEAK_MODE


def main():
    logger = setup_logging()
    logger.info("Starting main pipeline")

    if WEAK_MODE:
        logger.warning("Weak laptop mode: skipping main pipeline")
        return

    with cpu_profile("load_baseline"):
        baseline_model, baseline_tokenizer = load_baseline_model(MODEL_NAME)

    with cpu_profile("load_finetuned"):
        finetuned_model, finetuned_tokenizer = load_finetuned_model(
            MODEL_NAME, LORA_ADAPTER_DIR
        )

    baseline_score = run_hellaswag(MODEL_NAME)
    finetuned_score = run_hellaswag(MODEL_NAME, LORA_ADAPTER_DIR)

    logger.info(f"Baseline Hellaswag acc_norm: {baseline_score:.4f}")
    logger.info(f"Finetuned Hellaswag acc_norm: {finetuned_score:.4f}")

    prompt = "Explain the difference between supervised and reinforcement learning."

    baseline_out = baseline_model.generate(
        **baseline_tokenizer(prompt, return_tensors="pt"),
        max_new_tokens=150
    )
    finetuned_out = finetuned_model.generate(
        **finetuned_tokenizer(prompt, return_tensors="pt"),
        max_new_tokens=150
    )

    logger.info("Baseline generation:")
    logger.info(baseline_tokenizer.decode(baseline_out[0], skip_special_tokens=True))

    logger.info("Finetuned generation:")
    logger.info(finetuned_tokenizer.decode(finetuned_out[0], skip_special_tokens=True))


if __name__ == "__main__":
    main()