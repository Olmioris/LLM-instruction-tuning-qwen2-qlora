from src.utils.logging import setup_logging
from src.utils.profiling import cpu_profile
from src.utils.model_loader import load_baseline_model, load_finetuned_model
from src.training.config import MODEL_NAME, OUTPUT_DIR


def main():
    logger = setup_logging()
    logger.info("Starting generation demo")

    # Load baseline model
    with cpu_profile("load_baseline"):
        baseline_model, baseline_tokenizer = load_baseline_model(MODEL_NAME)

    # Load finetuned model
    with cpu_profile("load_finetuned"):
        finetuned_model, finetuned_tokenizer = load_finetuned_model(
            MODEL_NAME, OUTPUT_DIR
        )

    prompt = "Give three recommendations for improving customer support in an online bank."

    # Baseline generation
    baseline_out = baseline_model.generate(
        **baseline_tokenizer(prompt, return_tensors="pt"),
        max_new_tokens=150
    )
    logger.info("Baseline output:")
    logger.info(baseline_tokenizer.decode(baseline_out[0], skip_special_tokens=True))

    # Finetuned generation
    finetuned_out = finetuned_model.generate(
        **finetuned_tokenizer(prompt, return_tensors="pt"),
        max_new_tokens=150
    )
    logger.info("Finetuned output:")
    logger.info(finetuned_tokenizer.decode(finetuned_out[0], skip_special_tokens=True))

    logger.info("Generation demo completed")


if __name__ == "__main__":
    main()