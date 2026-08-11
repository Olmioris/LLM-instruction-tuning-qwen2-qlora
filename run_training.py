from src.utils.logging import setup_logging
from src.utils.profiling import cpu_profile
from src.utils.model_loader import load_tokenizer

from src.training.trainer import (
    load_4bit_model,
    apply_lora,
    create_trainer,
    train_model
)

from src.data.dataset_loader import prepare_dataset
from src.training.config import MODEL_NAME, DATASET_PATH


def main():
    logger = setup_logging()
    logger.info("Starting training pipeline")

    # Load tokenizer
    with cpu_profile("load_tokenizer"):
        tokenizer = load_tokenizer(MODEL_NAME)

    # Load dataset
    with cpu_profile("load_dataset"):
        dataset = prepare_dataset(DATASET_PATH)

    # Load 4-bit model
    with cpu_profile("load_4bit_model"):
        model = load_4bit_model()

    # Apply LoRA
    with cpu_profile("apply_lora"):
        model = apply_lora(model)

    # Create trainer
    with cpu_profile("create_trainer"):
        trainer = create_trainer(model, tokenizer, dataset)

    # Train
    logger.info("Training started")
    train_out, eval_out = train_model(trainer)

    logger.info(f"Train loss: {train_out.training_loss:.4f}")
    logger.info(f"Eval loss: {eval_out['eval_loss']:.4f}")

    logger.info("Training completed successfully")


if __name__ == "__main__":
    main()