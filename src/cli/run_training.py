from src.utils.logging import setup_logging
from src.utils.profiling import cpu_profile
from src.data.dataset_loader import prepare_dataset
from src.training.trainer import (
    load_4bit_model,
    apply_lora,
    create_trainer,
    train_model,
    load_tokenizer,
    SFTTrainingConfig,
)
from src.training.config import WEAK_MODE


def main():
    logger = setup_logging()
    logger.info("Starting training pipeline")

    if WEAK_MODE:
        logger.warning("Weak laptop mode: skipping training pipeline")
        return

    cfg = SFTTrainingConfig()

    # Load tokenizer
    with cpu_profile("load_tokenizer"):
        tokenizer = load_tokenizer(cfg.model_name)

    # Load dataset
    with cpu_profile("load_dataset"):
        dataset = prepare_dataset(cfg.dataset_path)

    # Load 4-bit model
    with cpu_profile("load_4bit_model"):
        model = load_4bit_model(cfg.model_name)

    # Apply LoRA
    with cpu_profile("apply_lora"):
        model = apply_lora(model)

    # Create trainer
    with cpu_profile("create_trainer"):
        trainer = create_trainer(model, tokenizer, dataset, cfg)

    # Train
    logger.info("Training started")
    train_out, eval_out = train_model(trainer)

    logger.info(f"Train loss: {train_out.training_loss:.4f}")
    logger.info(f"Eval loss: {eval_out['eval_loss']:.4f}")
    logger.info("Training completed successfully")


if __name__ == "__main__":
    main()