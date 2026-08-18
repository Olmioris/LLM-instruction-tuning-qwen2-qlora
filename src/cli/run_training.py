import sys
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
from src.training.config import WEAK_MODE, DATASET_PATH


def main():
    logger = setup_logging()
    logger.info("=== Starting training pipeline ===")

    if WEAK_MODE:
        logger.warning("Weak laptop mode: skipping training pipeline")
        return

    # Конфигурация обучения
    cfg = SFTTrainingConfig(dataset_path=DATASET_PATH)

    logger.info(f"Model path: {cfg.model_name}")
    logger.info(f"Dataset path: {cfg.dataset_path}")
    logger.info(f"Output dir: {cfg.output_dir}")

    # -----------------------------
    # Load tokenizer
    # -----------------------------
    with cpu_profile("load_tokenizer"):
        tokenizer = load_tokenizer(cfg.model_name)

    if tokenizer is None:
        logger.error("Tokenizer failed to load. Aborting.")
        sys.exit(1)

    # -----------------------------
    # Load dataset
    # -----------------------------
    with cpu_profile("load_dataset"):
        dataset = prepare_dataset(cfg.dataset_path)

    if dataset is None:
        logger.error("Dataset failed to load. Aborting.")
        sys.exit(1)

    # -----------------------------
    # Load 4-bit model
    # -----------------------------
    with cpu_profile("load_4bit_model"):
        model = load_4bit_model(cfg.model_name)

    if model is None:
        logger.error("Model failed to load. Aborting.")
        sys.exit(1)

    # -----------------------------
    # Apply LoRA
    # -----------------------------
    with cpu_profile("apply_lora"):
        model = apply_lora(model)

    # -----------------------------
    # Create trainer
    # -----------------------------
    with cpu_profile("create_trainer"):
        trainer = create_trainer(model, tokenizer, dataset, cfg)

    if trainer is None:
        logger.error("Trainer failed to initialize. Aborting.")
        sys.exit(1)

    # -----------------------------
    # Train
    # -----------------------------
    logger.info("Training started")

    try:
        train_out, eval_out = train_model(trainer)
    except Exception as e:
        logger.error(f"Training crashed: {e}")
        sys.exit(1)

    if train_out is None or eval_out is None:
        logger.error("Training returned empty results. Aborting.")
        sys.exit(1)

    logger.info(f"Train loss: {train_out.training_loss:.4f}")
    logger.info(f"Eval loss: {eval_out['eval_loss']:.4f}")
    logger.info("=== Training completed successfully ===")


if __name__ == "__main__":
    main()