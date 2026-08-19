import sys
from pathlib import Path

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

    if WEAK_MODE:
        logger.warning("Weak mode: skipping training")
        return

    cfg = SFTTrainingConfig(dataset_path=str(DATASET_PATH))

    with cpu_profile("load_tokenizer"):
        tokenizer = load_tokenizer(cfg.model_name)

    with cpu_profile("load_dataset"):
        dataset = prepare_dataset(Path(cfg.dataset_path))

    with cpu_profile("load_4bit_model"):
        model = load_4bit_model(cfg.model_name)

    with cpu_profile("apply_lora"):
        model = apply_lora(model)

    with cpu_profile("create_trainer"):
        trainer = create_trainer(model, tokenizer, dataset, cfg)

    logger.info("Training started")

    try:
        train_out, eval_out = train_model(trainer)
    except Exception as e:
        logger.error(f"Training crashed: {e}")
        sys.exit(1)

    logger.info(f"Train loss: {train_out.training_loss:.4f}")
    logger.info(f"Eval loss: {eval_out['eval_loss']:.4f}")
    logger.info("=== Training completed successfully ===")


if __name__ == "__main__":
    main()