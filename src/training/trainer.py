import logging
from dataclasses import dataclass
from typing import Optional

from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments
from trl import SFTTrainer
from datasets import load_dataset

from src.training.config import (
    MODEL_NAME,
    OUTPUT_DIR,
    DATA_PATH,
    WEAK_MODE
)

logger = logging.getLogger("app")


@dataclass
class SFTTrainingConfig:
    model_name: str = MODEL_NAME
    data_path: str = DATA_PATH
    output_dir: str = OUTPUT_DIR
    max_seq_length: int = 1024
    num_train_epochs: int = 1
    per_device_train_batch_size: int = 1
    gradient_accumulation_steps: int = 4
    learning_rate: float = 2e-4
    logging_steps: int = 10
    save_steps: int = 200
    warmup_steps: int = 50


def run_sft_training(cfg: Optional[SFTTrainingConfig] = None):
    if WEAK_MODE:
        logger.warning("Weak laptop mode: skipping SFT training")
        return

    cfg = cfg or SFTTrainingConfig()

    logger.info("Loading dataset...")
    dataset = load_dataset("json", data_files=cfg.data_path)

    logger.info("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    logger.info("Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        cfg.model_name,
        device_map="auto",
        torch_dtype="auto"
    )

    logger.info("Preparing training arguments...")
    training_args = TrainingArguments(
        output_dir=cfg.output_dir,
        num_train_epochs=cfg.num_train_epochs,
        per_device_train_batch_size=cfg.per_device_train_batch_size,
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
        learning_rate=cfg.learning_rate,
        logging_steps=cfg.logging_steps,
        save_steps=cfg.save_steps,
        warmup_steps=cfg.warmup_steps,
        fp16=False,
        bf16=False,
        report_to="none"
    )

    logger.info("Initializing SFTTrainer...")
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset["train"],
        dataset_text_field="instruction",
        max_seq_length=cfg.max_seq_length,
        args=training_args,
    )

    logger.info("Starting training...")
    trainer.train()

    logger.info("Saving model...")
    trainer.save_model(cfg.output_dir)

    logger.info("Training completed successfully.")