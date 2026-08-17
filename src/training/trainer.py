import logging
from dataclasses import dataclass
from typing import Optional

import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments
from peft import LoraConfig, get_peft_model
from trl import SFTTrainer

from src.training.config import (
    MODEL_NAME,
    DATASET_PATH,
    OUTPUT_DIR,
    LORA_CONFIG,
    TRAINING_CONFIG,
    WEAK_MODE,
)

logger = logging.getLogger("app")


# -----------------------------
# Dataclass конфигурации
# -----------------------------
@dataclass
class SFTTrainingConfig:
    model_name: str = MODEL_NAME
    dataset_path: str = DATASET_PATH
    output_dir: str = OUTPUT_DIR
    max_seq_length: int = TRAINING_CONFIG["max_length"]
    per_device_train_batch_size: int = TRAINING_CONFIG["per_device_train_batch_size"]
    gradient_accumulation_steps: int = TRAINING_CONFIG["gradient_accumulation_steps"]
    learning_rate: float = TRAINING_CONFIG["learning_rate"]
    warmup_steps: int = TRAINING_CONFIG["warmup_steps"]
    logging_steps: int = TRAINING_CONFIG["logging_steps"]
    save_steps: int = TRAINING_CONFIG["save_steps"]
    num_train_epochs: int = 1  # TRL SFTTrainer обычно работает по max_steps
    dataset_text_field: str = TRAINING_CONFIG["dataset_text_field"]


# -----------------------------
# Загрузка модели в 4-бит
# -----------------------------
def load_4bit_model(model_name: str = MODEL_NAME):
    if WEAK_MODE:
        logger.warning("Weak laptop mode: skipping 4-bit model loading")
        return None

    logger.info(f"Loading 4-bit model: {model_name}")

    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            load_in_4bit=True,
            torch_dtype=torch.float16,
            device_map="auto",
        )
        return model
    except Exception as e:
        logger.error(f"Failed to load 4-bit model: {e}")
        raise


# -----------------------------
# Применение LoRA
# -----------------------------
def apply_lora(model):
    if WEAK_MODE:
        logger.warning("Weak laptop mode: skipping LoRA application")
        return model

    logger.info("Applying LoRA configuration")

    try:
        lora_cfg = LoraConfig(
            r=LORA_CONFIG["r"],
            lora_alpha=LORA_CONFIG["alpha"],
            lora_dropout=LORA_CONFIG["dropout"],
            bias=LORA_CONFIG["bias"],
            target_modules=LORA_CONFIG["target_modules"],
        )
        model = get_peft_model(model, lora_cfg)
        return model
    except Exception as e:
        logger.error(f"Failed to apply LoRA: {e}")
        raise


# -----------------------------
# Создание токенизатора
# -----------------------------
def load_tokenizer(model_name: str = MODEL_NAME):
    if WEAK_MODE:
        logger.warning("Weak laptop mode: skipping tokenizer loading")
        return None

    logger.info(f"Loading tokenizer: {model_name}")

    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            tokenizer.pad_token_id = tokenizer.eos_token_id
        return tokenizer
    except Exception as e:
        logger.error(f"Failed to load tokenizer: {e}")
        raise


# -----------------------------
# Загрузка датасета
# -----------------------------
def load_training_dataset(path: str):
    if WEAK_MODE:
        logger.warning("Weak laptop mode: skipping dataset loading")
        return None

    logger.info(f"Loading dataset from {path}")

    try:
        dataset = load_dataset("json", data_files=path)
        if "train" not in dataset:
            raise ValueError("Dataset must contain a 'train' split")
        return dataset
    except Exception as e:
        logger.error(f"Failed to load dataset: {e}")
        raise


# -----------------------------
# Создание SFTTrainer
# -----------------------------
def create_trainer(model, tokenizer, dataset, cfg: SFTTrainingConfig):
    if WEAK_MODE:
        logger.warning("Weak laptop mode: skipping trainer creation")
        return None

    logger.info("Preparing training arguments")

    training_args = TrainingArguments(
        output_dir=cfg.output_dir,
        per_device_train_batch_size=cfg.per_device_train_batch_size,
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
        learning_rate=cfg.learning_rate,
        warmup_steps=cfg.warmup_steps,
        logging_steps=cfg.logging_steps,
        save_steps=cfg.save_steps,
        fp16=False,
        bf16=False,
        report_to="none",
        num_train_epochs=cfg.num_train_epochs,
    )

    logger.info("Initializing SFTTrainer")

    try:
        trainer = SFTTrainer(
            model=model,
            tokenizer=tokenizer,
            train_dataset=dataset["train"],
            dataset_text_field=cfg.dataset_text_field,
            max_seq_length=cfg.max_seq_length,
            args=training_args,
        )
        return trainer
    except Exception as e:
        logger.error(f"Failed to initialize SFTTrainer: {e}")
        raise


# -----------------------------
# Обучение модели
# -----------------------------
def train_model(trainer):
    if WEAK_MODE:
        logger.warning("Weak laptop mode: skipping training")
        return None, None

    logger.info("Starting training")

    try:
        train_out = trainer.train()
        eval_out = trainer.evaluate()
        return train_out, eval_out
    except Exception as e:
        logger.error(f"Training failed: {e}")
        raise


# -----------------------------
# Высокоуровневая функцияч
# -----------------------------
def run_sft_training(cfg: Optional[SFTTrainingConfig] = None):
    if WEAK_MODE:
        logger.warning("Weak laptop mode: skipping SFT training")
        return

    cfg = cfg or SFTTrainingConfig()

    tokenizer = load_tokenizer(cfg.model_name)
    dataset = load_training_dataset(cfg.dataset_path)
    model = load_4bit_model(cfg.model_name)
    model = apply_lora(model)

    trainer = create_trainer(model, tokenizer, dataset, cfg)

    train_out, eval_out = train_model(trainer)

    logger.info(f"Training loss: {train_out.training_loss:.4f}")
    logger.info(f"Eval loss: {eval_out['eval_loss']:.4f}")

    logger.info("Training completed successfully")