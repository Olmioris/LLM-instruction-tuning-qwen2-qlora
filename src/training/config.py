import os
import logging

logger = logging.getLogger("app")

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))

LOCAL_MODEL_DIR = "/content/drive/MyDrive/local-qwen"
HF_MODEL_NAME = "Qwen/Qwen2-0.5B-Instruct"

def resolve_model_path():
    if os.path.exists(LOCAL_MODEL_DIR):
        logger.info(f"Using local model: {LOCAL_MODEL_DIR}")
        return LOCAL_MODEL_DIR

    logger.warning(
        f"Local model not found at {LOCAL_MODEL_DIR}. "
        f"Falling back to HuggingFace model: {HF_MODEL_NAME}"
    )
    return HF_MODEL_NAME

MODEL_NAME = resolve_model_path()

LORA_ADAPTER_DIR = "/content/drive/MyDrive/Qwen2-0.5B-SFT-MultiDomain"

DATASET_DIR = os.path.join(BASE_DIR, "data", "instructions_dataset")

def validate_dataset_path():
    if not os.path.exists(DATASET_DIR):
        logger.error(f"Dataset directory not found: {DATASET_DIR}")
        raise FileNotFoundError(
            f"Dataset directory does not exist: {DATASET_DIR}. "
            f"Expected a dataset saved via dataset.save_to_disk()."
        )
    return DATASET_DIR

DATASET_PATH = validate_dataset_path()

WEAK_MODE = True

TRAINING_CONFIG = {
    "max_length": 1024,
    "per_device_train_batch_size": 1,
    "gradient_accumulation_steps": 4,
    "learning_rate": 2e-4,
    "warmup_steps": 50,
    "logging_steps": 10,
    "save_steps": 200,
    "dataset_text_field": "text",
}

LORA_CONFIG = {
    "r": 8,
    "alpha": 16,
    "dropout": 0.05,
    "bias": "none",
    "target_modules": ["q_proj", "v_proj"],
}