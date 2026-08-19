from pathlib import Path

MODEL_NAME = "Qwen/Qwen2-0.5B-Instruct"
DATASET_PATH = Path("data/instructions_dataset")
OUTPUT_DIR = Path("models/sft-output")
LORA_ADAPTER_DIR = Path("models/lora-adapter")

WEAK_MODE = True  # ноут / локально → True, в Colab для тестов будем патчить на False

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