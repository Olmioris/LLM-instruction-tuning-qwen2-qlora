WEAK_MODE = False
MODEL_NAME = "Qwen/Qwen2-0.5B-Instruct"
MAX_SEQ_LENGTH = 512
SEED = 42

DATASET_PATH = "/content/drive/MyDrive/MultiDomain_Instruction_50k"
OUTPUT_DIR = "/content/drive/MyDrive/Qwen2-0.5B-SFT-MultiDomain"

LORA_CONFIG = {
    "r": 8,
    "alpha": 16,
    "dropout": 0.05,
    "bias": "none",
    "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"],
}

TRAINING_CONFIG = {
    "per_device_train_batch_size": 1,
    "gradient_accumulation_steps": 4,
    "learning_rate": 2e-4,
    "warmup_steps": 20,
    "max_steps": 80,
    "logging_steps": 10,
    "eval_strategy": "steps",
    "eval_steps": 40,
    "save_strategy": "steps",
    "save_steps": 40,
    "save_total_limit": 1,
    "load_best_model_at_end": True,
    "metric_for_best_model": "eval_loss",
    "greater_is_better": False,
    "optim": "adamw_torch",
    "fp16": False,
    "bf16": False,
    "report_to": "none",
    "seed": SEED,
    "dataset_text_field": "text",
    "max_length": MAX_SEQ_LENGTH,
    "packing": False,
}