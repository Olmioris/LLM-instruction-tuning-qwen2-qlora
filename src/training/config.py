import os

# -----------------------------
# Базовая директория проекта
# -----------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))

# -----------------------------
# Пути к модели
# -----------------------------
# Для Colab — локальная модель, которую ты копируешь в ./models/local-qwen
MODEL_NAME = os.path.join(BASE_DIR, "models", "local-qwen")
OUTPUT_DIR = MODEL_NAME

# -----------------------------
# Пути к датасету
# -----------------------------
DATASET_PATH = os.path.join(BASE_DIR, "data", "example_instructions.json")

# -----------------------------
# Режим работы
# -----------------------------
# WEAK_MODE=False → полноценный режим (Colab)
# WEAK_MODE=True → режим ноутбука (CPU, без CUDA, без тяжёлых тестов)
WEAK_MODE = False

# -----------------------------
# Конфигурация обучения
# -----------------------------
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

# -----------------------------
# Конфигурация LoRA
# -----------------------------
LORA_CONFIG = {
    "r": 8,
    "alpha": 16,
    "dropout": 0.05,
    "bias": "none",
    "target_modules": ["q_proj", "v_proj"],
}