import os
import logging
import random
import numpy as np
import torch
from datasets import load_from_disk
from src.training.config import WEAK_MODE

logger = logging.getLogger("app")


# -----------------------------
# Установка seed
# -----------------------------
def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    logger.debug(f"Seed set to {seed}")


# -----------------------------
# Загрузка датасета с диска
# -----------------------------
def load_raw_dataset(path: str):
    """
    Загружает уже отформатированный датасет,
    который содержит только поле 'text'.
    """
    if WEAK_MODE:
        logger.warning("Weak laptop mode: skipping dataset loading")
        return None

    logger.info(f"Loading dataset from: {path}")

    if not os.path.exists(path):
        raise FileNotFoundError(f"Dataset path does not exist: {path}")

    try:
        dataset = load_from_disk(path)
    except Exception as e:
        logger.error(f"Failed to load dataset from disk: {e}")
        raise RuntimeError(f"Dataset loading failed: {e}")

    # Проверка структуры
    if "text" not in dataset.column_names:
        logger.warning(
            f"Dataset loaded but 'text' field not found. Columns: {dataset.column_names}"
        )

    logger.info("Dataset loaded successfully")
    return dataset


# -----------------------------
# Подготовка датасета
# -----------------------------
def prepare_dataset(path: str, seed: int = 42, test_size: float = 0.02):
    """
    Загружает датасет и делает train/test split.
    Форматирования не требуется — оно было сделано заранее в Colab.
    """
    if WEAK_MODE:
        logger.warning("Weak laptop mode: skipping dataset preparation")
        return None

    logger.info("Preparing dataset...")

    set_seed(seed)

    dataset = load_raw_dataset(path)

    try:
        dataset = dataset.train_test_split(test_size=test_size, seed=seed)
    except Exception as e:
        logger.error(f"Failed to split dataset: {e}")
        raise RuntimeError(f"Dataset split failed: {e}")

    if "train" not in dataset or "test" not in dataset:
        raise RuntimeError("Dataset split did not produce 'train' and 'test' subsets")

    logger.info(
        f"Dataset prepared: train={len(dataset['train'])}, test={len(dataset['test'])}"
    )

    return dataset