import os
import logging
import random
import numpy as np
import torch
from datasets import load_from_disk
from src.training.config import WEAK_MODE, DATASET_PATH

logger = logging.getLogger("app")


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    logger.debug(f"Seed set to {seed}")


def validate_dataset_structure(dataset):
    required = "text"
    if required not in dataset.column_names:
        logger.error(
            f"Dataset missing required field '{required}'. Columns: {dataset.column_names}"
        )
        raise ValueError(
            f"Dataset must contain field '{required}', but columns are: {dataset.column_names}"
        )
    logger.debug("Dataset structure validated: field 'text' present")


def load_raw_dataset(path: str = DATASET_PATH):
    if WEAK_MODE:
        logger.warning("Weak laptop mode: skipping dataset loading")
        return None

    logger.info(f"Loading dataset from: {path}")

    if not os.path.exists(path):
        logger.error(f"Dataset path does not exist: {path}")
        raise FileNotFoundError(
            f"Dataset directory does not exist: {path}. Expected dataset.save_to_disk()."
        )

    try:
        dataset = load_from_disk(path)
    except Exception as e:
        logger.error(f"Failed to load dataset: {e}")
        raise RuntimeError(f"Dataset loading failed: {e}")

    validate_dataset_structure(dataset)

    logger.info("Dataset loaded successfully")
    return dataset


def prepare_dataset(path: str = DATASET_PATH, seed: int = 42, test_size: float = 0.02):
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
        logger.error("Dataset split missing 'train' or 'test'")
        raise RuntimeError("Dataset split failed: missing required subsets")

    logger.info(f"Dataset prepared: train={len(dataset['train'])}, test={len(dataset['test'])}")
    return dataset