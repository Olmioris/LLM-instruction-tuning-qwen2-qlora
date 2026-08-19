import random
import numpy as np
import torch
from datasets import load_from_disk
from pathlib import Path

from src.training.config import WEAK_MODE, DATASET_PATH


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def validate_dataset_structure(dataset):
    if "text" not in dataset.column_names:
        raise ValueError(
            f"Dataset must contain field 'text', but columns are: {dataset.column_names}"
        )


def load_raw_dataset(path: Path = DATASET_PATH):
    if WEAK_MODE:
        return None

    dataset = load_from_disk(str(path))
    validate_dataset_structure(dataset)
    return dataset


def prepare_dataset(path: Path = DATASET_PATH, seed: int = 42, test_size: float = 0.02):
    if WEAK_MODE:
        return None

    set_seed(seed)
    dataset = load_raw_dataset(path)
    dataset = dataset.train_test_split(test_size=test_size, seed=seed)
    return dataset