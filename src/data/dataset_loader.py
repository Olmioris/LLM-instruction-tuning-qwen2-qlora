import random
import numpy as np
import torch
from datasets import load_from_disk

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

def load_raw_dataset(path: str):
    """
    Загружает уже отформатированный датасет,
    который содержит только поле 'text'.
    """
    return load_from_disk(path)

def prepare_dataset(path: str, seed: int = 42):
    """
    Загружает датасет и делает train/test split.
    Никакого форматирования не требуется — оно уже сделано в Colab.
    """
    set_seed(seed)
    dataset = load_raw_dataset(path)
    dataset = dataset.train_test_split(test_size=0.02, seed=seed)
    return dataset