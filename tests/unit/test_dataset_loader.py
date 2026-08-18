import pytest
from src.data.dataset_loader import set_seed, load_raw_dataset, prepare_dataset
from src.training.config import WEAK_MODE

def test_set_seed_does_not_crash():
    set_seed(42)

@pytest.mark.skipif(WEAK_MODE, reason="Weak mode: dataset loading skipped")
def test_load_raw_dataset_path_not_exists(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_raw_dataset(tmp_path / "missing")

@pytest.mark.skipif(WEAK_MODE, reason="Weak mode: dataset loading skipped")
def test_prepare_dataset_split(tmp_path, monkeypatch):
    # здесь можно замокать load_from_disk, если нужно
    pass  # оставим как заглушку, если нет локального датасета