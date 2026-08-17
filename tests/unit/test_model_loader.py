# tests/test_model_loader.py

from src.utils.model_loader import load_local_model
import pytest
from pathlib import Path


def test_load_local_model_raises_on_missing_path(tmp_path):
    # Берём заведомо несуществующий путь
    fake_path = tmp_path / "no_model_here"

    with pytest.raises((FileNotFoundError, ValueError, OSError)):
        load_local_model(str(fake_path))