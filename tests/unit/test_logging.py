import logging
from src.utils.logging import setup_logging

def test_logging_initialization(tmp_path, monkeypatch):
    monkeypatch.setattr("src.utils.logging.LOG_DIR", tmp_path / "logs")
    logger = setup_logging()

    assert (tmp_path / "logs").exists()
    assert logger.name == "app"
    assert len(logging.getLogger().handlers) == 2