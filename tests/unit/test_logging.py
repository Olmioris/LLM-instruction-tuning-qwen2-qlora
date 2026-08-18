import logging
import os
from src.utils.logging import setup_logging

def test_setup_logging_creates_logs_dir(tmp_path, monkeypatch):
    # подменяем LOG_DIR на временную директорию
    from src.utils import logging as logging_module
    monkeypatch.setattr(logging_module, "LOG_DIR", tmp_path / "logs")
    monkeypatch.setattr(logging_module, "LOG_FILE", tmp_path / "logs" / "app.log")

    logger = setup_logging()

    assert logger.name == "app"
    assert (tmp_path / "logs").exists()
    assert (tmp_path / "logs" / "app.log").exists()
    assert len(logging.getLogger().handlers) >= 2