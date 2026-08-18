import logging
import os
from src.utils.logging import setup_logging

def test_setup_logging_creates_logs_dir(tmp_path, monkeypatch):
    # подменяем пути логов на временную директорию
    monkeypatch.setattr("src.utils.logging.LOG_DIR", tmp_path / "logs")
    monkeypatch.setattr("src.utils.logging.LOG_FILE", tmp_path / "logs" / "app.log")

    logger = setup_logging()

    # проверяем, что логгер корректно инициализирован
    assert logger.name == "app"

    # директория и файл должны быть созданы
    assert (tmp_path / "logs").exists()
    assert (tmp_path / "logs" / "app.log").exists()

    # должны быть и консольный, и файловый обработчики
    root_handlers = logging.getLogger().handlers
    assert len(root_handlers) >= 2