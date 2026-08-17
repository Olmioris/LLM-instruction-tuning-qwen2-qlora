import logging
import os
from logging.handlers import RotatingFileHandler

LOG_DIR = "logs"
LOG_FILE = os.path.join(LOG_DIR, "app.log")

def setup_logging(level=logging.INFO):
    # Создаём директорию
    os.makedirs(LOG_DIR, exist_ok=True)

    formatter = logging.Formatter(
        "[%(asctime)s] [%(levelname)s] %(name)s: %(message)s",
        "%Y-%m-%d %H:%M:%S"
    )

    handler = RotatingFileHandler(
        LOG_FILE,
        maxBytes=5_000_000,
        backupCount=5,
        encoding="utf-8"
    )
    handler.setFormatter(formatter)

    console = logging.StreamHandler()
    console.setFormatter(formatter)

    # ВАЖНО: сбрасываем старую конфигурацию
    logging.getLogger().handlers.clear()

    logging.basicConfig(
        level=level,
        handlers=[handler, console]
    )

    logger = logging.getLogger("app")
    return logger