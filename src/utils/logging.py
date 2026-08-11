import logging
import os
from logging.handlers import RotatingFileHandler

LOG_DIR = "logs"
LOG_FILE = os.path.join(LOG_DIR, "app.log")

# ANSI color codes
COLORS = {
    "DEBUG": "\033[36m",     # Cyan
    "INFO": "\033[32m",      # Green
    "WARNING": "\033[33m",   # Yellow
    "ERROR": "\033[31m",     # Red
    "CRITICAL": "\033[41m",  # Red background
    "ENDC": "\033[0m",
}

class ColorFormatter(logging.Formatter):
    def format(self, record):
        levelname = record.levelname
        if levelname in COLORS:
            record.levelname = f"{COLORS[levelname]}{levelname}{COLORS['ENDC']}"
        return super().format(record)

def setup_logging(level=None):
    """
    Configure application-wide logging with rotation and colored console output.
    Prevent duplicate handlers and allow log level override via ENV.
    """
    if level is None:
        env_level = os.getenv("LOG_LEVEL", "INFO").upper()
        level = getattr(logging, env_level, logging.INFO)

    os.makedirs(LOG_DIR, exist_ok=True)

    root_logger = logging.getLogger()
    if root_logger.handlers:
        return logging.getLogger("app")

    formatter = logging.Formatter(
        "[%(asctime)s] [%(levelname)s] %(name)s: %(message)s",
        "%Y-%m-%d %H:%M:%S"
    )

    color_formatter = ColorFormatter(
        "[%(asctime)s] [%(levelname)s] %(name)s: %(message)s",
        "%Y-%m-%d %H:%M:%S"
    )

    # File handler
    file_handler = RotatingFileHandler(
        LOG_FILE,
        maxBytes=5_000_000,
        backupCount=5,
        encoding="utf-8"
    )
    file_handler.setFormatter(formatter)

    # Console handler with colors
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(color_formatter)

    logging.basicConfig(level=level, handlers=[file_handler, console_handler])

    logger = logging.getLogger("app")
    logger.info("Logging initialized")
    return logger