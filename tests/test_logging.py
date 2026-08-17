from src.utils.logging import setup_logging
import os

def test_logging_setup():
    logger = setup_logging()
    logger.info("Test log entry")

    with open("logs/app.log", "r", encoding="utf-8") as f:
        content = f.read()

    assert "Test log entry" in content