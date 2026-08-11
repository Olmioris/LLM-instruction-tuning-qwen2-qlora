from src.utils.logging import setup_logging

def test_logging_initialization():
    logger = setup_logging()
    logger.info("Test log entry")
    assert logger is not None