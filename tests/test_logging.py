from src.utils.logging import setup_logging

def test_logging_setup():
    logger = setup_logging()
    assert logger is not None
    assert hasattr(logger, "info")