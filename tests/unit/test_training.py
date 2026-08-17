from src.training.config import WEAK_MODE

def test_weak_mode_flag():
    assert isinstance(WEAK_MODE, bool)