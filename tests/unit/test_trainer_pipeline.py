import pytest
from src.training.config import WEAK_MODE
from src.training.trainer import SFTTrainingConfig, load_4bit_model

@pytest.mark.skipif(WEAK_MODE, reason="Weak mode: trainer pipeline skipped")
def test_trainer_initializes_and_model_loads():
    cfg = SFTTrainingConfig()
    assert cfg.model_name
    assert cfg.dataset_path
    assert cfg.output_dir

    model = load_4bit_model()
    assert model is not None