import pytest
from src.training.trainer import SFTTrainingConfig, load_4bit_model
from src.training.config import WEAK_MODE

def test_sft_training_config_defaults():
    cfg = SFTTrainingConfig()
    assert cfg.model_name
    assert cfg.dataset_path
    assert cfg.output_dir

@pytest.mark.skipif(WEAK_MODE, reason="Weak mode: model loading skipped")
def test_load_4bit_model_runs():
    model = load_4bit_model()
    assert model is not None