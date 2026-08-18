import sys
import os

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, PROJECT_ROOT)

from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

import pytest
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

COLAB_MODEL_PATH = "/content/drive/MyDrive/Qwen2-0.5B-SFT-MultiDomain"
COLAB_ADAPTER_PATH = "/content/drive/MyDrive/qwen2-0.5b-lora"


@pytest.fixture(scope="session")
def model_path():
    assert os.path.exists(COLAB_MODEL_PATH)
    return COLAB_MODEL_PATH


@pytest.fixture(scope="session")
def adapter_path():
    assert os.path.exists(COLAB_ADAPTER_PATH)
    return COLAB_ADAPTER_PATH


@pytest.fixture(scope="session")
def tokenizer(model_path):
    tok = AutoTokenizer.from_pretrained(model_path)
    return tok


@pytest.fixture(scope="session")
def base_model(model_path):
    model = AutoModelForCausalLM.from_pretrained(model_path)
    model.eval()
    return model


@pytest.fixture(scope="session")
def lora_model(base_model, adapter_path):
    model = PeftModel.from_pretrained(base_model, adapter_path)
    model.eval()
    return model