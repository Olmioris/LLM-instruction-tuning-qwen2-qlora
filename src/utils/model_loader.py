import logging
import os
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from src.training.config import WEAK_MODE, MODEL_NAME, LORA_ADAPTER_DIR

logger = logging.getLogger("app")


def validate_model_directory(model_path: str):
    required = ["model.safetensors", "config.json", "tokenizer.json"]
    missing = [f for f in required if not os.path.exists(os.path.join(model_path, f))]

    if missing:
        logger.error(f"Model directory '{model_path}' missing: {missing}")
        raise FileNotFoundError(f"Incomplete model directory: {missing}")

    logger.debug(f"Model directory validated: {model_path}")


def load_tokenizer(model_name: str = MODEL_NAME):
    if WEAK_MODE:
        logger.warning("Weak mode: tokenizer not loaded")
        return None

    logger.info(f"Loading tokenizer: {model_name}")

    tok = AutoTokenizer.from_pretrained(model_name)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
        tok.pad_token_id = tok.eos_token_id
    return tok


def load_baseline_model(model_name: str = MODEL_NAME):
    if WEAK_MODE:
        logger.warning("Weak mode: baseline model not loaded")
        return None, None

    logger.info(f"Loading baseline model: {model_name}")

    if os.path.isdir(model_name):
        validate_model_directory(model_name)

    tok = load_tokenizer(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto")
    return model, tok


def load_finetuned_model(model_name: str = MODEL_NAME, adapter_path: str = LORA_ADAPTER_DIR):
    if WEAK_MODE:
        logger.warning("Weak mode: finetuned model not loaded")
        return None, None

    if not os.path.exists(adapter_path):
        logger.error(f"Adapter path not found: {adapter_path}")
        raise FileNotFoundError(f"Adapter missing: {adapter_path}")

    logger.info(f"Loading finetuned model: {model_name}")
    logger.info(f"Applying adapter: {adapter_path}")

    if os.path.isdir(model_name):
        validate_model_directory(model_name)

    tok = load_tokenizer(model_name)
    base = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto")
    model = PeftModel.from_pretrained(base, adapter_path)
    return model, tok