import logging
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from src.training.config import WEAK_MODE

logger = logging.getLogger("app")


def load_tokenizer(model_name: str):
    tok = AutoTokenizer.from_pretrained(model_name)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return tok


def load_baseline_model(model_name: str):
    if WEAK_MODE:
        logger.warning("Weak mode: baseline model not loaded")
        return None, None

    logger.info(f"Loading baseline model: {model_name}")
    tok = load_tokenizer(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    return model, tok


def load_finetuned_model(model_name: str, adapter_path: str):
    if WEAK_MODE:
        logger.warning("Weak mode: finetuned model not loaded")
        return None, None

    logger.info(f"Loading finetuned model: {model_name} with adapter {adapter_path}")
    tok = load_tokenizer(model_name)
    base_model = AutoModelForCausalLM.from_pretrained(model_name)
    model = PeftModel.from_pretrained(base_model, adapter_path)
    return model, tok