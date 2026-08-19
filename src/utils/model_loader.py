import logging
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from src.training.config import WEAK_MODE, MODEL_NAME, LORA_ADAPTER_DIR

logger = logging.getLogger("app")


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

    tok = load_tokenizer(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto")
    return model, tok


def load_finetuned_model(model_name: str = MODEL_NAME, adapter_path: str = str(LORA_ADAPTER_DIR)):
    if WEAK_MODE:
        logger.warning("Weak mode: finetuned model not loaded")
        return None, None

    logger.info(f"Loading finetuned model: {model_name}")
    logger.info(f"Applying adapter: {adapter_path}")

    tok = load_tokenizer(model_name)
    base = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto")
    model = PeftModel.from_pretrained(base, adapter_path)
    return model, tok