import logging
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from src.training.config import WEAK_MODE

def load_tokenizer(model_name: str):
    if WEAK_MODE:
        logger.warning("Weak laptop mode: tokenizer loading skipped")
        return None

    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
    except Exception as e:
        logger.error(f"Tokenizer load failed: {e}")
        raise

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    logger.debug(f"Tokenizer loaded for {model_name}")
    return tokenizer