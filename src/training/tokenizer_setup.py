import logging
from transformers import AutoTokenizer
from src.training.config import WEAK_MODE

logger = logging.getLogger("app")

def load_tokenizer(model_name: str):
    if WEAK_MODE:
        logger.warning("Weak laptop mode: tokenizer skipped")
        return None

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    logger.debug(f"Tokenizer loaded: {model_name}")
    return tokenizer