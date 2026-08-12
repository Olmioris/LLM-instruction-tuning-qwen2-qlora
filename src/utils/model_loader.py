import logging
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from src.training.config import WEAK_MODE

logger = logging.getLogger("app")

def load_tokenizer(model_name: str):
    if WEAK_MODE:
        logger.warning("Weak laptop mode: tokenizer loading skipped")
        return None

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    logger.debug(f"Tokenizer loaded for {model_name}")
    return tokenizer

def load_baseline_model(model_name: str, device="cpu"):
    if WEAK_MODE:
        logger.warning("Weak laptop mode: baseline model loading skipped")
        return None, None

    logger.info(f"Loading baseline model: {model_name}")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float32,
        device_map=device
    )
    tokenizer = load_tokenizer(model_name)
    return model, tokenizer

def load_finetuned_model(model_name: str, adapter_path: str, device="cpu"):
    if WEAK_MODE:
        logger.warning("Weak laptop mode: finetuned model loading skipped")
        return None, None

    logger.info(f"Loading finetuned model from adapter: {adapter_path}")
    base_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float32,
        device_map=device
    )
    tokenizer = load_tokenizer(model_name)
    model = PeftModel.from_pretrained(base_model, adapter_path)
    model.eval()
    return model, tokenizer