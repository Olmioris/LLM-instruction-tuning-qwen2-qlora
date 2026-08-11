import logging
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

logger = logging.getLogger("app")

def load_tokenizer(model_name: str):
    """
    Load tokenizer and ensure pad_token is set.
    """
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            tokenizer.pad_token_id = tokenizer.eos_token_id
        logger.debug(f"Tokenizer loaded for {model_name}")
        return tokenizer
    except Exception as e:
        logger.error(f"Failed to load tokenizer: {e}")
        raise

def load_baseline_model(model_name: str, device="cpu"):
    """
    Load baseline model for inference.
    """
    logger.info(f"Loading baseline model: {model_name}")
    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float32,
            device_map=device
        )
        tokenizer = load_tokenizer(model_name)
        logger.info("Baseline model loaded successfully")
        return model, tokenizer
    except Exception as e:
        logger.error(f"Failed to load baseline model: {e}")
        raise

def load_finetuned_model(model_name: str, adapter_path: str, device="cpu"):
    """
    Load finetuned model with LoRA adapters.
    """
    logger.info(f"Loading finetuned model from adapter: {adapter_path}")
    try:
        base_model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float32,
            device_map=device
        )
        tokenizer = load_tokenizer(model_name)
        model = PeftModel.from_pretrained(base_model, adapter_path)
        model.eval()
        logger.info("Finetuned model loaded successfully")
        return model, tokenizer
    except Exception as e:
        logger.error(f"Failed to load finetuned model: {e}")
        raise