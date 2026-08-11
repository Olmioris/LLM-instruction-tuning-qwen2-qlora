import torch
from transformers import AutoModelForCausalLM, BitsAndBytesConfig
from peft import prepare_model_for_kbit_training, LoraConfig, get_peft_model
from trl import SFTTrainer, SFTConfig

from .config import MODEL_NAME, OUTPUT_DIR, LORA_CONFIG, TRAINING_CONFIG

def load_4bit_model():
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float32,
    )

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        quantization_config=bnb_config,
        device_map="cpu",
    )

    model.config.use_cache = False
    return model

def apply_lora(model):
    lora_cfg = LoraConfig(
        r=LORA_CONFIG["r"],
        lora_alpha=LORA_CONFIG["alpha"],
        lora_dropout=LORA_CONFIG["dropout"],
        bias=LORA_CONFIG["bias"],
        task_type="CAUSAL_LM",
        target_modules=LORA_CONFIG["target_modules"],
    )
    model = prepare_model_for_kbit_training(model)
    return get_peft_model(model, lora_cfg)

def create_trainer(model, tokenizer, dataset):
    args = SFTConfig(output_dir=OUTPUT_DIR, **TRAINING_CONFIG)
    trainer = SFTTrainer(
        model=model,
        args=args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"],
        processing_class=tokenizer,
    )
    return trainer

def train_model(trainer):
    train_out = trainer.train()
    eval_out = trainer.evaluate()
    trainer.save_model(OUTPUT_DIR)
    return train_out, eval_out