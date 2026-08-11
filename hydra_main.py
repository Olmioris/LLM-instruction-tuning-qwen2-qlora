import hydra
from omegaconf import DictConfig
from src.utils.logging import setup_logging
from src.utils.model_loader import load_baseline_model


@hydra.main(config_path="conf", config_name="config", version_base=None)
def main(cfg: DictConfig):
    logger = setup_logging()
    logger.info("Hydra configuration loaded")
    logger.info(cfg)

    model, tokenizer = load_baseline_model(cfg.model.name)
    logger.info("Baseline model loaded successfully")

    prompt = cfg.generation.prompt
    out = model.generate(
        **tokenizer(prompt, return_tensors="pt"),
        max_new_tokens=cfg.generation.max_new_tokens
    )

    logger.info("Generated text:")
    logger.info(tokenizer.decode(out[0], skip_special_tokens=True))


if __name__ == "__main__":
    main()