import logging
from lm_eval import evaluator
from src.training.config import WEAK_MODE

logger = logging.getLogger("app")


def run_hellaswag(model_path: str, adapter_path: str = None, limit: int = 500):
    """
    Run Hellaswag evaluation for baseline or finetuned model.

    Parameters:
        model_path: путь к локальной модели (директория с config.json, tokenizer.json, model.safetensors)
        adapter_path: путь к LoRA адаптеру (если есть)
        limit: ограничение количества примеров

    Returns:
        float: значение метрики acc_norm
    """

    if WEAK_MODE:
        logger.warning("Weak laptop mode: skipping Hellaswag evaluation")
        return None

    # -----------------------------
    # Формирование аргументов модели
    # -----------------------------
    if adapter_path:
        logger.info(f"Running Hellaswag for finetuned model (adapter: {adapter_path})")
        model_args = f"pretrained={model_path},peft={adapter_path},dtype=float32"
    else:
        logger.info("Running Hellaswag for baseline model")
        model_args = f"pretrained={model_path},dtype=float32"

    # -----------------------------
    # Запуск LM‑Eval
    # -----------------------------
    try:
        results = evaluator.simple_evaluate(
            model="hf",
            model_args=model_args,
            tasks=["hellaswag"],
            num_fewshot=0,
            limit=limit,
            batch_size=1,
        )
    except Exception as e:
        logger.error(f"LM‑Eval Hellaswag failed: {e}")
        raise RuntimeError(f"Hellaswag evaluation failed: {e}")

    # -----------------------------
    # Проверка структуры результатов
    # -----------------------------
    if "results" not in results or "hellaswag" not in results["results"]:
        raise RuntimeError("Invalid LM‑Eval output: missing 'results.hellaswag'")

    hella = results["results"]["hellaswag"]

    # -----------------------------
    # Поиск ключа acc_norm
    # -----------------------------
    acc_norm_key = next((k for k in hella.keys() if "acc_norm" in k), None)
    if acc_norm_key is None:
        raise RuntimeError(
            f"acc_norm metric not found in Hellaswag results. Available keys: {list(hella.keys())}"
        )

    score = hella[acc_norm_key]

    # -----------------------------
    # Логирование результата
    # -----------------------------
    logger.info(f"Hellaswag acc_norm: {score:.4f}")

    return score