# QLoRA Instruction‑Tuning Pipeline for Qwen2‑0.5B‑Instruct

A fully reproducible, engineering‑grade project for instruction‑tuning a compact LLM using **QLoRA**.

---

## Overview
This repository contains a complete training and evaluation workflow for instruction‑tuning **Qwen2‑0.5B‑Instruct** with **QLoRA**.  
The project demonstrates how a small‑scale model can be adapted to multi‑domain instruction tasks efficiently and reproducibly.

**Key features:**
- Structured data preparation and tokenizer alignment  
- Reproducible QLoRA training pipeline  
- Baseline vs. finetuned evaluation  
- Generative comparison on Russian and English prompts  
- Modular engineering architecture  
- Production‑ready code organization  

---

## Project Structure

```
LLM-instruction-tuning-qwen2-qlora/
│
├── src/
│   ├── utils/
│   │   ├── logging.py              # логирование
│   │   ├── model_loader.py         # загрузка baseline/finetuned моделей
│   │   └── profiling.py            # профилирование
│   │
│   ├── training/
│   │   ├── trainer.py              # обучение SFT (обновлено под TRL 0.7.4)
│   │   ├── config.py               # конфигурация путей, WEAK_MODE
│   │   └── tokenizer_setup.py      # настройка токенизатора
│   │
│   ├── evaluation/
│   │   └── hellaswag_runner.py     # оценка на Hellaswag
│   │
│   └── data/
│       └── dataset_loader.py       # загрузка датасетов
│
├── data/
│   └── example_instructions.json   # примеры инструкций для SFT
│
├── models/
│   └── qwen2-0.5b-lora/
│       ├── adapter_config.json     # конфиг LoRA
│       └── adapter_model.safetensors # веса LoRA
│
├── results/
│   ├── baseline_vs_finetuned.md    # сравнение моделей
│   └── hellaswag_scores.json       # результаты Hellaswag
│
├── conf/
│   └── config.yaml                 # Hydra конфигурация
│
├── notebooks/
│   ├── 01_sft_training_qwen2.ipynb
│   ├── 02_baseline_evaluation.ipynb
│   └── 03_finetuned_evaluation.ipynb
│
├── run_training.py                 # запуск обучения
├── run_eval.py                     # запуск оценки
├── run_generation.py               # генерация текста
├── hydra_main.py                   # запуск Hydra
│
├── requirements.txt                # зависимости
├── Makefile                        # команды (Linux/macOS)
└── README.md
```

---

## Architectural diagram

                          ┌──────────────────────────────┐
                          │        run_training.py        │
                          └───────────────┬───────────────┘
                                          │
                                          ▼
                          ┌──────────────────────────────┐
                          │      src/training/trainer.py │
                          └───────────────┬──────────────┘
                                          │
                                          ▼
                          ┌──────────────────────────────┐
                          │   src/utils/model_loader.py   │
                          └───────────────┬──────────────┘
                                          │
                                          ▼
                          ┌──────────────────────────────┐
                          │   Qwen2-0.5B + LoRA Adapter   │
                          └──────────────────────────────┘


┌──────────────────────────────┐
│        run_eval.py           │
└───────────────┬──────────────┘
                │
                ▼
┌──────────────────────────────┐
│ src/evaluation/hellaswag     │
└───────────────┬──────────────┘
                │
                ▼
┌──────────────────────────────┐
│         lm-eval              │
└──────────────────────────────┘


┌──────────────────────────────┐
│      run_generation.py       │
└───────────────┬──────────────┘
                │
                ▼
┌──────────────────────────────┐
│   src/utils/model_loader.py   │
└──────────────────────────────┘


┌──────────────────────────────┐
│        hydra_main.py         │
└───────────────┬──────────────┘
                │
                ▼
┌──────────────────────────────┐
│         conf/config.yaml      │
└──────────────────────────────┘


┌──────────────────────────────┐
│         tests/*.py           │
└──────────────────────────────┘


┌──────────────────────────────┐
│         data/*.json          │
└──────────────────────────────┘


┌──────────────────────────────┐
│         models/LoRA          │
└──────────────────────────────┘

---

## Training Pipeline (QLoRA)

This repository demonstrates a complete LLM workflow:

- preparing data for SFT (Supervised Fine‑Tuning)
- training the model using QLoRA
- saving the LoRA adapter
- evaluating baseline and finetuned models on Hellaswag
- text generation
- CPU/PyTorch profiling
- structured logging
- Hydra configuration
- pytest tests
- Makefile (Linux/macOS)
- **WEAK_MODE** for low‑power laptops

---

## Usage
```
python run_training.py
python run_eval.py
python run_generation.py
python hydra_main.py
python -m pytest -q
```

---

## WEAK_MODE

In src/training/config.py: **WEAK_MODE = True**

This mode:

 - skips model loading

 - skips LoRA loading

 - skips training

 - skips generation

 - skips evaluation

But keeps the entire pipeline functional.

---

## Example Results

| Model | acc_norm |
| --- | --- |
| Baseline | 0.42 |
| Finetuned | 0.61 |