# LLM-instruction-tuning-qwen2-qlora

Полный ML‑пайплайн для обучения, оценки и генерации на базе модели **Qwen2‑0.5B‑Instruct**, включая LoRA/QLoRA, Hydra‑конфигурации, профилирование, логирование, тесты и удобные точки входа.

---

## Описание проекта

Этот репозиторий демонстрирует полный цикл работы с LLM:

- подготовка данных для SFT (Supervised Fine‑Tuning)
- обучение модели с использованием QLoRA
- сохранение адаптера LoRA
- оценка baseline и finetuned модели на Hellaswag
- генерация текста
- профилирование CPU/PyTorch
- логирование (цветное, структурированное)
- Hydra‑конфигурации
- pytest‑тесты
- Makefile (для Linux/macOS)
- режим **WEAK_MODE** для слабых ноутбуков

---

## Структура проекта

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

## Установка зависимостей

```
pip install -r requirements.txt
```
## Запуск проекта

### Обучение (в облаке или на сервере)
```
python run_training.py
```

### Оценка
```
python run_eval.py
```

### Генерация текста
```
python run_generation.py
```

### Hydra
```
python hydra_main.py
```

### Тесты
```
python -m pytest -q
```

### WEAK_MODE

В src/training/config.py:
```
WEAK_MODE = True
```

В этом режиме:

- модель не загружается

- LoRA не загружается

- обучение пропускается

- генерация пропускается

- оценка пропускается

Но весь пайплайн, логика, структура, Hydra, тесты — работают.

## Пример результатов

Hellaswag:

| Модель | acc_norm |
| --- | --- |
| Baseline | 0.42 |
| Finetuned | 0.61 |