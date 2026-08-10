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

LLM-instruction-tuning-qwen2-qlora/
│
├── notebooks/                # Jupyter notebooks for training & evaluation
│   ├── 01_sft_training_qwen2.ipynb
│   ├── 02_baseline_evaluation.ipynb
│   └── 03_finetuned_evaluation.ipynb
│
├── src/                      # Modular Python code
│   ├── data/                 # Dataset loading and preprocessing
│   ├── training/             # QLoRA training pipeline
│   ├── evaluation/           # Baseline and finetuned evaluation
│   ├── generation/           # Text generation and comparison
│   └── utils/                # Logging, profiling, model loading
│
├── models/                   # Saved LoRA adapters
├── data/                     # Sample dataset fragments
├── results/                  # Evaluation outputs
├── README.md                 # Main documentation (RU)
├── README_EN.md              # English documentation
├── requirements.txt          # Dependencies


---

## Training Pipeline (QLoRA)

The training pipeline includes:
- Loading **Qwen2‑0.5B‑Instruct**  
- Applying **LoRA adapters**  
- Preparing a **multi‑domain instruction dataset (50k samples)**  
- Configuring QLoRA hyperparameters  
- Monitoring **loss**, **entropy**, and **token accuracy**  
- Saving model artifacts for inference  

Training implementation:
- `notebooks/01_sft_training_qwen2.ipynb`
- `src/training/`

---

## Evaluation

Two evaluation modes are provided:

### 1️⃣ Baseline Evaluation
Before finetuning, the model is evaluated on:
- **Hellaswag (acc_norm metric)**  
- **Generative tasks** (ML, analytics, SQL, banking domain)

### 2️⃣ Finetuned Evaluation
After QLoRA training, the model is re‑evaluated using the same tasks.

Evaluation code:
- `notebooks/02_baseline_evaluation.ipynb`
- `notebooks/03_finetuned_evaluation.ipynb`
- `src/evaluation/`

---

## Generation Comparison

The project includes structured comparison of baseline vs. finetuned outputs on:
- Machine learning explanations  
- SQL queries  
- Product analytics tasks  
- Banking domain prompts  
- Cybersecurity reasoning  

Generation code:
- `src/generation/`

---

## Reproducibility

The project is fully reproducible:
- Deterministic training configuration  
- Isolated environment via `requirements.txt`  
- Modular code organization  
- Clear separation of training, evaluation, and inference  
- Version‑controlled model artifacts  

---

## Example Results

| Metric | Baseline | Finetuned |
|---------|-----------|-----------|
| Hellaswag acc_norm | 0.486 | 0.484 |
| Generation quality | General | Structured, domain‑specific |

---

## Requirements

Install dependencies:
```bash
pip install -r requirements.txt

Run notebooks locally or in Colab.