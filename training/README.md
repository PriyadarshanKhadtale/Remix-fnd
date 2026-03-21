# Training

## Overview

Scripts and configurations for training REMIX-FND models.

## Structure

```
training/
├── scripts/
│   ├── train_text_model.py    # Train text classifier
│   └── evaluate.py            # Evaluate models
├── configs/
│   └── training_config.yaml   # Hyperparameters
├── notebooks/
│   └── experiments.ipynb      # Experiment tracking
└── README.md
```

## Quick Start

Install training deps (from repo root):

```bash
pip install -r training/requirements-train.txt
```

### Train Text Model

```bash
python training/scripts/train_text_model.py \
  --data_dir data/processed/fakenewsnet \
  --epochs 5 \
  --batch_size 32 \
  --device auto
```

`--device` can be `cpu`, `cuda`, `mps` (Apple Silicon), or `auto`.

### Evaluate Model

```bash
python training/scripts/evaluate.py \
  --model_path models/text_classifier/best_model.pt \
  --test_data data/processed/fakenewsnet/test.json \
  --device auto \
  --output_json benchmark_eval_metrics.json
```

### One-shot benchmark (veracity only: train + full test + JSON)

From repository root:

```bash
python training/scripts/run_benchmarks.py --device auto --epochs 3
```

### Full automated suite (veracity + DANN + stance + AI micro + latency)

Runs all training scripts with JSON artifacts under `benchmark_runs/<UTC-time>/` and a root **`manifest.json`** (includes **paper gaps** not covered by code).

```bash
python training/scripts/run_all_benchmarks.py --device auto
# Smoke / CI-friendly:
python training/scripts/run_all_benchmarks.py --device auto --quick
```

**Colab logging:** notebook cell output captures **stdout/stderr** from every subprocess. **Structured results** are the JSON files in `benchmark_runs/.../` (zip/download in the last notebook cell). No extra service is required.

### Google Colab (T4 GPU)

1. Open `colab/REMIX_FND_T4_Benchmarks.ipynb` in [Colab](https://colab.research.google.com/).
2. **Runtime → Change runtime type → GPU**.
3. Run all cells; optional **`--quick`** in the benchmark cell for a shorter smoke run.

Ensure `data/processed/fakenewsnet/*.json` and LIAR TSVs under `data/fact_checking/` or `backend/data_fact_checking/` are present (they ship with this repo).

## Configuration

Edit `configs/training_config.yaml` to adjust hyperparameters.

## Requirements

- PyTorch >= 2.0
- transformers >= 4.30
- scikit-learn >= 1.3

