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

### One-shot benchmark (train + full test + JSON)

From repository root:

```bash
python training/scripts/run_benchmarks.py --device auto --epochs 3
```

### Google Colab (T4 GPU)

1. Open `colab/REMIX_FND_T4_Benchmarks.ipynb` in [Colab](https://colab.research.google.com/) (File → Upload notebook, or open from GitHub).
2. **Runtime → Change runtime type → GPU**.
3. Run all cells. Downloads `benchmark_summary.json` and `benchmark_eval_metrics.json` at the end.

Ensure `data/processed/fakenewsnet/{train,val,test}.json` is in the repo or uploaded; the notebook clones from GitHub.

## Configuration

Edit `configs/training_config.yaml` to adjust hyperparameters.

## Requirements

- PyTorch >= 2.0
- transformers >= 4.30
- scikit-learn >= 1.3

