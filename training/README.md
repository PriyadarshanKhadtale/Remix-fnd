# Training

## Overview

Scripts and configurations for training REMIX-FND models.

## Structure

```
training/
├── scripts/
│   ├── train_text_model.py    # Train text classifier
│   ├── evaluate.py            # Evaluate models (+ optional --json_out)
│   ├── run_benchmarks.py      # Full test-set metrics + AUROC + latency JSON
│   ├── orchestrate.py         # Veracity (+ optional stance) pipeline
│   └── ...
├── configs/
│   └── training_config.yaml   # Hyperparameters
├── notebooks/
│   └── experiments.ipynb      # Experiment tracking
└── README.md
```

## Quick Start

### Train Text Model

```bash
python scripts/train_text_model.py \
  --data_dir ../data/processed/fakenewsnet \
  --epochs 5 \
  --batch_size 32
```

### Evaluate Model

From repo root:

```bash
python3 training/scripts/evaluate.py \
  --model_path models/text_classifier/best_model.pt \
  --test_data data/processed/fakenewsnet/test.json \
  --device cpu
```

Optional machine-readable metrics (accuracy, macro F1, AUROC, full report):

```bash
python3 training/scripts/evaluate.py \
  --model_path models/text_classifier/best_model.pt \
  --test_data data/processed/fakenewsnet/test.json \
  --device cuda \
  --json_out benchmark_results/eval_snapshot.json
```

### Reproducible benchmark bundle

Runs the full test split, records git commit, torch version, optional batch-1 latency (not the same as full API latency):

```bash
python3 training/scripts/run_benchmarks.py --device auto
# Quick smoke test:
python3 training/scripts/run_benchmarks.py --max_samples 200 --latency_runs 0
```

Outputs `benchmark_results/benchmark_<utc_timestamp>.json` and `benchmark_results/latest.json` (gitignored except `.gitkeep`).

## Configuration

Edit `configs/training_config.yaml` to adjust hyperparameters.

## Requirements

- PyTorch >= 2.0
- transformers >= 4.30
- scikit-learn >= 1.3

