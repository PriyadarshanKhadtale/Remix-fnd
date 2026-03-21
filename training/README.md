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

### Train Text Model

```bash
python scripts/train_text_model.py \
  --data_dir ../data/processed/fakenewsnet \
  --epochs 5 \
  --batch_size 32
```

### Evaluate Model

```bash
python scripts/evaluate.py \
  --model_path ../models/text_classifier/best_model.pt \
  --test_data ../data/processed/fakenewsnet/test.json
```

## Configuration

Edit `configs/training_config.yaml` to adjust hyperparameters.

## Requirements

- PyTorch >= 2.0
- transformers >= 4.30
- scikit-learn >= 1.3

