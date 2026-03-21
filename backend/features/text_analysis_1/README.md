# Text Analysis Feature

## Overview

This feature analyzes text content (news headlines and articles) to detect fake news using Natural Language Processing (NLP).

## Architecture

```
Text Input → Preprocessor → RoBERTa Encoder → Classifier → Prediction
```

## Components

| File | Purpose |
|------|---------|
| `model.py` | Neural network architecture (RoBERTa + classifier head) |
| `preprocessor.py` | Text cleaning and tokenization |
| `predictor.py` | High-level prediction interface |

## Usage

```python
from features.text_analysis_1 import predict

result = predict("Breaking news: Scientists discover new planet")
print(result)
# {
#     "prediction": "REAL",
#     "confidence": 92.5,
#     "fake_probability": 7.5,
#     "real_probability": 92.5
# }
```

## Model Details

- **Base Model**: DistilRoBERTa (82M parameters)
- **Fine-tuned on**: FakeNewsNet dataset (21K+ articles)
- **Accuracy**: ~86% on test set
- **F1 Score**: ~80%

## Training

See `/training/scripts/train_text_model.py` to retrain the model.

