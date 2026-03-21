# Explainability Feature

## Overview

This feature generates human-readable explanations for why the model classified news as fake or real.

## Status: ✅ Basic Implementation Available

## Features

| Feature | Status | Description |
|---------|--------|-------------|
| Keyword Highlighting | ✅ Done | Highlights suspicious/credible words |
| Pattern Detection | ✅ Done | Detects clickbait, sensationalism |
| Key Factors | ✅ Done | Lists main reasons for prediction |
| Suggestions | ✅ Done | Gives user actionable advice |
| Attention Visualization | 🚧 Planned | Show model attention weights |

## How It Works

```
Prediction → Pattern Analysis → Word Importance → Explanation Generation
                  ↓                    ↓                    ↓
            Fake Signals        Highlight Words      Human Summary
            Credible Signals    Importance Scores    Suggestions
```

## Detection Patterns

### Fake News Indicators
- Sensational words: "shocking", "secret", "exposed"
- Clickbait patterns: "you won't believe", "doctors hate"
- Exaggeration: "always", "never", "100% proven"

### Credibility Indicators
- Attribution: "according to", "study shows"
- Balanced language: "however", "critics argue"
- Specific details: dates, percentages, sources

## Usage

```python
from features.explainability_5 import explain, get_detailed_explanation

# Simple explanation
result = explain("Some news text", prediction)

# Detailed explanation with highlighted words
detailed = get_detailed_explanation(
    "Some news text", 
    prediction, 
    detail_level="detailed"
)
```

## Detail Levels

| Level | Description |
|-------|-------------|
| `simple` | One-sentence summary |
| `detailed` | Summary + key factors |
| `expert` | Full technical analysis |

