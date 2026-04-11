# AI Model Performance Metrics

This is a small analytics project for exploring AI model experiment data.
The notebook walks through data checks, outlier analysis, model training, and visualizations.

## What is in this project

- `main.ipynb`: full analysis workflow
- `ai_model_experiments.csv`: dataset used in the notebook
- `requirements.txt`: Python dependencies

## How to run

1. Install dependencies:

```bash
pip install -r requirements.txt
```

2. Open and run `main.ipynb` from top to bottom.

## Visualizations included

- Accuracy distribution histogram
- Compute cost distribution histogram
- Memory usage box plot
- Tokens per second box plot
- Correlation heat map
- Average compute cost by GPU type (bar chart)
- Mean accuracy by batch size (line plot)
- Compute cost distribution by dataset (box plot)
- Latency vs tokens per second by GPU type (scatter plot)

## Model training

The notebook uses Linear Regression to predict `compute_cost_usd`.
Features include numeric metrics (`latency_ms`, `tokens_per_second`, `memory_usage_gb`, `batch_size`) and encoded categorical columns (`Model`, `dataset`, `gpu_type`).
