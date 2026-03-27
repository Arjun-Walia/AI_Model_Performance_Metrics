from __future__ import annotations

import pandas as pd

from src.analysis.scoring import compute_efficiency_score


def _sample_df() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "experiment_id": 1,
                "Model": "A",
                "dataset": "D1",
                "gpu_type": "A100",
                "batch_size": 8,
                "accuracy": 0.90,
                "latency_ms": 110.0,
                "tokens_per_second": 42.0,
                "memory_usage_gb": 20.0,
                "compute_cost_usd": 0.85,
                "run_timestamp": "2026-01-01T00:00:00+00:00",
            },
            {
                "experiment_id": 2,
                "Model": "B",
                "dataset": "D1",
                "gpu_type": "H100",
                "batch_size": 8,
                "accuracy": 0.95,
                "latency_ms": 95.0,
                "tokens_per_second": 58.0,
                "memory_usage_gb": 18.0,
                "compute_cost_usd": 0.76,
                "run_timestamp": "2026-01-01T00:05:00+00:00",
            },
        ]
    )


def test_efficiency_score_in_0_1_range() -> None:
    df = _sample_df()
    scored = compute_efficiency_score(df, weights={"accuracy": 0.4, "latency": 0.2, "throughput": 0.2, "memory": 0.1, "cost": 0.1})

    assert "ai_efficiency_score" in scored.columns
    assert scored["ai_efficiency_score"].between(0.0, 1.0).all()


def test_weight_normalization_still_computes_scores() -> None:
    df = _sample_df()
    scored = compute_efficiency_score(df, weights={"accuracy": 4.0, "latency": 2.0, "throughput": 2.0, "memory": 1.0, "cost": 1.0})

    assert scored["ai_efficiency_score"].between(0.0, 1.0).all()
