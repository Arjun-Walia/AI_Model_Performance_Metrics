from __future__ import annotations

import pandas as pd

from src.collection.augment import augment_dataset


def _sample_frame(rows: int = 10) -> pd.DataFrame:
    data = {
        "experiment_id": list(range(1, rows + 1)),
        "Model": ["ModelA"] * rows,
        "dataset": ["DatasetA"] * rows,
        "gpu_type": ["A100"] * rows,
        "batch_size": [8] * rows,
        "accuracy": [0.9] * rows,
        "latency_ms": [100.0] * rows,
        "tokens_per_second": [45.0] * rows,
        "memory_usage_gb": [16.0] * rows,
        "compute_cost_usd": [0.5] * rows,
        "run_timestamp": ["2026-01-01T00:00:00+00:00"] * rows,
        "source_name": ["bootstrap"] * rows,
        "is_synthetic": [False] * rows,
        "synthetic_fields": [""] * rows,
        "confidence_score": [1.0] * rows,
    }
    return pd.DataFrame(data)


def test_augment_respects_ratio_cap() -> None:
    df = _sample_frame(rows=10)

    augmented, meta = augment_dataset(
        df=df,
        target_min_rows=100,
        synthetic_max_ratio=0.3,
        random_seed=42,
    )

    # For 10 initial rows and max ratio 0.3, max synthetic rows is floor(0.3*10/0.7)=4.
    assert len(augmented) == 14
    assert meta["synthetic_rows_added"] == 4
    assert meta["synthetic_cap_respected"] is True
    assert meta["target_achieved"] is False
    assert meta["cap_limited"] is True


def test_augment_noop_when_target_already_met() -> None:
    df = _sample_frame(rows=25)

    augmented, meta = augment_dataset(
        df=df,
        target_min_rows=20,
        synthetic_max_ratio=0.3,
        random_seed=42,
    )

    assert len(augmented) == 25
    assert meta["synthetic_rows_added"] == 0
    assert meta["target_achieved"] is True
