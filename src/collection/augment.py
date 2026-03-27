from __future__ import annotations

from datetime import timedelta
from typing import Dict, Tuple

import numpy as np
import pandas as pd


def _max_allowed_synthetic(current_rows: int, synthetic_max_ratio: float) -> int:
    if current_rows <= 0 or synthetic_max_ratio <= 0.0:
        return 0
    if synthetic_max_ratio >= 1.0:
        return current_rows
    return int(np.floor((synthetic_max_ratio * current_rows) / (1.0 - synthetic_max_ratio)))


def _safe_sample(df: pd.DataFrame, rng: np.random.Generator) -> pd.Series:
    idx = int(rng.integers(0, len(df)))
    return df.iloc[idx]


def augment_dataset(
    df: pd.DataFrame,
    target_min_rows: int,
    synthetic_max_ratio: float,
    random_seed: int,
) -> Tuple[pd.DataFrame, Dict[str, float | int | bool]]:
    current_rows = int(len(df))
    needed_rows = max(target_min_rows - current_rows, 0)
    max_synthetic_rows = _max_allowed_synthetic(current_rows=current_rows, synthetic_max_ratio=synthetic_max_ratio)
    synthetic_rows_to_add = min(needed_rows, max_synthetic_rows)

    if synthetic_rows_to_add <= 0:
        ratio = 0.0
        return df, {
            "synthetic_rows_added": 0,
            "synthetic_ratio_actual": ratio,
            "synthetic_cap_respected": True,
            "target_achieved": bool(len(df) >= target_min_rows),
            "cap_limited": bool(needed_rows > 0),
        }

    if df.empty:
        # No bootstrap/API data to sample from; fail-safe returns unchanged dataset.
        return df, {
            "synthetic_rows_added": 0,
            "synthetic_ratio_actual": 0.0,
            "synthetic_cap_respected": True,
            "target_achieved": False,
            "cap_limited": True,
        }

    rng = np.random.default_rng(random_seed)
    generated_rows = []

    for _ in range(synthetic_rows_to_add):
        base = _safe_sample(df, rng)

        latency_scale = float(rng.uniform(0.9, 1.1))
        throughput_scale = float(rng.uniform(0.9, 1.1))
        memory_scale = float(rng.uniform(0.95, 1.08))
        cost_scale = float(rng.uniform(0.9, 1.15))
        accuracy_delta = float(rng.normal(loc=0.0, scale=0.01))

        timestamp = pd.to_datetime(base["run_timestamp"], utc=True, errors="coerce")
        if pd.isna(timestamp):
            timestamp = pd.Timestamp.utcnow().tz_localize("UTC")
        timestamp = timestamp + timedelta(minutes=int(rng.integers(1, 30)))

        generated_rows.append(
            {
                "experiment_id": None,
                "Model": str(base["Model"]),
                "dataset": str(base["dataset"]),
                "gpu_type": str(base["gpu_type"]),
                "batch_size": int(max(1, int(base["batch_size"]))),
                "accuracy": float(np.clip(float(base["accuracy"]) + accuracy_delta, 0.0, 1.0)),
                "latency_ms": float(max(0.0, float(base["latency_ms"]) * latency_scale)),
                "tokens_per_second": float(max(0.0, float(base["tokens_per_second"]) * throughput_scale)),
                "memory_usage_gb": float(max(0.0, float(base["memory_usage_gb"]) * memory_scale)),
                "compute_cost_usd": float(max(0.0, float(base["compute_cost_usd"]) * cost_scale)),
                "run_timestamp": timestamp.isoformat(),
                "source_name": "synthetic",
                "is_synthetic": True,
                "synthetic_fields": "accuracy,latency_ms,tokens_per_second,memory_usage_gb,compute_cost_usd",
                "confidence_score": 0.6,
            }
        )

    synthetic_df = pd.DataFrame(generated_rows)
    augmented = pd.concat([df, synthetic_df], ignore_index=True)

    final_rows = int(len(augmented))
    actual_ratio = float(synthetic_rows_to_add / final_rows) if final_rows > 0 else 0.0

    return augmented, {
        "synthetic_rows_added": int(synthetic_rows_to_add),
        "synthetic_ratio_actual": actual_ratio,
        "synthetic_cap_respected": bool(actual_ratio <= synthetic_max_ratio + 1e-9),
        "target_achieved": bool(final_rows >= target_min_rows),
        "cap_limited": bool(needed_rows > synthetic_rows_to_add),
    }