from __future__ import annotations

from typing import Dict, List, Tuple

import pandas as pd

REQUIRED_COLUMNS: List[str] = [
    "experiment_id",
    "Model",
    "dataset",
    "gpu_type",
    "batch_size",
    "accuracy",
    "latency_ms",
    "tokens_per_second",
    "memory_usage_gb",
    "compute_cost_usd",
    "run_timestamp",
]

NUMERIC_BOUNDS: Dict[str, Tuple[float, float | None]] = {
    "accuracy": (0.0, 1.0),
    "latency_ms": (0.0, None),
    "tokens_per_second": (0.0, None),
    "memory_usage_gb": (0.0, None),
    "compute_cost_usd": (0.0, None),
}


def validate_required_columns(df: pd.DataFrame) -> None:
    missing = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")


def coerce_schema_types(df: pd.DataFrame) -> pd.DataFrame:
    cleaned = df.copy()
    cleaned["experiment_id"] = pd.to_numeric(cleaned["experiment_id"], errors="coerce").astype("Int64")
    cleaned["batch_size"] = pd.to_numeric(cleaned["batch_size"], errors="coerce").astype("Int64")

    numeric_cols = [
        "accuracy",
        "latency_ms",
        "tokens_per_second",
        "memory_usage_gb",
        "compute_cost_usd",
    ]
    for col in numeric_cols:
        cleaned[col] = pd.to_numeric(cleaned[col], errors="coerce")

    cleaned["run_timestamp"] = pd.to_datetime(cleaned["run_timestamp"], errors="coerce", utc=True)

    return cleaned


def validate_value_ranges(df: pd.DataFrame) -> None:
    for col, (minimum, maximum) in NUMERIC_BOUNDS.items():
        series = df[col].dropna()
        if (series < minimum).any():
            raise ValueError(f"Column '{col}' has values below {minimum}.")
        if maximum is not None and (series > maximum).any():
            raise ValueError(f"Column '{col}' has values above {maximum}.")
