from __future__ import annotations

from typing import Dict

import pandas as pd


METRIC_COLUMNS = [
    "accuracy",
    "latency_ms",
    "tokens_per_second",
    "memory_usage_gb",
    "compute_cost_usd",
]


def _min_max_scale(series: pd.Series, reverse: bool = False) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce")
    min_v = values.min(skipna=True)
    max_v = values.max(skipna=True)

    if pd.isna(min_v) or pd.isna(max_v) or max_v == min_v:
        scaled = pd.Series([0.5] * len(values), index=values.index, dtype="float64")
    else:
        scaled = (values - min_v) / (max_v - min_v)

    if reverse:
        scaled = 1.0 - scaled

    return scaled.clip(lower=0.0, upper=1.0)


def add_score_components(df: pd.DataFrame) -> pd.DataFrame:
    scored = df.copy()

    scored["norm_accuracy"] = _min_max_scale(scored["accuracy"], reverse=False)
    scored["norm_latency"] = _min_max_scale(scored["latency_ms"], reverse=True)
    scored["norm_throughput"] = _min_max_scale(scored["tokens_per_second"], reverse=False)
    scored["norm_memory"] = _min_max_scale(scored["memory_usage_gb"], reverse=True)
    scored["norm_cost"] = _min_max_scale(scored["compute_cost_usd"], reverse=True)

    return scored


def compute_efficiency_score(df: pd.DataFrame, weights: Dict[str, float]) -> pd.DataFrame:
    scored = add_score_components(df)

    weight_accuracy = float(weights.get("accuracy", 0.35))
    weight_latency = float(weights.get("latency", 0.2))
    weight_throughput = float(weights.get("throughput", 0.2))
    weight_memory = float(weights.get("memory", 0.1))
    weight_cost = float(weights.get("cost", 0.15))

    weight_sum = weight_accuracy + weight_latency + weight_throughput + weight_memory + weight_cost
    if weight_sum <= 0:
        raise ValueError("Efficiency score weights must sum to a positive value.")

    # Normalize weights to avoid accidental scaling issues from config values.
    weight_accuracy /= weight_sum
    weight_latency /= weight_sum
    weight_throughput /= weight_sum
    weight_memory /= weight_sum
    weight_cost /= weight_sum

    scored["ai_efficiency_score"] = (
        scored["norm_accuracy"] * weight_accuracy
        + scored["norm_latency"] * weight_latency
        + scored["norm_throughput"] * weight_throughput
        + scored["norm_memory"] * weight_memory
        + scored["norm_cost"] * weight_cost
    ).clip(lower=0.0, upper=1.0)

    return scored
