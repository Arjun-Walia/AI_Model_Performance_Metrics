from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import pandas as pd

from src.common.schema import REQUIRED_COLUMNS


def _pick_first(item: Dict[str, Any], candidates: List[str]) -> Any:
    for key in candidates:
        if key in item and item[key] not in (None, ""):
            return item[key]
    return None


def _to_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _normalize_accuracy(raw: Any) -> float:
    value = _to_float(raw, default=0.0)
    if value > 1.0 and value <= 100.0:
        return value / 100.0
    if value < 0.0:
        return 0.0
    if value > 1.0:
        return 1.0
    return value


def _normalize_timestamp(raw: Optional[Any]) -> str:
    if raw is None or raw == "":
        return datetime.now(timezone.utc).isoformat()

    ts = pd.to_datetime(raw, errors="coerce", utc=True)
    if pd.isna(ts):
        return datetime.now(timezone.utc).isoformat()
    return ts.isoformat()


def normalize_api_rows(rows: List[Dict[str, Any]], source_name: str) -> pd.DataFrame:
    normalized: List[Dict[str, Any]] = []

    for index, item in enumerate(rows, start=1):
        if not isinstance(item, dict):
            continue

        model = _pick_first(item, ["Model", "model", "model_name", "name", "id"]) or f"{source_name}_model_{index}"
        dataset = _pick_first(item, ["dataset", "dataset_name", "benchmark", "task", "did"]) or "unknown_dataset"
        gpu_type = _pick_first(item, ["gpu_type", "hardware", "accelerator", "device"]) or "Unknown"
        batch_size = int(_to_float(_pick_first(item, ["batch_size", "batch", "micro_batch_size"]), default=32.0))

        accuracy = _normalize_accuracy(_pick_first(item, ["accuracy", "acc", "score", "f1", "metric_value"]))

        latency_ms = _to_float(_pick_first(item, ["latency_ms", "latency", "inference_latency_ms"]), default=0.0)
        if latency_ms < 0.0:
            latency_ms = 0.0

        tokens_per_second = _to_float(
            _pick_first(item, ["tokens_per_second", "throughput", "speed", "downloads"]),
            default=0.0,
        )
        if tokens_per_second < 0.0:
            tokens_per_second = 0.0

        memory_usage_gb = _to_float(_pick_first(item, ["memory_usage_gb", "memory_gb", "memory"]), default=0.0)
        if memory_usage_gb < 0.0:
            memory_usage_gb = 0.0

        compute_cost_usd = _to_float(_pick_first(item, ["compute_cost_usd", "cost_usd", "cost"]), default=0.0)
        if compute_cost_usd < 0.0:
            compute_cost_usd = 0.0

        run_timestamp = _normalize_timestamp(
            _pick_first(item, ["run_timestamp", "timestamp", "created_at", "date", "upload_time"])
        )

        normalized.append(
            {
                "experiment_id": index,
                "Model": str(model),
                "dataset": str(dataset),
                "gpu_type": str(gpu_type),
                "batch_size": max(batch_size, 1),
                "accuracy": accuracy,
                "latency_ms": latency_ms,
                "tokens_per_second": tokens_per_second,
                "memory_usage_gb": memory_usage_gb,
                "compute_cost_usd": compute_cost_usd,
                "run_timestamp": run_timestamp,
                "source_name": source_name,
                "is_synthetic": False,
                "synthetic_fields": "",
                "confidence_score": 1.0,
            }
        )

    frame = pd.DataFrame(normalized)
    if frame.empty:
        return pd.DataFrame(columns=REQUIRED_COLUMNS + ["source_name", "is_synthetic", "synthetic_fields", "confidence_score"])

    return frame[REQUIRED_COLUMNS + ["source_name", "is_synthetic", "synthetic_fields", "confidence_score"]]


def merge_and_deduplicate(bootstrap_df: pd.DataFrame, api_frames: List[pd.DataFrame]) -> pd.DataFrame:
    valid_api_frames = [frame for frame in api_frames if not frame.empty]
    if not valid_api_frames:
        merged = bootstrap_df.copy()
    else:
        merged = pd.concat([bootstrap_df, *valid_api_frames], ignore_index=True)

    merged = merged.drop_duplicates(
        subset=["Model", "dataset", "gpu_type", "batch_size", "run_timestamp"],
        keep="first",
    ).reset_index(drop=True)

    merged["experiment_id"] = pd.Series(range(1, len(merged) + 1), dtype="Int64")
    return merged
