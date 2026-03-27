from __future__ import annotations

import pandas as pd

from src.collection.normalize import merge_and_deduplicate
from src.collection.normalize import normalize_api_rows


def test_normalize_api_rows_maps_required_schema() -> None:
    raw_rows = [
        {
            "id": "model-x",
            "dataset_name": "dataset-y",
            "hardware": "H100",
            "batch": 4,
            "score": 92.5,
            "latency": 120.2,
            "throughput": 63.1,
            "memory": 18.4,
            "cost": 0.72,
            "created_at": "2026-02-01T01:02:03Z",
        }
    ]

    df = normalize_api_rows(raw_rows, source_name="testsource")

    assert len(df) == 1
    assert float(df.loc[0, "accuracy"]) == 0.925
    assert df.loc[0, "Model"] == "model-x"
    assert df.loc[0, "dataset"] == "dataset-y"
    assert df.loc[0, "gpu_type"] == "H100"
    assert df.loc[0, "source_name"] == "testsource"
    assert bool(df.loc[0, "is_synthetic"]) is False


def test_merge_and_deduplicate_prefers_higher_priority_source() -> None:
    bootstrap_df = pd.DataFrame(
        [
            {
                "experiment_id": 1,
                "Model": "model-a",
                "dataset": "dataset-a",
                "gpu_type": "A100",
                "batch_size": 8,
                "accuracy": 0.91,
                "latency_ms": 120.0,
                "tokens_per_second": 50.0,
                "memory_usage_gb": 18.0,
                "compute_cost_usd": 0.8,
                "run_timestamp": "2026-02-01T01:02:03+00:00",
                "source_name": "bootstrap",
                "is_synthetic": False,
                "synthetic_fields": "",
                "confidence_score": 1.0,
            }
        ]
    )

    api_df = pd.DataFrame(
        [
            {
                "experiment_id": 99,
                "Model": "model-a",
                "dataset": "dataset-a",
                "gpu_type": "A100",
                "batch_size": 8,
                "accuracy": 0.85,
                "latency_ms": 130.0,
                "tokens_per_second": 45.0,
                "memory_usage_gb": 17.0,
                "compute_cost_usd": 0.7,
                "run_timestamp": "2026-02-01T01:02:03+00:00",
                "source_name": "openml",
                "is_synthetic": False,
                "synthetic_fields": "",
                "confidence_score": 1.0,
            }
        ]
    )

    merged = merge_and_deduplicate(bootstrap_df=bootstrap_df, api_frames=[api_df])

    assert len(merged) == 1
    assert merged.loc[0, "source_name"] == "bootstrap"
