from __future__ import annotations

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
