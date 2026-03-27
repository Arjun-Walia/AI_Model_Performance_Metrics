from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.common.schema import (
    coerce_schema_types,
    validate_required_columns,
    validate_value_ranges,
)


def load_bootstrap_dataset(csv_path: Path) -> pd.DataFrame:
    if not csv_path.exists():
        raise FileNotFoundError(f"Bootstrap dataset not found: {csv_path}")

    df = pd.read_csv(csv_path)
    validate_required_columns(df)
    df = coerce_schema_types(df)
    validate_value_ranges(df)

    if df["run_timestamp"].isna().any():
        raise ValueError("Invalid run_timestamp values found during bootstrap load.")

    return df
