from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Dict

import pandas as pd
from dotenv import load_dotenv

from src.collection.augment import augment_dataset
from src.collection.bootstrap import load_bootstrap_dataset
from src.collection.clients.huggingface import fetch_huggingface_rows
from src.collection.clients.openml import fetch_openml_rows
from src.collection.clients.paperswithcode import fetch_paperswithcode_rows
from src.collection.normalize import merge_and_deduplicate, normalize_api_rows
from src.common.config import AppConfig
from src.common.schema import coerce_schema_types, validate_required_columns, validate_value_ranges


class DataCollectionPipeline:
    """Phase 1 collection pipeline with API merge, augmentation controls, and manifest output."""

    def __init__(self, config: AppConfig) -> None:
        self.config = config

    def run(self) -> Dict[str, int]:
        load_dotenv(self.config.pipeline.project_root / ".env")

        bootstrap_df = load_bootstrap_dataset(self.config.pipeline.bootstrap_csv)
        bootstrap_df = bootstrap_df.copy()
        bootstrap_df["source_name"] = "bootstrap"
        bootstrap_df["is_synthetic"] = False
        bootstrap_df["synthetic_fields"] = ""
        bootstrap_df["confidence_score"] = 1.0

        paperswithcode_raw = fetch_paperswithcode_rows(self.config.sources.paperswithcode)
        openml_raw = fetch_openml_rows(self.config.sources.openml)
        huggingface_raw = fetch_huggingface_rows(self.config.sources.huggingface)

        paperswithcode_df = normalize_api_rows(paperswithcode_raw, source_name="paperswithcode")
        openml_df = normalize_api_rows(openml_raw, source_name="openml")
        huggingface_df = normalize_api_rows(huggingface_raw, source_name="huggingface")

        merged_df = merge_and_deduplicate(
            bootstrap_df=bootstrap_df,
            api_frames=[paperswithcode_df, openml_df, huggingface_df],
        )
        augmented_df, augmentation_meta = augment_dataset(
            df=merged_df,
            target_min_rows=self.config.pipeline.target_min_rows,
            synthetic_max_ratio=self.config.pipeline.synthetic_max_ratio,
            random_seed=self.config.run.random_seed,
        )

        validate_required_columns(augmented_df)
        augmented_df = coerce_schema_types(augmented_df)
        validate_value_ranges(augmented_df)

        augmented_df["experiment_id"] = pd.Series(range(1, len(augmented_df) + 1), dtype="Int64")

        output_dir = self.config.pipeline.output_csv.parent
        output_dir.mkdir(parents=True, exist_ok=True)
        augmented_df.to_csv(self.config.pipeline.output_csv, index=False)

        manifest = {
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "row_count": int(len(augmented_df)),
            "target_min_rows": int(self.config.pipeline.target_min_rows),
            "synthetic_max_ratio": float(self.config.pipeline.synthetic_max_ratio),
            "sources_enabled": {
                "paperswithcode": self.config.sources.paperswithcode.enabled,
                "openml": self.config.sources.openml.enabled,
                "huggingface": self.config.sources.huggingface.enabled,
            },
            "source_row_counts": {
                "bootstrap": int(len(bootstrap_df)),
                "paperswithcode": int(len(paperswithcode_df)),
                "openml": int(len(openml_df)),
                "huggingface": int(len(huggingface_df)),
            },
            "augmentation": {
                "synthetic_rows_added": int(augmentation_meta["synthetic_rows_added"]),
                "synthetic_ratio_actual": float(augmentation_meta["synthetic_ratio_actual"]),
                "synthetic_cap_respected": bool(augmentation_meta["synthetic_cap_respected"]),
                "target_achieved": bool(augmentation_meta["target_achieved"]),
                "cap_limited": bool(augmentation_meta["cap_limited"]),
            },
            "note": "Phase 1 ingestion merges bootstrap/API data, applies capped synthetic augmentation when needed, and logs provenance.",
        }

        manifest_path = self.config.pipeline.manifest_json
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        with manifest_path.open("w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2)

        return {
            "rows_written": int(len(augmented_df)),
            "target_min_rows": int(self.config.pipeline.target_min_rows),
        }
