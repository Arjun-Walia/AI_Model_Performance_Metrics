from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict

import pandas as pd

from src.analysis.scoring import compute_efficiency_score
from src.common.config import AppConfig
from src.common.manifest import sha256_file
from src.common.schema import validate_required_columns


class AnalysisPipeline:
    def __init__(self, config: AppConfig) -> None:
        self.config = config

    def _load_input(self) -> pd.DataFrame:
        input_csv = self.config.analysis.input_csv
        if not input_csv.exists():
            raise FileNotFoundError(f"Analysis input CSV not found: {input_csv}")

        df = pd.read_csv(input_csv)
        validate_required_columns(df)
        return df

    def _build_model_summary(self, df: pd.DataFrame) -> pd.DataFrame:
        summary = (
            df.groupby("Model", dropna=False)
            .agg(
                experiments=("experiment_id", "count"),
                mean_accuracy=("accuracy", "mean"),
                mean_latency_ms=("latency_ms", "mean"),
                mean_tokens_per_second=("tokens_per_second", "mean"),
                mean_memory_usage_gb=("memory_usage_gb", "mean"),
                mean_compute_cost_usd=("compute_cost_usd", "mean"),
                mean_ai_efficiency_score=("ai_efficiency_score", "mean"),
            )
            .reset_index()
            .sort_values("mean_ai_efficiency_score", ascending=False)
        )
        return summary

    def run(self) -> Dict[str, int]:
        df = self._load_input()
        scored = compute_efficiency_score(df, weights=self.config.analysis.weights)

        output_csv = self.config.analysis.output_csv
        output_csv.parent.mkdir(parents=True, exist_ok=True)
        scored.to_csv(output_csv, index=False)

        model_summary = self._build_model_summary(scored)
        model_summary_csv = self.config.analysis.model_summary_csv
        model_summary_csv.parent.mkdir(parents=True, exist_ok=True)
        model_summary.to_csv(model_summary_csv, index=False)

        top_n = max(1, int(self.config.analysis.top_n_models))
        top_models = model_summary.head(top_n)

        report_json = self.config.analysis.report_json
        report_json.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "input_csv": str(self.config.analysis.input_csv),
            "output_csv": str(output_csv),
            "output_csv_sha256": sha256_file(output_csv),
            "model_summary_csv": str(model_summary_csv),
            "model_summary_sha256": sha256_file(model_summary_csv),
            "rows_scored": int(len(scored)),
            "models_analyzed": int(model_summary["Model"].nunique()),
            "top_n_models": top_n,
            "top_models": top_models.to_dict(orient="records"),
            "score_weights": self.config.analysis.weights,
        }
        with report_json.open("w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)

        return {
            "rows_scored": int(len(scored)),
            "models_analyzed": int(model_summary["Model"].nunique()),
        }
