from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict

import yaml


@dataclass
class PipelineConfig:
    project_root: Path
    bootstrap_csv: Path
    output_csv: Path
    manifest_json: Path
    target_min_rows: int
    synthetic_max_ratio: float


@dataclass
class SourceEndpointConfig:
    enabled: bool
    base_url: str
    timeout_seconds: int
    max_retries: int
    max_rows: int


@dataclass
class SourceConfig:
    paperswithcode: SourceEndpointConfig
    openml: SourceEndpointConfig
    huggingface: SourceEndpointConfig


@dataclass
class RunConfig:
    random_seed: int
    timezone: str


@dataclass
class AppConfig:
    pipeline: PipelineConfig
    sources: SourceConfig
    run: RunConfig


def _resolve_path(project_root: Path, value: str) -> Path:
    p = Path(value)
    if p.is_absolute():
        return p
    return project_root / p


def load_config(config_path: Path) -> AppConfig:
    with config_path.open("r", encoding="utf-8") as f:
        raw: Dict[str, Any] = yaml.safe_load(f)

    project_root = config_path.parent.parent.resolve()

    pipeline_raw = raw["pipeline"]
    pipeline = PipelineConfig(
        project_root=project_root,
        bootstrap_csv=_resolve_path(project_root, pipeline_raw["bootstrap_csv"]),
        output_csv=_resolve_path(project_root, pipeline_raw["output_csv"]),
        manifest_json=_resolve_path(project_root, pipeline_raw["manifest_json"]),
        target_min_rows=int(pipeline_raw["target_min_rows"]),
        synthetic_max_ratio=float(pipeline_raw["synthetic_max_ratio"]),
    )

    def _build_source_endpoint(raw_sources: Dict[str, Any], key: str, default_base_url: str) -> SourceEndpointConfig:
        section = raw_sources.get(key, {})
        return SourceEndpointConfig(
            enabled=bool(section.get("enabled", True)),
            base_url=str(section.get("base_url", default_base_url)),
            timeout_seconds=int(section.get("timeout_seconds", 30)),
            max_retries=int(section.get("max_retries", 3)),
            max_rows=int(section.get("max_rows", 100)),
        )

    sources_raw = raw["sources"]
    sources = SourceConfig(
        paperswithcode=_build_source_endpoint(sources_raw, "paperswithcode", "https://paperswithcode.com/api/v1"),
        openml=_build_source_endpoint(sources_raw, "openml", "https://www.openml.org/api/v1"),
        huggingface=_build_source_endpoint(sources_raw, "huggingface", "https://huggingface.co/api"),
    )

    run_raw = raw.get("run", {})
    run = RunConfig(
        random_seed=int(run_raw.get("random_seed", 42)),
        timezone=str(run_raw.get("timezone", "UTC")),
    )

    return AppConfig(pipeline=pipeline, sources=sources, run=run)
