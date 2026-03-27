from __future__ import annotations

import argparse
from pathlib import Path

from src.analysis.pipeline import AnalysisPipeline
from src.collection.pipeline import DataCollectionPipeline
from src.common.config import load_config


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="ai-metrics",
        description="AI Performance Metrics Analysis and Optimization",
    )
    parser.add_argument(
        "--config",
        default="config/config.yaml",
        help="Path to YAML config file.",
    )
    parser.add_argument(
        "--phase",
        choices=["collect", "analyze", "all"],
        default="all",
        help="Pipeline phase to run.",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    config = load_config(Path(args.config).resolve())

    if args.phase in {"collect", "all"}:
        collect_pipeline = DataCollectionPipeline(config)
        collect_result = collect_pipeline.run()
        print(
            f"Collection finished. Rows written: {collect_result['rows_written']} / target: {collect_result['target_min_rows']}"
        )

    if args.phase in {"analyze", "all"}:
        analysis_pipeline = AnalysisPipeline(config)
        analysis_result = analysis_pipeline.run()
        print(
            f"Analysis finished. Rows scored: {analysis_result['rows_scored']} / models: {analysis_result['models_analyzed']}"
        )


if __name__ == "__main__":
    main()
