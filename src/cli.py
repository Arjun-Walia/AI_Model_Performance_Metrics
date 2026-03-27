from __future__ import annotations

import argparse
from pathlib import Path

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
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    config = load_config(Path(args.config).resolve())
    pipeline = DataCollectionPipeline(config)
    result = pipeline.run()

    print(
        f"Pipeline finished. Rows written: {result['rows_written']} / target: {result['target_min_rows']}"
    )


if __name__ == "__main__":
    main()
