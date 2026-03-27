from __future__ import annotations

import json
from pathlib import Path

from src.common.manifest import sha256_file


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MANIFEST_PATH = PROJECT_ROOT / "reports" / "data_manifest.json"


def test_manifest_checksum_matches_output_csv_when_present() -> None:
    if not MANIFEST_PATH.exists():
        # Pipeline artifacts are generated during smoke runs; skip hard failure if absent.
        return

    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    artifacts = manifest.get("artifacts", {})
    output_csv = artifacts.get("output_csv")
    expected_hash = artifacts.get("output_csv_sha256")

    if not output_csv or not expected_hash:
        return

    output_path = Path(output_csv)
    if not output_path.exists():
        return

    assert sha256_file(output_path) == expected_hash
