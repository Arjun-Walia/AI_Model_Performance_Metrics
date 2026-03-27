from __future__ import annotations

from pathlib import Path

from src.common.manifest import sha256_file, validate_augmentation_guard


def test_sha256_file_stable(tmp_path: Path) -> None:
    file_path = tmp_path / "sample.txt"
    file_path.write_text("abc123\n", encoding="utf-8")

    h1 = sha256_file(file_path)
    h2 = sha256_file(file_path)

    assert h1 == h2
    assert len(h1) == 64


def test_validate_augmentation_guard_accepts_valid_ratio() -> None:
    manifest = {
        "augmentation": {
            "synthetic_ratio_actual": 0.2,
            "synthetic_cap_respected": True,
        }
    }

    validate_augmentation_guard(manifest, synthetic_max_ratio=0.3)


def test_validate_augmentation_guard_rejects_invalid_ratio() -> None:
    manifest = {
        "augmentation": {
            "synthetic_ratio_actual": 0.31,
            "synthetic_cap_respected": False,
        }
    }

    raised = False
    try:
        validate_augmentation_guard(manifest, synthetic_max_ratio=0.3)
    except ValueError:
        raised = True

    assert raised is True
