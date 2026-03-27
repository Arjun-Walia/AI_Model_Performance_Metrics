from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Dict


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate_augmentation_guard(manifest: Dict[str, Any], synthetic_max_ratio: float) -> None:
    augmentation = manifest.get("augmentation", {})
    ratio = float(augmentation.get("synthetic_ratio_actual", 0.0))
    cap_ok = bool(augmentation.get("synthetic_cap_respected", ratio <= synthetic_max_ratio + 1e-9))

    if ratio > synthetic_max_ratio + 1e-9 or not cap_ok:
        raise ValueError(
            f"Synthetic ratio validation failed: ratio={ratio:.6f}, limit={synthetic_max_ratio:.6f}, cap_ok={cap_ok}"
        )
