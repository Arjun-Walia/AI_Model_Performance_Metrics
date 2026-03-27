from __future__ import annotations

import os
from typing import Any, Dict, List

from src.collection.clients.http import get_json
from src.common.config import SourceEndpointConfig


def fetch_huggingface_rows(config: SourceEndpointConfig) -> List[Dict[str, Any]]:
    if not config.enabled:
        return []

    token = os.getenv("HUGGINGFACE_TOKEN", "").strip()
    headers: Dict[str, str] = {}
    if token:
        headers["Authorization"] = f"Bearer {token}"

    url = f"{config.base_url.rstrip('/')}/models"
    params = {"limit": config.max_rows, "full": "true"}

    try:
        payload = get_json(
            url=url,
            timeout_seconds=config.timeout_seconds,
            max_retries=config.max_retries,
            headers=headers or None,
            params=params,
        )
    except Exception:
        return []

    if isinstance(payload, list):
        return payload

    if isinstance(payload, dict):
        models = payload.get("models", [])
        if isinstance(models, list):
            return models

    return []
