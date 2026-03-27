from __future__ import annotations

from typing import Any, Dict, List

from src.collection.clients.http import get_json
from src.common.config import SourceEndpointConfig


def fetch_paperswithcode_rows(config: SourceEndpointConfig) -> List[Dict[str, Any]]:
    if not config.enabled:
        return []

    # PapersWithCode API may vary by route availability; this endpoint is a pragmatic baseline.
    url = f"{config.base_url.rstrip('/')}/evaluations/"
    params = {"format": "json", "page_size": config.max_rows}

    try:
        payload = get_json(
            url=url,
            timeout_seconds=config.timeout_seconds,
            max_retries=config.max_retries,
            params=params,
        )
    except Exception:
        return []

    if isinstance(payload, dict):
        results = payload.get("results", [])
        if isinstance(results, list):
            return results

    if isinstance(payload, list):
        return payload

    return []
