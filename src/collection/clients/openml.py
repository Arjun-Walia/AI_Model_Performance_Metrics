from __future__ import annotations

import os
from typing import Any, Dict, List

from src.collection.clients.http import get_json
from src.common.config import SourceEndpointConfig


def fetch_openml_rows(config: SourceEndpointConfig) -> List[Dict[str, Any]]:
    if not config.enabled:
        return []

    # OpenML provides dataset/task metadata; this route returns a list of datasets in JSON.
    url = f"{config.base_url.rstrip('/')}/json/data/list/limit/{config.max_rows}"
    apikey = os.getenv("OPENML_API_KEY", "").strip()
    params = {"api_key": apikey} if apikey else None

    try:
        payload = get_json(
            url=url,
            timeout_seconds=config.timeout_seconds,
            max_retries=config.max_retries,
            params=params,
        )
    except Exception:
        return []

    if not isinstance(payload, dict):
        return []

    data_block = payload.get("data", {})
    if isinstance(data_block, dict):
        datasets = data_block.get("dataset", [])
        if isinstance(datasets, list):
            return datasets

    return []
