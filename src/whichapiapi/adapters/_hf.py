"""Hugging Face Hub helpers shared by benchmark adapters: anonymous requests get 429s under load, so retry politely."""

from __future__ import annotations

import os
import time
from typing import Any

import httpx

HUB = "https://huggingface.co"
ROWS = "https://datasets-server.huggingface.co/rows"


def auth() -> dict[str, str]:
    """A (free) Hugging Face token in HF_TOKEN raises the anonymous rate limits; optional."""
    token = os.environ.get("HF_TOKEN")
    return {"Authorization": f"Bearer {token}"} if token else {}


def get(client: httpx.Client, url: str, retries: int = 3, **kw: Any) -> httpx.Response:
    """GET with up to `retries` retries on 429/5xx, honouring Retry-After (capped at 30 s). Raises on final error."""
    for attempt in range(retries + 1):
        r = client.get(url, follow_redirects=True, headers=auth(), **kw)
        if (r.status_code != 429 and r.status_code < 500) or attempt == retries:
            r.raise_for_status()
            return r
        try:
            wait = float(r.headers.get("retry-after", ""))
        except ValueError:
            wait = 2.0 * 2**attempt
        time.sleep(min(wait, 30.0))
    raise AssertionError("unreachable")


def rows(
    client: httpx.Client, dataset: str, config: str = "default", split: str = "train"
) -> list[dict[str, Any]]:
    """All rows of a small split through the datasets-server rows API (100 per request)."""
    out: list[dict[str, Any]] = []
    for offset in range(0, 100_000, 100):
        params = {"dataset": dataset, "config": config, "split": split, "offset": offset, "length": 100}
        batch = [x["row"] for x in get(client, ROWS, params=params).json().get("rows", [])]
        out.extend(batch)
        if len(batch) < 100:
            break
    return out
