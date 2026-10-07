"""Export a suite to plain promptfoo YAML (ADR-0002): channels become openai providers with apiBaseUrl."""

from __future__ import annotations

import copy
from typing import Any

from whichapiapi.schema.suite import Suite


def to_promptfoo(suite: Suite) -> dict[str, Any]:
    raw = copy.deepcopy(suite.raw)
    raw.pop("x-whichapiapi", None)
    providers = []
    for spec in suite.providers:
        if spec.channel in (None, "mock"):
            continue
        ch = suite.ext.channels.get(spec.channel)
        cfg: dict[str, Any] = dict(spec.config)
        if ch is not None:
            cfg["apiBaseUrl"] = ch.base_url
            cfg["apiKeyEnvar"] = ch.key_env
        providers.append({"id": f"openai:chat:{spec.model}", "label": spec.label, "config": cfg})
    raw["providers"] = providers
    for p in raw.get("providers", []):
        p.pop("x-whichapiapi", None)
    return raw
