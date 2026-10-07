"""LiteLLM's public pricing JSON (MIT), read as plain HTTP data — never the `litellm` package (ADR-0003).

`model_prices_and_context_window.json` mostly answers "how much per token", which `models_dev.py` already
covers. What it has that we don't get elsewhere: per-entry `*_batches` cost fields (batch API pricing sitting
next to the normal price, not a separate `-batch` model key — checked directly against the live file before
writing this) and, for a handful of models, an `off_peak_pricing` block with its own discounted costs and
time windows. Both let us derive `Conditions.batch_discount` / `Conditions.off_peak` as *fractions off the
list price*, which is all `cost.engine.monthly_cost` needs. Fields the file simply doesn't carry
(`trains_on_data`, `free_tier`, `byok_fee`, `data_sharing_discount`, batch turnaround SLA) are left unset
rather than guessed.
"""

from __future__ import annotations

from typing import Any

import httpx

from whichapiapi.schema.naming import model_key

URL = "https://raw.githubusercontent.com/BerriAI/litellm/main/model_prices_and_context_window.json"


def _discount(list_cost: float | None, other_cost: float | None) -> float | None:
    if not list_cost or other_cost is None or list_cost <= 0:
        return None
    d = 1 - other_cost / list_cost
    return round(d, 4) if -0.01 <= d <= 1.0 else None  # drop nonsensical (negative/markup) diffs


def _format_windows(windows: list[dict[str, Any]]) -> str:
    parts = []
    for w in windows:
        hours = w.get("hours_utc")
        if isinstance(hours, list):
            hours = ",".join(hours)
        wd = w.get("weekdays")
        parts.append(f"{hours} (weekdays {wd})" if wd else str(hours))
    return "; ".join(p for p in parts if p) or "off-peak"


def _richness(cond: dict[str, Any]) -> int:
    return sum(v is not None for v in cond.values())


class LiteLLMConditionsAdapter:
    id = "litellm"
    license = "MIT"
    attribution = (
        "Batch/off-peak discount conditions derived from LiteLLM's model_prices_and_context_window.json "
        "(MIT) — read as HTTP data, the `litellm` package is not installed (ADR-0003)"
    )

    def __init__(self, client: httpx.Client | None = None, url: str = URL):
        self.client = client or httpx.Client(timeout=60)
        self.url = url

    def fetch(self) -> dict[str, dict[str, Any]]:
        r = self.client.get(self.url)
        r.raise_for_status()
        return self.parse(r.json())

    def _conditions_for(self, entry: dict[str, Any]) -> dict[str, Any]:
        cond: dict[str, Any] = {}
        in_cost = entry.get("input_cost_per_token")
        out_cost = entry.get("output_cost_per_token")
        fracs = [
            d
            for d in (
                _discount(in_cost, entry.get("input_cost_per_token_batches")),
                _discount(out_cost, entry.get("output_cost_per_token_batches")),
            )
            if d is not None
        ]
        if fracs:
            cond["batch_discount"] = round(sum(fracs) / len(fracs), 4)

        off_peak = entry.get("off_peak_pricing")
        if isinstance(off_peak, dict):
            discount = _discount(in_cost, off_peak.get("input_cost_per_token"))
            windows = off_peak.get("windows")
            if discount and isinstance(windows, list) and windows:
                cond["off_peak"] = {
                    "window": _format_windows(windows),
                    "tz": "UTC",
                    "discount": discount,
                }
        return cond

    def parse(self, data: dict[str, Any]) -> dict[str, dict[str, Any]]:
        out: dict[str, dict[str, Any]] = {}
        for raw_key, entry in data.items():
            if not isinstance(entry, dict):
                continue
            cond = self._conditions_for(entry)
            if not cond:
                continue
            key = model_key(raw_key)
            prev = out.get(key)
            if prev and _richness(prev) >= _richness(cond):
                continue  # keep whichever variant carries more derivable fields if names collapse
            out[key] = cond
        return out
