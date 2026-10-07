"""OpenRouter usage signal: which models people actually use per use case.

`GET https://openrouter.ai/api/v1/models?category=<c>` (public, no key) returns the top 20 models of a category ordered
by usage on OpenRouter — the same data as its Rankings page (programming, roleplay, translation, legal, …). Stored as
boards `openrouter_usage/<category>` with `rank` 1–20 and score = 21 − rank (higher = more used). It's a popularity
signal, not quality: the selector shows it on model cards but never blends it into the quality score.
"""

from __future__ import annotations

from datetime import UTC, datetime

import httpx

from whichapiapi.schema.benchmark import BenchmarkScore

URL = "https://openrouter.ai/api/v1/models"
CATEGORIES = (
    "programming", "roleplay", "marketing", "marketing/seo", "technology", "science", "translation", "legal",
    "finance", "health", "trivia", "academia",
)  # fmt: skip


class OpenRouterUsageAdapter:
    id = "openrouter_usage"
    license = "OpenRouter public API (usage rankings); facts, cite OpenRouter"
    attribution = "Most-used models per category on OpenRouter (openrouter.ai/rankings)"

    def __init__(self, client: httpx.Client | None = None, categories: tuple[str, ...] = CATEGORIES):
        self.client = client or httpx.Client(timeout=60)
        self.categories = categories

    def fetch(self) -> list[BenchmarkScore]:
        out: list[BenchmarkScore] = []
        for cat in self.categories:
            r = self.client.get(URL, params={"category": cat})
            r.raise_for_status()
            out += self.parse(cat, r.json())
        return out

    @staticmethod
    def parse(category: str, body: dict) -> list[BenchmarkScore]:
        today = datetime.now(UTC).date().isoformat()
        rows = body.get("data") or []
        return [
            BenchmarkScore(
                source="openrouter_usage",
                board=category,
                model_name=m["id"].split("/", 1)[-1],
                organization=m["id"].split("/", 1)[0] if "/" in m["id"] else None,
                score=float(len(rows) + 1 - i),
                unit="usage_rank",
                rank=i,
                date=today,
            )
            for i, m in enumerate(rows, 1)
            if m.get("id")
        ]
