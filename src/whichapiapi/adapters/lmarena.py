"""LMArena leaderboard dataset (HF `lmarena-ai/leaderboard-dataset`, CC BY 4.0): Arena Elo per model, per board.

LMArena has no public API; it publishes its leaderboards as this dataset. Each config is one arena (text, vision,
webdev, search, document, agent, text-to-image, ...) and holds every category of that arena back to back (text alone
has ~30: overall, coding, math, hard_prompts, russian, industry_legal_and_government, ...). Each config's split is a
parquet file in the dataset repo (text/latest is ~0.6 MB): one download per arena. (The datasets-server rows API
needs ~140 requests for the same data and rate-limits with 429s.)
"""

from __future__ import annotations

import io
from collections.abc import Iterable
from typing import Any

import httpx
import pyarrow.parquet as pq

from whichapiapi.adapters import _hf as hf
from whichapiapi.adapters._hf import HUB
from whichapiapi.schema.benchmark import BenchmarkScore

DATASET = "lmarena-ai/leaderboard-dataset"
# Arenas pulled by default. The *_style_control / *_factuality variants and the agent_* sub-metrics are skipped.
DEFAULT_CONFIGS = (
    "text",
    "vision",
    "webdev",
    "search",
    "document",
    "agent",
    "text_to_image",
    "image_edit",
    "text_to_video",
    "image_to_video",
    "video_edit",
)


def _int(v: Any) -> int | None:
    try:
        return int(v)
    except (TypeError, ValueError):
        return None


def _float(v: Any, digits: int = 2) -> float | None:
    try:
        return round(float(v), digits)
    except (TypeError, ValueError):
        return None


class LMArenaAdapter:
    id = "lmarena"
    license = "CC BY 4.0"
    attribution = "Arena Elo from the LMArena leaderboard dataset (lmarena-ai/leaderboard-dataset, CC BY 4.0)"

    def __init__(
        self,
        client: httpx.Client | None = None,
        configs: Iterable[str] = DEFAULT_CONFIGS,
        split: str = "latest",
    ):
        self.client = client or httpx.Client(timeout=60)
        self.configs = tuple(configs)
        self.split = split
        self.errors: dict[str, str] = {}

    def fetch(self) -> list[BenchmarkScore]:
        out: list[BenchmarkScore] = []
        for config in self.configs:
            try:  # one missing/renamed arena must not drop the others
                out.extend(self.parse(self._rows(config), config))
            except (httpx.HTTPError, ValueError, KeyError) as e:
                self.errors[config] = f"{type(e).__name__}: {str(e)[:120]}"
        return out

    def _files(self, config: str) -> list[str]:
        """The split's parquet files: normally the single `-00000-of-00001` shard, so try that name first and list
        the folder only if it's gone (the Hub API is the part that rate-limits)."""
        first = f"{config}/{self.split}-00000-of-00001.parquet"
        head = self.client.head(
            f"{HUB}/datasets/{DATASET}/resolve/main/{first}", follow_redirects=True, headers=hf.auth()
        )
        if head.status_code != 404:
            return [first]
        tree = hf.get(self.client, f"{HUB}/api/datasets/{DATASET}/tree/main/{config}").json()
        files = sorted(
            f["path"]
            for f in tree
            if f.get("type") == "file" and f["path"].split("/")[-1].startswith(f"{self.split}-")
        )
        if not files:
            raise ValueError(f"no {self.split} parquet files for {config}")
        return files

    def _rows(self, config: str) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for path in self._files(config):
            blob = hf.get(self.client, f"{HUB}/datasets/{DATASET}/resolve/main/{path}").content
            rows.extend(pq.read_table(io.BytesIO(blob)).to_pylist())
        return rows

    def parse(self, rows: Iterable[dict[str, Any]], config: str = "text") -> list[BenchmarkScore]:
        out = []
        for row in rows:
            # arenas publish Elo as `rating`; the agent arena a win-rate style `score` with `session_count`
            elo = row.get("rating") is not None
            name, rating = (
                row.get("model_name"),
                _float(row.get("rating") if elo else row.get("score"), 2 if elo else 4),
            )
            if not name or rating is None:
                continue
            date = row.get("leaderboard_publish_date")
            out.append(
                BenchmarkScore(
                    source=self.id,
                    board=f"{config}/{row.get('category') or 'overall'}",
                    model_name=name,
                    score=rating,
                    unit="elo" if elo else "score",
                    rank=_int(row.get("rank")),
                    ci_low=_float(row.get("rating_lower" if elo else "score_ci_lower"), 2 if elo else 4),
                    ci_high=_float(row.get("rating_upper" if elo else "score_ci_upper"), 2 if elo else 4),
                    votes=_int(row.get("vote_count") if elo else row.get("session_count")),
                    date=str(date)[:10] if date else None,
                    organization=row.get("organization"),
                    license=row.get("license"),
                )
            )
        return out
