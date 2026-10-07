"""Berkeley Function-Calling Leaderboard (BFCL): function-calling accuracy across evaluation boards.

One row per model and prompting mode: overall accuracy, live accuracy, multi-turn accuracy, and non-live AST accuracy.
Higher accuracy is better.
"""

from __future__ import annotations

import csv
import io
import re

import httpx

from whichapiapi.schema.benchmark import BenchmarkScore

URL = "https://gorilla.cs.berkeley.edu/data_overall.csv"


def _num(value: str | None) -> float | None:
    if value is None or value.strip() in {"", "N/A"}:
        return None
    try:
        return float(value.strip().removesuffix("%"))
    except ValueError:
        return None


def _rank(value: str | None) -> int | None:
    try:
        return int(value) if value is not None and value.strip().isdigit() else None
    except ValueError:
        return None


class BFCLAdapter:
    id = "bfcl"
    license = "BFCL leaderboard data (Apache-2.0, gorilla.cs.berkeley.edu)"
    attribution = "Function-calling accuracy from the Berkeley Function-Calling Leaderboard (BFCL)"

    def __init__(self, client: httpx.Client | None = None):
        self.client = client or httpx.Client(timeout=60)

    def fetch(self) -> list[BenchmarkScore]:
        response = self.client.get(URL)
        response.raise_for_status()
        return self.parse(response.text)

    def parse(self, text: str) -> list[BenchmarkScore]:
        reader = csv.DictReader(io.StringIO(text))
        headers = reader.fieldnames or []
        out: list[BenchmarkScore] = []

        for row in reader:
            raw_name = row.get("Model", "")
            if not raw_name:
                continue

            match = re.match(r"^(.*?)\s*\(([^()]*)\)\s*$", raw_name)
            if match:
                model_name = match.group(1).strip()
                suffix = match.group(2).strip().lower()
                mode = "prompt" if suffix == "prompt" else "fc"
            else:
                model_name = raw_name.strip()
                mode = "fc"

            organization = row.get("Organization") if "Organization" in headers else None
            rank = _rank(row.get("Rank"))

            values = [
                (f"overall/{mode}", row.get("Overall Acc"), rank),
                (f"live/{mode}", row.get("Live Acc"), None),
                (f"multi_turn/{mode}", row.get("Multi Turn Acc"), None),
                (f"non_live/{mode}", row.get("Non-Live AST Acc"), None),
            ]

            out += [
                BenchmarkScore(
                    source=self.id,
                    board=board,
                    model_name=model_name,
                    score=score,
                    unit="pct",
                    higher_is_better=True,
                    rank=score_rank,
                    organization=organization,
                )
                for board, raw_score, score_rank in values
                if (score := _num(raw_score)) is not None
            ]

        return out
