"""SWE-bench leaderboards (swebench.com): % of real GitHub issues resolved, boards Verified / Lite / Test / Multimodal /
Multilingual. Entries are agent systems (scaffold + model); `pull_benchmarks` keeps the best run per model. Mostly
2024–early 2026 submissions: check `date`, the newest models may be missing.
"""

from __future__ import annotations

from numbers import Real
from typing import Any

import httpx

from whichapiapi.schema.benchmark import BenchmarkScore

URL = "https://raw.githubusercontent.com/SWE-bench/swe-bench.github.io/master/data/leaderboards.json"


class SWEBenchAdapter:
    id = "swebench"
    license = "SWE-bench leaderboard data (MIT, github.com/SWE-bench)"
    attribution = (
        "% resolved from the SWE-bench leaderboards (swebench.com); entries are agent systems, "
        "the score is the best agent run per model"
    )

    def __init__(self, client: httpx.Client | None = None):
        self.client = client or httpx.Client(timeout=60)

    def fetch(self) -> list[BenchmarkScore]:
        r = self.client.get(URL)
        r.raise_for_status()
        return self.parse(r.json())

    def parse(self, data: dict[str, Any]) -> list[BenchmarkScore]:
        out: list[BenchmarkScore] = []
        for leaderboard in data.get("leaderboards", []):
            board = leaderboard["name"].lower()
            for result in leaderboard.get("results", []):
                model_name = result.get("model_display")
                resolved = result.get("resolved")
                if model_name in (None, "Multiple", "Undisclosed", ""):
                    continue
                if isinstance(resolved, bool) or not isinstance(resolved, Real):
                    continue
                if result.get("warning"):
                    continue
                out.append(
                    BenchmarkScore(
                        source=self.id,
                        board=board,
                        model_name=model_name,
                        score=round(float(resolved), 2),
                        unit="pct",
                        higher_is_better=True,
                        date=str(result["date"])[:10] if result.get("date") else None,
                        organization=result.get("model_org"),
                    )
                )
        return out
