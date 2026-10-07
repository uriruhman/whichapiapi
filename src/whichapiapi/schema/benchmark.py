"""Benchmark scores: third-party evidence of model quality, per board (a leaderboard or one of its categories)."""

from __future__ import annotations

from pydantic import BaseModel


class BenchmarkScore(BaseModel):
    source: str  # "lmarena", "aider", ...
    board: str  # leaderboard/category inside the source: "text/coding", "webdev/overall", "polyglot"
    model_name: str  # as the source spells it
    score: float
    unit: str = "elo"  # "elo" | "pct" | "wer" | ...
    higher_is_better: bool = True
    rank: int | None = None
    ci_low: float | None = None
    ci_high: float | None = None
    votes: int | None = None
    date: str | None = None  # when the source published this number
    organization: str | None = None
    license: str | None = None  # of the model, when the source says ("Proprietary", "MIT", ...)

    @property
    def id(self) -> str:
        return f"{self.source}/{self.board}"
