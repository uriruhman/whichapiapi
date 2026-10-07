"""Hugging Face Open ASR Leaderboard (`hf-audio/open-asr-leaderboard-results`): speech-to-text WER and speed.

One row per model (~70): average WER over the English test sets, WER per test set (LibriSpeech clean/other, AMI,
Earnings22, GigaSpeech, SPGISpeech, VoxPopuli, ...) and RTFx (seconds of audio transcribed per second; open models
only). Lower WER is better. Proprietary APIs (AssemblyAI, ElevenLabs, ...) are listed too, without RTFx.
"""

from __future__ import annotations

from typing import Any

import httpx

from whichapiapi.adapters import _hf as hf
from whichapiapi.schema.benchmark import BenchmarkScore

DATASET = "hf-audio/open-asr-leaderboard-results"


def _num(v: Any) -> float | None:
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return None if f != f else round(f, 3)  # NaN → None


def _board(column: str) -> str:
    return column.removesuffix(" WER").lower().replace(" ", "_")


class OpenASRAdapter:
    id = "open_asr"
    license = "Hugging Face Open ASR Leaderboard results (dataset card terms)"
    attribution = (
        "WER and RTFx from the Hugging Face Open ASR Leaderboard (hf-audio/open-asr-leaderboard-results)"
    )

    def __init__(self, client: httpx.Client | None = None):
        self.client = client or httpx.Client(timeout=60)

    def fetch(self) -> list[BenchmarkScore]:
        return self.parse(hf.rows(self.client, DATASET))

    def parse(self, rows: list[dict[str, Any]]) -> list[BenchmarkScore]:
        out: list[BenchmarkScore] = []
        ranked = sorted((r for r in rows if _num(r.get("avg")) is not None), key=lambda r: _num(r["avg"]))
        rank = {r["model"]: i for i, r in enumerate(ranked, 1)}
        for r in rows:
            name = r.get("model")
            if not name:
                continue
            values = [
                ("avg_wer", _num(r.get("avg")), "wer", False),
                ("rtfx", _num(r.get("RTFx")), "rtfx", True),
            ]
            values += [
                (f"wer/{_board(c)}", _num(v), "wer", False) for c, v in r.items() if c.endswith(" WER")
            ]
            out += [
                BenchmarkScore(
                    source=self.id,
                    board=board,
                    model_name=name,
                    score=value,
                    unit=unit,
                    higher_is_better=higher,
                    rank=rank.get(name) if board == "avg_wer" else None,
                )
                for board, value, unit, higher in values
                if value is not None
            ]
        return out
