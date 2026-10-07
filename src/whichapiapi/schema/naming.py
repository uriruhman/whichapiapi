"""Model-name normalisation so the same model can be compared across channels.

"deepseek/deepseek-v4.1-flash", "deepseek-v4.1-flash", "DeepSeek-V4.1-Flash" → "deepseek-v4.1-flash";
"claude-haiku-4-5-20251001" → "claude-haiku-4.5", "Claude 4.5 Haiku (FC)" → "claude-haiku-4.5" (one-digit version parts written with a dash, as Anthropic and some
resellers do, become the dotted form leaderboards use; "gemma-3-4b" stays: "4b" is a size). Deliberately
conservative: never merges different versions.
"""

from __future__ import annotations

import re

_DATE = re.compile(
    r"[-_@](20\d{6}|20\d{2}-\d{2}-\d{2})$"
)  # only full dates; short snapshot tags stay distinct
_SUFFIX = re.compile(r"[-:](latest|preview|free)$")
_DASHED_VERSION = re.compile(r"(?<![\d.])(\d)-(\d)(?=$|-)")
_PAREN = re.compile(r"\s*\([^)]*\)\s*$")  # leaderboard annotations: "(FC)", "(32k thinking)"
_CLAUDE_OLD_ORDER = re.compile(r"^claude-(\d+(?:\.\d+)?)-(opus|sonnet|haiku)(?=$|-)")


def model_key(model: str) -> str:
    m = _PAREN.sub("", model.strip().lower()).split("/")[-1]
    m = re.sub(r"\s+", "-", m.strip())  # display names: "Claude 4.5 Opus", "Gemini 3 Pro Preview"
    m = m.split(":")[0] if ":" in m and not m.startswith("ft:") else m
    for _ in range(2):
        m = _DATE.sub("", m)
        m = _SUFFIX.sub("", m)
    m = _DASHED_VERSION.sub(r"\1.\2", m)
    return _CLAUDE_OLD_ORDER.sub(
        r"claude-\2-\1", m
    )  # "claude-3.5-sonnet" and "claude-sonnet-3.5" are one model
