"""Feedback inbox for agents using the MCP server / REST API from other projects (like Tributary's
EXTERNAL_FEEDBACK.md). Entries are appended to a Markdown file the maintainers triage:
`WHICHAPIAPI_FEEDBACK_FILE`, else docs/EXTERNAL_FEEDBACK.md of the source checkout when present, else
`$WHICHAPIAPI_HOME/EXTERNAL_FEEDBACK.md`.
"""

from __future__ import annotations

import os
from datetime import UTC, datetime
from pathlib import Path

from whichapiapi.transports.base import redact

CATEGORIES = ("bug", "wrong-data", "friction", "missing-feature", "praise")
SEVERITIES = ("high", "medium", "low")
_REPO_FILE = Path(__file__).resolve().parents[2] / "docs" / "EXTERNAL_FEEDBACK.md"


def feedback_file() -> Path:
    if env := os.environ.get("WHICHAPIAPI_FEEDBACK_FILE"):
        return Path(env)
    if _REPO_FILE.exists():
        return _REPO_FILE
    return (
        Path(os.environ.get("WHICHAPIAPI_HOME", Path.home() / ".local/share/whichapiapi"))
        / "EXTERNAL_FEEDBACK.md"
    )


def _one_line(text: str, limit: int) -> str:
    return " ".join(redact(str(text or "")).split())[:limit]


def add_entry(
    summary: str,
    what_happened: str,
    category: str = "bug",
    severity: str = "medium",
    tool: str = "",
    evidence: str = "",
    suggested_fix: str = "",
    project: str | None = None,
) -> dict[str, str]:
    """Append one entry; secrets are redacted and fields length-capped (the file is read by people and agents)."""
    category = category if category in CATEGORIES else "bug"
    severity = severity if severity in SEVERITIES else "medium"
    project = project or os.getcwd()
    now = datetime.now(UTC)
    entry = (
        f"\n## {now:%Y-%m-%d %H:%M} UTC — {_one_line(summary, 120)} — {project}\n\n"
        f"- **Category:** {category}\n"
        f"- **Tool:** {_one_line(tool, 80) or '—'} · **Severity:** {severity}\n"
        f"- **What happened:** {_one_line(what_happened, 2000)}\n"
        f"- **Evidence:** {_one_line(evidence, 2000) or '—'}\n"
        f"- **Suggested fix:** {_one_line(suggested_fix, 1000) or '—'}\n"
        f"- **Status:** open\n"
    )
    path = feedback_file()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(entry)
    return {"saved_to": str(path), "entry": entry.strip()}
