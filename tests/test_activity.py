import json

import pytest

from whichapiapi import activity


def test_log_redacts_truncates_and_reports():
    activity.log_event("cli", "offers list", {"api_key": "sk-live-abcdefghijklmnop1234", "key_env": "RESELLER_API_KEY",  # gitleaks:allow (fake test value)
                       "note": "x" * 1000, "workflow": {"name": "w", "nodes": [1, 2, 3]}}, duration_ms=12.5)  # fmt: skip
    activity.log_event(
        "mcp",
        "run_eval",
        {"suite_path": "s.yaml"},
        ok=False,
        error="BudgetError: over cap sk-proj-AAAAAAAAAAAA1111",
    )
    events = activity.read_events(1)
    assert len(events) == 2
    first = events[0]["args"]
    assert first["api_key"] == "<redacted>" and first["key_env"] == "RESELLER_API_KEY"
    assert first["note"].endswith("<1000 chars>") and first["workflow"] == "<workflow 'w': 3 nodes>"
    assert "AAAAAAAAAAAA" not in events[1]["error"]
    rep = activity.report(1)
    assert rep["events"] == 2 and rep["errors"] == 1 and rep["by_action"]["mcp:run_eval"]["errors"] == 1
    path = next(activity.log_dir().glob("activity-*.jsonl"))
    assert oct(path.stat().st_mode & 0o777) == "0o600"


def test_track_records_exceptions():
    with pytest.raises(ValueError), activity.track("rest", "/api/x", {"q": 1}):
        raise ValueError("boom")
    (e,) = activity.read_events(1)
    assert not e["ok"] and e["error"] == "ValueError: boom" and e["ms"] is not None


async def test_mcp_tool_calls_are_logged(monkeypatch, store):
    from fastmcp import Client

    from whichapiapi.surfaces import mcp

    monkeypatch.setattr(mcp.core, "_store", lambda s=None: store)
    async with Client(mcp.mcp) as c:
        await c.call_tool("benchmark_boards", {})
    events = activity.read_events(1)
    assert [e["action"] for e in events] == ["benchmark_boards"] and events[0]["surface"] == "mcp"
    assert events[0]["ok"] and json.dumps(events[0]["result"])


def test_disabled(monkeypatch):
    monkeypatch.setenv("WHICHAPIAPI_ACTIVITY_LOG", "0")
    activity.log_event("cli", "x")
    assert activity.read_events(1) == []


async def test_report_feedback_tool(tmp_path, monkeypatch):
    from fastmcp import Client

    from whichapiapi.surfaces import mcp

    inbox = tmp_path / "EXTERNAL_FEEDBACK.md"
    monkeypatch.setenv("WHICHAPIAPI_FEEDBACK_FILE", str(inbox))
    async with Client(mcp.mcp) as c:
        await c.call_tool(
            "report_feedback",
            {"summary": "rank_models picks a retired model", "what_happened": "top pick 404s on OpenRouter",
             "category": "wrong-data", "tool": "rank_models", "evidence": "key sk-live-abcdefghijklmnop1234"},
        )  # fmt: skip
    text = inbox.read_text()
    assert "rank_models picks a retired model" in text and "**Category:** wrong-data" in text
    assert "abcdefghijklmnop" not in text and "**Status:** open" in text
