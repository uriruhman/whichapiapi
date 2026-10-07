"""Trust-boundary checks: suites may come from someone else, the MCP server may be reachable over HTTP."""

import jinja2
import pytest
from starlette.testclient import TestClient

from whichapiapi.evaluator.assertions import render
from whichapiapi.schema.suite import load_suite
from whichapiapi.transports.base import redact


def test_templates_are_sandboxed():
    assert render("hi {{ name }}", {"name": "Ann"}) == "hi Ann"
    with pytest.raises(jinja2.exceptions.SecurityError):
        render("{{ ''.__class__.__mro__[1].__subclasses__() }}", {})


def test_file_refs_cannot_leave_the_suite_dir(tmp_path):
    (tmp_path / "secret.txt").write_text("TOKEN=abc")
    suite = tmp_path / "suite"
    suite.mkdir()
    (suite / "suite.yaml").write_text(
        "prompts: ['{{x}}']\nproviders: [mock:echo]\ntests: [{vars: {x: 'file://../secret.txt'}}]\n"
    )
    with pytest.raises(ValueError, match="inside the suite directory"):
        load_suite(suite / "suite.yaml")


def test_redact_strips_keys():
    msg = "401: Incorrect API key provided: sk-live-1234567890abcdef, also tvly-ABCDEFGHIJKLMNOP"
    out = redact(msg, "sk-live-1234567890abcdef")
    assert "1234567890" not in out and "ABCDEFGHIJ" not in out
    assert redact("key=mysecretvalue42 failed", "mysecretvalue42") == "key=*** failed"


def test_mcp_suite_roots(tmp_path, monkeypatch):
    from whichapiapi.surfaces import mcp

    monkeypatch.setenv("WHICHAPIAPI_SUITE_ROOTS", str(tmp_path))
    assert mcp._suite(str(tmp_path / "a/suite.yaml")).startswith(str(tmp_path))
    with pytest.raises(ValueError, match="outside the allowed roots"):
        mcp._suite("/etc/passwd")


def test_http_refuses_public_bind_without_token(monkeypatch):
    from whichapiapi.surfaces import mcp

    monkeypatch.delenv("WHICHAPIAPI_MCP_TOKEN", raising=False)
    with pytest.raises(SystemExit, match="refusing"):
        mcp.serve(http=True, host="0.0.0.0")
    assert mcp._loopback("127.0.0.1") and mcp._loopback("::1") and not mcp._loopback("0.0.0.0")


def test_webhook_token_and_size(monkeypatch, store):
    from whichapiapi.surfaces import mcp

    monkeypatch.setenv("WHICHAPIAPI_WEBHOOK_TOKEN", "s3cret")
    client = TestClient(mcp.mcp.http_app())
    url = "/webhooks/changedetection"
    assert client.post(url, json={"watch_url": "https://x.dev"}).status_code == 401
    assert client.post(url + "?token=nope", json={}).status_code == 401
    big = {"diff_added": "x" * (mcp.WEBHOOK_MAX_BYTES + 1)}
    assert client.post(url + "?token=s3cret", json=big).status_code == 413


def test_rest_endpoints_and_token(monkeypatch, store, fix):
    import json as _json

    from whichapiapi.surfaces import mcp

    monkeypatch.setattr(mcp.core, "_store", lambda s=None: store)
    client = TestClient(mcp.mcp.http_app())
    monkeypatch.delenv("WHICHAPIAPI_MCP_TOKEN", raising=False)
    assert client.get("/api/offers?query=gpt&limit=3").status_code == 200
    assert "models" in client.get("/api/rank?preset=value").json()
    wf = (fix / "n8n_workflow.json").read_text()
    rep = client.post("/api/audit/n8n", content=wf).json()
    assert rep["api_calls"] == 8
    assert client.post("/api/audit/n8n", content=_json.dumps({"x": 1})).status_code == 400
    monkeypatch.setenv("WHICHAPIAPI_MCP_TOKEN", "t0k")
    assert client.get("/api/rank").status_code == 401
    assert client.get("/api/rank", headers={"Authorization": "Bearer t0k"}).status_code == 200
