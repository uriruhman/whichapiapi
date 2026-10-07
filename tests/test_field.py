import json

from whichapiapi.evaluator import field


def test_field_stats_from_tributary_log(tmp_path):
    rows = [
        {
            "type": "call",
            "job_id": "j1",
            "provider": "reseller",
            "model": "gpt-6-sol",
            "status": "timeout",
            "ms": 120000,
        },
        {
            "type": "call",
            "job_id": "j1",
            "provider": "reseller",
            "model": "grok-4.7",
            "status": "ok",
            "ms": 60000,
        },
        {
            "type": "call",
            "job_id": "j2",
            "provider": "reseller",
            "model": "gpt-6-sol",
            "status": "ok",
            "ms": 80000,
        },
        {
            "type": "call",
            "job_id": "j3",
            "provider": "reseller",
            "model": "gpt-6-sol",
            "status": "http_502",
            "ms": 10,
        },
        {"type": "rating", "job_id": "j1", "verdict": "bad"},
        {"type": "rating", "job_id": "j2", "verdict": "good"},
        {"type": "measure", "job_id": "j2", "model": "gpt-6-sol"},
    ]
    log = tmp_path / "usage.jsonl"
    log.write_text("\n".join(json.dumps({"ts": "2099-01-01T00:00:00Z", **r}) for r in rows) + "\nnot json\n")
    got = field.stats(paths=[log])
    sol, grok = got["reseller:gpt-6-sol"], got["reseller:grok-4.7"]
    assert (sol["calls"], sol["ok"], sol["timeouts"], sol["errors"], sol["p50_ms"]) == (3, 1, 1, 1, 80000)
    assert (sol["good"], sol["bad"], grok["bad"]) == (
        1,
        0,
        1,
    )  # j1's verdict belongs to the model that answered
    assert sol["reliable"] is None  # too few calls to judge
    assert field.stats(days=1, paths=[tmp_path / "missing.jsonl"]) == {}
