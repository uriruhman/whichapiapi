import json

import httpx

from whichapiapi.implementer.apply import N8nClient, plan_retries, retry_fix, rollback


def _wf(fix):
    wf = json.loads((fix / "n8n_workflow.json").read_text())
    wf["id"] = "wf1"
    wf["settings"] = {"executionOrder": "v1", "binaryMode": "separate"}
    return wf


def test_plan_targets_executing_nodes_and_skips_protected(fix):
    changes = {c.node: c.because for c in plan_retries(_wf(fix))}
    assert "Chain" not in changes  # its two models are a fallback pair already
    assert "Call OpenAI" not in changes and "Local" not in changes  # retry on / self-hosted
    assert changes["Lonely model"] == ["Lonely model"] and "Transcribe" in changes


def test_retry_fix_dry_run_apply_verify_and_rollback(fix, tmp_path):
    state = {"wf": _wf(fix), "puts": []}

    def handler(request):
        if request.method == "GET" and request.url.path.endswith("/workflows"):
            return httpx.Response(200, json={"data": [{"id": "wf1", "name": state["wf"]["name"]}]})
        if request.method == "GET":
            return httpx.Response(200, json=state["wf"])
        body = json.loads(request.content)
        assert set(body["settings"]) == {"executionOrder"}  # unsupported settings are not sent
        state["puts"].append(body)
        state["wf"] = {**state["wf"], **body}
        return httpx.Response(200, json=state["wf"])

    client = N8nClient("http://n8n/api/v1", "k", client=httpx.Client(transport=httpx.MockTransport(handler)))
    dry = retry_fix(client, "Synthetic", dry_run=True)
    assert (
        dry["changes"] and not dry["applied"] and not state["puts"] and '"retryOnFail": true' in dry["diff"]
    )
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "wf.json").write_text(json.dumps(_wf(fix)))
    done = retry_fix(client, "wf1", dry_run=False, repo_dir=repo)
    assert done["applied"] and len(state["puts"]) == 1
    git_copy = json.loads((repo / "wf.json").read_text())
    assert next(n for n in git_copy["nodes"] if n["name"] == "Lonely model")["retryOnFail"] is True
    assert retry_fix(client, "wf1", dry_run=True)["changes"] == []  # idempotent
    rollback(client, __import__("pathlib").Path(done["backup"]))
    assert not next(n for n in state["wf"]["nodes"] if n["name"] == "Lonely model").get("retryOnFail")


def test_plan_and_apply_fallbacks():
    from whichapiapi.implementer.apply import apply_fallbacks, plan_fallbacks

    wf = {
        "name": "w",
        "nodes": [
            {"name": "Chain A", "type": "@n8n/n8n-nodes-langchain.chainLlm", "parameters": {}},
            {"name": "LM A", "type": "@n8n/n8n-nodes-langchain.lmChatOpenRouter", "typeVersion": 1, "position": [0, 0],
             "parameters": {"model": "gpt-5.5"}, "credentials": {"openRouterApi": {"id": "1", "name": "main"}}},
            {"name": "Chain B", "type": "@n8n/n8n-nodes-langchain.chainLlm", "parameters": {"needsFallback": True}},
            {"name": "LM B1", "type": "@n8n/n8n-nodes-langchain.lmChatOpenRouter", "parameters": {}},
            {"name": "LM B2", "type": "@n8n/n8n-nodes-langchain.lmChatOpenRouter", "parameters": {}},
        ],
        "connections": {
            "LM A": {"ai_languageModel": [[{"node": "Chain A", "type": "ai_languageModel", "index": 0}]]},
            "LM B1": {"ai_languageModel": [[{"node": "Chain B", "type": "ai_languageModel", "index": 0}]]},
            "LM B2": {"ai_languageModel": [[{"node": "Chain B", "type": "ai_languageModel", "index": 1}]]},
        },
    }  # fmt: skip
    cred = {"openAiApi": {"id": "2", "name": "fallback channel"}}
    plans = plan_fallbacks(wf, "deepseek-v4.1-flash", cred, base_url="https://api.reseller.test/v1")
    assert [p.chain for p in plans] == ["Chain A"]
    out = apply_fallbacks(wf, plans)
    new = next(n for n in out["nodes"] if n["name"] == "Fallback · Chain A")
    assert new["credentials"] == cred and new["parameters"]["model"]["value"] == "deepseek-v4.1-flash"
    assert (
        new["type"].endswith("lmChatOpenAi")
        and new["parameters"]["options"]["baseURL"] == "https://api.reseller.test/v1"
    )
    assert out["connections"]["Fallback · Chain A"]["ai_languageModel"][0][0] == {
        "node": "Chain A",
        "type": "ai_languageModel",
        "index": 1,
    }
    assert next(n for n in out["nodes"] if n["name"] == "Chain A")["parameters"]["needsFallback"] is True
    assert plan_fallbacks(out, "x", cred) == []  # idempotent
