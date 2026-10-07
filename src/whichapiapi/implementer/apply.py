"""Phase 5, step 1: apply audit fixes to n8n workflows — as a reviewed diff, with a backup and one-command rollback.

The first fix is the safest one: "Retry On Fail" on every AI call that has no retry, no error output and no fallback
model. For a LangChain model sub-node the retry goes on the chain/agent node that actually executes it. Nothing is
sent to n8n unless `dry_run=False`; before sending, the live workflow is saved to
`$WHICHAPIAPI_HOME/n8n-backups/<id>-<timestamp>.json`, and after sending it is read back to verify the change landed.
If the workflow also lives in a git checkout (a JSON file with the same name), the same change is written there so git
stays the source of truth.
"""

from __future__ import annotations

import copy
import difflib
import json
import os
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx
from pydantic import BaseModel, Field

from whichapiapi.implementer.n8n import _api_use

RETRY = {"retryOnFail": True, "maxTries": 3, "waitBetweenTries": 5000}
# settings the public API accepts on PUT (others make it answer 400)
_API_SETTINGS = {
    "saveExecutionProgress", "saveManualExecutions", "saveDataErrorExecution", "saveDataSuccessExecution",
    "executionTimeout", "errorWorkflow", "timezone", "executionOrder", "callerPolicy", "timeSavedPerExecution",
}  # fmt: skip


class Change(BaseModel):
    node: str  # the node that gets the setting
    because: list[str]  # the AI node(s) it protects
    before: dict[str, Any] = Field(default_factory=dict)
    after: dict[str, Any] = Field(default_factory=dict)


def plan_retries(wf: dict[str, Any]) -> list[Change]:
    """Retry On Fail for unprotected AI calls (see module docstring). Idempotent: protected nodes are skipped."""
    by_name = {n.get("name"): n for n in wf.get("nodes") or []}
    targets: dict[str, list[str]] = {}
    for node in wf.get("nodes") or []:
        use = _api_use(node)
        if use is None or use.local:
            continue
        target = node
        links = ((wf.get("connections") or {}).get(node.get("name")) or {}).get("ai_languageModel") or []
        for branch in links:
            for link in branch or []:
                target = by_name.get(link.get("node"), node)
        if len((links and links[0]) or []) and _fallback_wired(wf, target.get("name")):
            continue  # a second model is already attached to this chain
        if target.get("retryOnFail") or target.get("onError") not in (None, "stop", "stopWorkflow"):
            continue
        targets.setdefault(target["name"], []).append(node["name"])
    return [
        Change(
            node=name,
            because=sources,
            before={k: by_name[name].get(k) for k in RETRY},
            after=dict(RETRY),
        )
        for name, sources in targets.items()
    ]


def _fallback_wired(wf: dict[str, Any], chain: str | None) -> bool:
    models = [
        src
        for src, outs in (wf.get("connections") or {}).items()
        for branch in outs.get("ai_languageModel") or []
        for link in branch or []
        if link.get("node") == chain
    ]
    return len(models) > 1


def apply_changes(wf: dict[str, Any], changes: list[Change]) -> dict[str, Any]:
    out = copy.deepcopy(wf)
    wanted = {c.node: c.after for c in changes}
    for node in out.get("nodes") or []:
        if node.get("name") in wanted:
            node.update(wanted[node["name"]])
    return out


def diff(before: dict[str, Any], after: dict[str, Any], changes: list[Change]) -> str:
    """Unified diff of the changed nodes only (what a reviewer needs to read)."""
    names = {c.node for c in changes}

    def dump(wf: dict[str, Any]) -> list[str]:
        nodes = [n for n in wf.get("nodes") or [] if n.get("name") in names]
        return json.dumps(nodes, indent=2, ensure_ascii=False, sort_keys=True).splitlines(keepends=True)

    return "".join(difflib.unified_diff(dump(before), dump(after), f"{before.get('name')} (now)", "(after)"))


class N8nClient:
    """n8n public API (`X-N8N-API-KEY`). Base URL like http://127.0.0.1:5678/api/v1."""

    def __init__(
        self, base_url: str | None = None, api_key: str | None = None, client: httpx.Client | None = None
    ):
        self.base = (base_url or os.environ.get("N8N_API_URL") or "http://127.0.0.1:5678/api/v1").rstrip("/")
        key = api_key or os.environ.get("N8N_API_KEY")
        if not key:
            raise RuntimeError("N8N_API_KEY is not set (see ~/.config/secrets/n8n.env)")
        self.http = client or httpx.Client(timeout=60)
        self.headers = {"X-N8N-API-KEY": key, "Accept": "application/json"}

    def list(self) -> list[dict[str, Any]]:
        out, cursor = [], None
        while True:
            params = {"limit": 100} | ({"cursor": cursor} if cursor else {})
            r = self.http.get(f"{self.base}/workflows", params=params, headers=self.headers)
            r.raise_for_status()
            body = r.json()
            out += body.get("data") or []
            cursor = body.get("nextCursor")
            if not cursor:
                return out

    def get(self, wf_id: str) -> dict[str, Any]:
        r = self.http.get(f"{self.base}/workflows/{wf_id}", headers=self.headers)
        r.raise_for_status()
        return r.json()

    def put(self, wf_id: str, wf: dict[str, Any]) -> dict[str, Any]:
        body = {
            "name": wf["name"],
            "nodes": wf["nodes"],
            "connections": wf.get("connections") or {},
            "settings": {k: v for k, v in (wf.get("settings") or {}).items() if k in _API_SETTINGS},
        }
        if wf.get("staticData") is not None:
            body["staticData"] = wf["staticData"]
        r = self.http.put(f"{self.base}/workflows/{wf_id}", json=body, headers=self.headers)
        r.raise_for_status()
        return r.json()

    def find(self, name_or_id: str) -> dict[str, Any]:
        for w in self.list():
            if name_or_id in (w.get("id"), w.get("name")):
                return self.get(w["id"])
        matches = [w for w in self.list() if name_or_id.lower() in (w.get("name") or "").lower()]
        if len(matches) == 1:
            return self.get(matches[0]["id"])
        raise ValueError(f"{len(matches)} workflows match {name_or_id!r}: {[m['name'] for m in matches][:8]}")


def backup_dir() -> Path:
    return Path(os.environ.get("WHICHAPIAPI_HOME", Path.home() / ".local/share/whichapiapi")) / "n8n-backups"


def backup(wf: dict[str, Any]) -> Path:
    d = backup_dir()
    d.mkdir(mode=0o700, parents=True, exist_ok=True)
    path = d / f"{wf['id']}-{datetime.now(UTC):%Y%m%dT%H%M%SZ}.json"
    path.write_text(json.dumps(wf, ensure_ascii=False, indent=2), encoding="utf-8")
    path.chmod(0o600)
    return path


def sync_git_copy(repo_dir: Path, workflow_name: str, changes: list[Change]) -> Path | None:
    """Apply the same node changes to the JSON file in a git checkout that holds this workflow (matched by name)."""
    for path in sorted(repo_dir.rglob("*.json")):
        if "node_modules" in path.parts:
            continue
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (ValueError, OSError):
            continue
        if isinstance(data, dict) and data.get("name") == workflow_name and "nodes" in data:
            ending = (
                "\n" if path.read_text(encoding="utf-8").endswith("\n") else ""
            )  # keep the file's own style
            text = json.dumps(apply_changes(data, changes), ensure_ascii=False, indent=2) + ending
            path.write_text(text, encoding="utf-8")
            return path
    return None


def retry_fix(
    client: N8nClient, workflow: str, dry_run: bool = True, repo_dir: Path | None = None
) -> dict[str, Any]:
    """Plan (and with dry_run=False, apply) Retry On Fail for one workflow's unprotected AI calls."""
    live = client.find(workflow)
    changes = plan_retries(live)
    after = apply_changes(live, changes)
    report: dict[str, Any] = {
        "workflow": live["name"],
        "id": live["id"],
        "changes": [c.model_dump() for c in changes],
        "diff": diff(live, after, changes),
        "applied": False,
    }
    if dry_run or not changes:
        return report
    report["backup"] = str(backup(live))
    client.put(live["id"], after)
    check = client.get(live["id"])
    by_name = {n["name"]: n for n in check.get("nodes") or []}
    missing = [
        c.node for c in changes if any(by_name.get(c.node, {}).get(k) != v for k, v in c.after.items())
    ]
    report["applied"] = not missing
    report["not_applied"] = missing
    if repo_dir and not missing:
        synced = sync_git_copy(repo_dir, live["name"], changes)
        report["git_file"] = str(synced) if synced else None
    return report


def rollback(client: N8nClient, backup_path: Path) -> dict[str, Any]:
    wf = json.loads(backup_path.read_text(encoding="utf-8"))
    client.put(wf["id"], wf)
    return {"workflow": wf["name"], "id": wf["id"], "restored_from": str(backup_path)}


# ---------------------------------------------------------------- step 2: a fallback model on another channel

OPENAI_CHAT = "@n8n/n8n-nodes-langchain.lmChatOpenAi"


class FallbackPlan(BaseModel):
    chain: str
    primary: str  # the model node already attached
    new_node: str
    model: str
    credential: dict[
        str, Any
    ]  # {"openAiApi": {"id", "name"}}: must be a DIFFERENT channel/key than the primary
    node_type: str = OPENAI_CHAT  # OpenAI Chat Model: any OpenAI-compatible channel via its Base URL
    type_version: float = 1.2  # chat completions (later versions may default to the Responses API)
    base_url: str | None = None


def plan_fallbacks(
    wf: dict[str, Any], model: str, credential: dict[str, Any], base_url: str | None = None
) -> list[FallbackPlan]:
    """For every chain/agent fed by exactly one LangChain chat model: add a second model node on `credential` and
    switch the chain's "Enable Fallback Model" on. n8n calls it only when the first model fails (quota, outage,
    rate limit), so it costs nothing while the primary works. Idempotent: chains with two models are skipped."""
    by_name = {n.get("name"): n for n in wf.get("nodes") or []}
    feeds: dict[str, list[str]] = {}
    for src, outs in (wf.get("connections") or {}).items():
        for branch in outs.get("ai_languageModel") or []:
            for link in branch or []:
                feeds.setdefault(link.get("node"), []).append(src)
    plans = []
    for chain, models in feeds.items():
        node = by_name.get(chain) or {}
        if len(models) != 1 or node.get("type", "").rsplit(".", 1)[-1] not in ("chainLlm", "agent"):
            continue
        plans.append(
            FallbackPlan(
                chain=chain,
                primary=models[0],
                new_node=f"Fallback · {chain}"[:60],
                model=model,
                credential=credential,
                base_url=base_url,
            )
        )
    return plans


def apply_fallbacks(wf: dict[str, Any], plans: list[FallbackPlan]) -> dict[str, Any]:
    out = copy.deepcopy(wf)
    by_name = {n.get("name"): n for n in out.get("nodes") or []}
    for p in plans:
        x, y = (by_name[p.primary].get("position") or [0, 0])[:2]
        params: dict[str, Any] = (
            {"model": {"__rl": True, "value": p.model, "mode": "id"}, "options": {}}
            if p.node_type == OPENAI_CHAT
            else {"model": p.model, "options": {}}
        )
        if p.base_url:
            params["options"]["baseURL"] = p.base_url
        out["nodes"].append(
            {
                "parameters": params,
                "type": p.node_type,
                "typeVersion": p.type_version,
                "position": [x + 160, y + 160],
                "id": str(uuid.uuid4()),
                "name": p.new_node,
                "credentials": p.credential,
            }
        )
        link = {"node": p.chain, "type": "ai_languageModel", "index": 1}
        out.setdefault("connections", {})[p.new_node] = {"ai_languageModel": [[link]]}
        by_name[p.chain].setdefault("parameters", {})["needsFallback"] = True
    return out


def fallback_fix(
    client: N8nClient,
    workflow: str,
    model: str,
    credential: dict[str, Any],
    base_url: str | None = None,
    dry_run: bool = True,
    repo_dir: Path | None = None,
) -> dict[str, Any]:
    """Plan (and with dry_run=False, apply) a fallback model on another channel for one workflow's chains."""
    live = client.find(workflow)
    plans = plan_fallbacks(live, model, credential, base_url)
    report: dict[str, Any] = {
        "workflow": live["name"],
        "id": live["id"],
        "plans": [p.model_dump() for p in plans],
        "applied": False,
    }
    if dry_run or not plans:
        return report
    report["backup"] = str(backup(live))
    client.put(live["id"], apply_fallbacks(live, plans))
    check = client.get(live["id"])
    names = {n["name"] for n in check.get("nodes") or []}
    report["not_applied"] = [p.chain for p in plans if p.new_node not in names]
    report["applied"] = not report["not_applied"]
    if repo_dir and report["applied"]:
        for path in sorted(repo_dir.rglob("*.json")):
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
            except (ValueError, OSError):
                continue
            if isinstance(data, dict) and data.get("name") == live["name"] and "nodes" in data:
                ending = "\n" if path.read_text(encoding="utf-8").endswith("\n") else ""
                updated = apply_fallbacks(data, plan_fallbacks(data, model, credential, base_url))
                path.write_text(json.dumps(updated, ensure_ascii=False, indent=2) + ending, encoding="utf-8")
                report["git_file"] = str(path)
                break
    return report
