"""Audit → eval hand-off: turn one AI node of an n8n workflow into an eval suite skeleton.

The suite reuses the node's own prompt (system/user messages from the chain, agent or OpenAI node, with n8n
expressions turned into suite variables), pits the node's current model against the alternatives the audit proposed,
and leaves the test inputs as placeholders to fill from real n8n executions — so a model switch is decided on this
node's real traffic, not on a leaderboard.
"""

from __future__ import annotations

import json
import re
from typing import Any

import yaml

from whichapiapi.implementer.n8n import _api_use, load


def known_channels() -> dict[str, dict[str, Any]]:
    """Suite channel entries for every built-in and custom channel (see whichapiapi.channels)."""
    from whichapiapi import channels as chs

    return {n: chs.suite_channel(n) for n in chs.all_channels()}  # type: ignore[misc]


_JSON_FIELD = re.compile(r"""\$json(?:\.([A-Za-z_]\w*)|\[['"]([^'"]+)['"]\])""")
_ITEM_FIELD = re.compile(r"""\.json(?:\.([A-Za-z_]\w*)|\[['"]([^'"]+)['"]\])""")


def _judge() -> str:
    from whichapiapi import channels as chs

    return chs.default_judge()


def to_template(value: Any, names: list[str]) -> str:
    """n8n text ("=Hi {{ $json.name }}") → suite template ("Hi {{name}}"); every expression becomes one variable.
    Expressions that aren't a plain field read become `var_N` so the user sees what to fill."""
    text = str(value or "")
    if not text.startswith("="):
        return text
    text = text[1:]

    def repl(m: re.Match[str]) -> str:
        expr = m.group(1)
        field = _JSON_FIELD.search(expr) or _ITEM_FIELD.search(expr)
        name = (field.group(1) or field.group(2)) if field else f"var_{len(names) + 1}"
        name = re.sub(r"\W+", "_", name).strip("_") or f"var_{len(names) + 1}"
        if name not in names:
            names.append(name)
        return "{{" + name + "}}"

    return re.sub(r"\{\{(.*?)\}\}", repl, text, flags=re.S)


def _messages(wf: dict[str, Any], node: dict[str, Any], names: list[str]) -> list[dict[str, str]]:
    """The prompt that reaches this node's model."""
    by_name = {n.get("name"): n for n in wf.get("nodes") or []}
    target = node
    if node.get("type", "").rsplit(".", 1)[-1].startswith(("lmChat", "lmOpenAi")):  # the model is a sub-node
        for link in (
            ((wf.get("connections") or {}).get(node.get("name")) or {}).get("ai_languageModel") or [[]]
        )[0]:
            target = by_name.get(link.get("node"), node)
    params, suffix = target.get("parameters") or {}, target.get("type", "").rsplit(".", 1)[-1]
    msgs: list[dict[str, str]] = []
    if suffix == "chainLlm":
        roles = {"AIMessagePromptTemplate": "assistant", "HumanMessagePromptTemplate": "user"}
        for m in (params.get("messages") or {}).get("messageValues") or []:
            msgs.append(
                {"role": roles.get(m.get("type"), "system"), "content": to_template(m.get("message"), names)}
            )
        msgs.append(
            {"role": "user", "content": to_template(params.get("text", "={{ $json.chatInput }}"), names)}
        )
    elif suffix == "agent":
        if system := (params.get("options") or {}).get("systemMessage"):
            msgs.append({"role": "system", "content": to_template(system, names)})
        msgs.append(
            {"role": "user", "content": to_template(params.get("text", "={{ $json.chatInput }}"), names)}
        )
    elif suffix == "openAi":
        for m in (params.get("messages") or {}).get("values") or []:
            msgs.append({"role": m.get("role") or "user", "content": to_template(m.get("content"), names)})
    if not msgs:
        names.append("input")
        msgs = [
            {"role": "user", "content": "{{input}}  # TODO: paste this node's prompt (not found in the node)"}
        ]
    return msgs


def build_suite(
    data: Any,
    node_name: str,
    alternatives: list[str] | None = None,
    cases: int = 5,
    executions: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """{"suite.yaml", "prompt.json", "tests.yaml"} texts for one node. `alternatives` are "channel:model" ids."""
    for wf in load(data):
        node = next((n for n in wf.get("nodes") or [] if n.get("name") == node_name), None)
        if node is not None:
            break
    else:
        raise ValueError(f"no node named {node_name!r} in the workflows")
    use = _api_use(node)
    if use is None or use.model is None:
        raise ValueError(f"{node_name!r} is not an AI call with a known model")
    names: list[str] = []
    messages = _messages(wf, node, names)
    wants_json = "json" in " ".join(m["content"] for m in messages).lower()
    current = f"{use.provider}:{use.model}"
    ids = [current, *[a for a in alternatives or [] if a != current]]
    channels = {i.split(":", 1)[0] for i in ids}
    suite = {
        "description": f"n8n · {wf.get('name')} · {node_name}: current model vs alternatives, on this node's inputs",
        "prompts": ["file://prompt.json"],
        "providers": [
            {
                "id": i,
                "label": i.split(":", 1)[1],
                **({"x-whichapiapi": {"current": True}} if i == current else {}),
            }
            for i in ids
        ],
        "defaultTest": {
            "assert": [
                *([{"type": "is-json", "weight": 0.3}] if wants_json else []),
                {
                    "type": "llm-rubric",
                    "weight": 0.7 if wants_json else 1.0,
                    "value": "The answer follows every instruction in the prompt and handles this input correctly "
                    "and completely, as the workflow's next step expects. TODO: replace with this node's real "
                    "acceptance criteria.",
                },
            ]
        },
        "tests": "file://tests.yaml",
        "x-whichapiapi": {
            "capability": use.capability,
            "contains_pii": True,
            "channels": {
                c: known_channels().get(c) or {"base_url": "TODO", "key_env": "TODO"}
                for c in sorted(channels)
            },
            "judge": {"provider": {"id": _judge(), "label": "judge"}},
            "budget": {"run_usd": 0.10},
        },
    }
    judge_channel = _judge().partition(":")[0]
    if judge_channel not in channels and known_channels().get(judge_channel):
        suite["x-whichapiapi"]["channels"][judge_channel] = known_channels()[judge_channel]
    real = cases_from_executions(executions or [], _prompt_node(wf, node), names, limit=max(cases, 20))
    tests = real or [
        {"description": f"case {i + 1} — TODO from a real execution", "vars": {n: "TODO" for n in names}}
        for i in range(cases)
    ]
    header = (
        f"# Generated by `whichapiapi audit n8n-suite` from node {node_name!r}. Fill the TODOs with real inputs from\n"
        "# n8n executions (Executions → the run → this node's input), then: whichapiapi eval plan suite.yaml\n"
    )
    return {
        "suite.yaml": header + yaml.safe_dump(suite, allow_unicode=True, sort_keys=False),
        "prompt.json": json.dumps(messages, ensure_ascii=False, indent=2),
        "tests.yaml": yaml.safe_dump(tests, allow_unicode=True, sort_keys=False),
    }


def _prompt_node(wf: dict[str, Any], node: dict[str, Any]) -> str:
    """The node whose input feeds the prompt: the chain/agent a model sub-node is attached to, else the node."""
    for link in (((wf.get("connections") or {}).get(node.get("name")) or {}).get("ai_languageModel") or [[]])[
        0
    ]:
        return link.get("node") or node.get("name")
    return node.get("name")


def cases_from_executions(
    executions: list[dict[str, Any]], prompt_node: str, names: list[str], limit: int = 20
) -> list[dict[str, Any]]:
    """Test cases from real n8n executions (API `includeData=true`): for each run of `prompt_node`, the items it
    received (its parent's output). A variable is the item's field of that name, else the first field of that name in
    any node's output of the same execution (covers `$('Other node').item.json.x`)."""
    cases: list[dict[str, Any]] = []
    for ex in executions:
        run_data = (((ex.get("data") or {}).get("resultData") or {}).get("runData")) or {}
        runs = run_data.get(prompt_node) or []
        if not runs:
            continue
        everywhere: dict[str, Any] = {}
        for node_runs in run_data.values():
            for run in node_runs:
                for item in ((run.get("data") or {}).get("main") or [[]])[0] or []:
                    for k, v in (item.get("json") or {}).items():
                        everywhere.setdefault(k, v)
        for run in runs:
            parent = ((run.get("source") or [{}])[0] or {}).get("previousNode")
            parent_runs = run_data.get(parent) or []
            items = ((parent_runs[-1].get("data") or {}).get("main") or [[]])[0] if parent_runs else []
            for item in items or [{}]:
                data = item.get("json") or {}
                vars_ = {n: data.get(n, everywhere.get(n)) for n in names}
                if any(v is not None for v in vars_.values()):
                    cases.append({"description": f"execution {ex.get('id')}", "vars": vars_})
                if len(cases) >= limit:
                    return cases
    return cases
