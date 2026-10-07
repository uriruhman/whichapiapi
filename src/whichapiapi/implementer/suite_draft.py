""" "Describe the task → a benchmark": write a runnable eval suite from a task description, example inputs and
acceptance criteria. Meant for agents (MCP `create_suite`): the calling agent does the thinking — turns the owner's
words into a prompt template, cases and criteria — and this module turns that into files the evaluator runs.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import yaml

from whichapiapi.implementer.suite_gen import known_channels

_VAR = re.compile(r"\{\{\s*(\w+)\s*\}\}")


def draft_suite(
    directory: Path,
    title: str,
    prompt: str,
    cases: list[dict[str, Any]],
    candidates: list[str],
    criteria: str = "",
    system: str = "",
    json_output: bool = False,
    capability: str = "llm.chat",
    judge: str | None = None,
    budget_usd: float = 0.10,
    contains_pii: bool = False,
    task: str = "general",
) -> dict[str, Any]:
    """Write suite.yaml, prompt.json and tests.yaml into `directory` (created). `prompt` uses {{var}} placeholders
    that each case fills (`{"vars": {...}}` or a flat dict); a case may carry `expected` (a substring the answer must
    contain). `candidates` are "channel:model" ids (the first is the current choice)."""
    if not candidates:
        raise ValueError("give at least one candidate like 'openrouter:openai/gpt-5-mini'")
    from whichapiapi import channels as chs

    judge = judge or chs.default_judge()
    known = known_channels()
    names = sorted(set(_VAR.findall(prompt)))
    tests = []
    for i, case in enumerate(cases):
        vars_ = dict(
            case.get("vars") or {k: v for k, v in case.items() if k not in ("expected", "description")}
        )
        missing = [n for n in names if n not in vars_]
        if missing:
            raise ValueError(f"case {i + 1} has no value for {missing}")
        test: dict[str, Any] = {"description": case.get("description") or f"case {i + 1}", "vars": vars_}
        if case.get("expected"):
            test["assert"] = [{"type": "icontains", "value": str(case["expected"]), "weight": 0.3}]
        tests.append(test)
    messages = ([{"role": "system", "content": system}] if system else []) + [
        {"role": "user", "content": prompt}
    ]
    asserts: list[dict[str, Any]] = []
    if json_output:
        asserts.append({"type": "is-json", "weight": 0.3})
    asserts.append(
        {"type": "llm-rubric", "value": criteria or f"The answer does this task well: {title}", "weight": 1.0}
    )
    channels = {c.split(":", 1)[0] for c in [*candidates, judge]}
    suite = {
        "description": title,
        "prompts": ["file://prompt.json"],
        "providers": [
            {"id": c, "label": c.split(":", 1)[1], **({"x-whichapiapi": {"current": True}} if i == 0 else {})}
            for i, c in enumerate(candidates)
        ],
        "defaultTest": {"assert": asserts},
        "tests": "file://tests.yaml",
        "x-whichapiapi": {
            "capability": capability,
            "task": task,
            "contains_pii": contains_pii,
            "channels": {c: known[c] for c in sorted(channels) if known.get(c)},
            "judge": {"provider": {"id": judge, "label": "judge"}},
            "budget": {"run_usd": budget_usd},
        },
    }
    unknown = sorted(c for c in channels if not known.get(c) and c != "mock")
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "suite.yaml").write_text(
        yaml.safe_dump(suite, allow_unicode=True, sort_keys=False), encoding="utf-8"
    )
    (directory / "prompt.json").write_text(
        json.dumps(messages, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (directory / "tests.yaml").write_text(
        yaml.safe_dump(tests, allow_unicode=True, sort_keys=False), encoding="utf-8"
    )
    return {
        "suite": str(directory / "suite.yaml"),
        "cases": len(tests),
        "candidates": candidates,
        "unknown_channels": unknown,  # add their base_url/key_env to suite.yaml before running
        "next": "eval_plan(suite) → show the owner the cost → run_eval(suite) after consent",
    }
