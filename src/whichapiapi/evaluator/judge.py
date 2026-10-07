"""LLM-as-judge for `llm-rubric` (promptfoo-compatible semantics: judge returns {reason, pass, score})."""

from __future__ import annotations

from typing import Any

from whichapiapi.evaluator.assertions import GradingResult, extract_json
from whichapiapi.schema.suite import Assertion

JUDGE_SYSTEM = """You are a strict, fair grader. You grade an OUTPUT against a RUBRIC written by the user.
Read the rubric carefully; it may contain reference material (for example a source text) and criteria.
Judge only what the rubric asks. Do not reward length or style unless the rubric asks for it.
Score from 0.0 (fails every criterion) to 1.0 (fully satisfies all criteria); use intermediate values.
"pass" is true only if the output is acceptable by the rubric's standard.
Respond with ONLY a JSON object: {"reason": "<one or two sentences>", "score": <0..1>, "pass": <true|false>}"""

JUDGE_SCHEMA: dict[str, Any] = {
    "type": "json_schema",
    "json_schema": {
        "name": "grade",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "reason": {"type": "string"},
                "score": {"type": "number"},
                "pass": {"type": "boolean"},
            },
            "required": ["reason", "score", "pass"],
            "additionalProperties": False,
        },
    },
}


def judge_messages(rubric: str, output: str) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": JUDGE_SYSTEM},
        {"role": "user", "content": f"<RUBRIC>\n{rubric}\n</RUBRIC>\n\n<OUTPUT>\n{output}\n</OUTPUT>"},
    ]


def parse_grade(text: str, a: Assertion, cost: float = 0.0) -> GradingResult:
    try:
        obj = extract_json(text)
        score = max(0.0, min(1.0, float(obj.get("score", 1.0 if obj.get("pass") else 0.0))))
        passed = bool(obj.get("pass", score >= 0.5))
        if a.threshold is not None:
            passed = score >= a.threshold
        return GradingResult(
            type=a.type,
            passed=passed,
            score=score,
            reason=str(obj.get("reason", ""))[:500],
            weight=a.weight,
            metric=a.metric,
            cost=cost,
        )
    except (ValueError, AttributeError, TypeError) as e:
        return GradingResult(
            type=a.type,
            passed=False,
            score=0.0,
            reason=f"unparseable judge output: {e}",
            weight=a.weight,
            cost=cost,
        )


PAIRWISE_SYSTEM = """You compare two answers to the same TASK and pick the better one for the person who wrote the task.
Judge correctness, completeness and how well each follows the task's instructions; ignore length and style unless
the task asks for them. If they are equally good (or equally bad), answer "tie".
Respond with ONLY a JSON object: {"reason": "<one or two sentences>", "winner": "first" | "second" | "tie"}"""

PAIRWISE_SCHEMA: dict[str, Any] = {
    "type": "json_schema",
    "json_schema": {
        "name": "verdict",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "reason": {"type": "string"},
                "winner": {"type": "string", "enum": ["first", "second", "tie"]},
            },
            "required": ["reason", "winner"],
            "additionalProperties": False,
        },
    },
}


def pairwise_messages(
    task: str, first: str, second: str, criteria: str | None = None
) -> list[dict[str, str]]:
    extra = f"\n\n<CRITERIA>\n{criteria}\n</CRITERIA>" if criteria else ""
    return [
        {"role": "system", "content": PAIRWISE_SYSTEM},
        {
            "role": "user",
            "content": f"<TASK>\n{task}\n</TASK>{extra}\n\n<FIRST>\n{first}\n</FIRST>\n\n<SECOND>\n{second}\n</SECOND>",
        },
    ]


def parse_winner(text: str) -> str | None:
    try:
        w = str(extract_json(text).get("winner", "")).lower()
    except (ValueError, AttributeError, TypeError):
        return None
    return w if w in ("first", "second", "tie") else None
