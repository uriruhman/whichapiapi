"""Assertion types (promptfoo-compatible subset, ADR-0002). Each returns a GradingResult; none raise."""

from __future__ import annotations

import importlib.util
import json
import re
import textwrap
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import jinja2
import jiwer
import jsonschema
from jinja2.sandbox import SandboxedEnvironment
from pydantic import BaseModel

from whichapiapi.evaluator.whisper_normalizers import BasicTextNormalizer, EnglishTextNormalizer
from whichapiapi.schema.suite import Assertion, inside

# Sandboxed: suites may come from someone else; prompts are plain text, so no autoescaping.
_jinja = SandboxedEnvironment(undefined=jinja2.ChainableUndefined, autoescape=False)

# Normalized WER, as STT benchmarks report it. English: OpenAI Whisper's EnglishTextNormalizer (vendored, MIT) — "Mr."
# = "mister", spelled-out numbers = digits, British/American spellings, contractions. Other languages: Whisper's
# BasicTextNormalizer (case, punctuation, symbols) plus "ё" = "е".
_EN_NORM = EnglishTextNormalizer()
_BASIC_NORM = BasicTextNormalizer()


def normalize_for_wer(text: str, mode: str = "auto", reference: str = "") -> str:
    """mode: auto (English normalizer when the reference is Latin script, else basic) | en | basic."""
    if mode == "auto":
        letters = [c for c in (reference or text) if c.isalpha()]
        mode = "en" if letters and sum(c.isascii() for c in letters) / len(letters) > 0.9 else "basic"
    if mode == "en":
        return _EN_NORM(text)
    joined = re.sub(
        r"(?<=\w)[-'’](?=\w)", "", text
    )  # "Wi-Fi" = "WiFi" (the basic normalizer splits on hyphens)
    return " ".join(_BASIC_NORM(joined).replace("ё", "е").split())


def render(template: str, vars_: dict[str, Any]) -> str:
    return _jinja.from_string(template).render(**vars_)


class GradingResult(BaseModel):
    type: str
    passed: bool
    score: float
    reason: str = ""
    weight: float = 1.0
    metric: str | None = None
    cost: float = 0.0  # judge spend, if any


JudgeFn = Callable[[str, str, Assertion], Awaitable[GradingResult]]


@dataclass
class AssertContext:
    vars: dict[str, Any]
    base_dir: Path
    latency_ms: float = 0.0
    cost: float | None = None
    judge: JudgeFn | None = None
    extra: dict[str, Any] = field(default_factory=dict)


def _res(a: Assertion, passed: bool, score: float | None = None, reason: str = "") -> GradingResult:
    return GradingResult(
        type=a.type,
        passed=passed,
        score=(1.0 if passed else 0.0) if score is None else float(score),
        reason=reason,
        weight=a.weight,
        metric=a.metric,
    )


def _as_text(value: Any, vars_: dict[str, Any]) -> str:
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False)
    return render(str(value), vars_)


def extract_json(text: str) -> Any:
    """First JSON object/array in text (handles ```json fences). Raises ValueError if none."""
    s = text.strip()
    fence = re.search(r"```(?:json)?\s*(.*?)```", s, re.S)
    if fence:
        s = fence.group(1).strip()
    try:
        return json.loads(s)
    except json.JSONDecodeError:
        pass
    dec = json.JSONDecoder()
    for i, ch in enumerate(s):
        if ch in "{[":
            try:
                return dec.raw_decode(s[i:])[0]
            except json.JSONDecodeError:
                continue
    raise ValueError("no JSON found")


def _schema_check(obj: Any, schema: Any) -> str | None:
    if not schema:
        return None
    try:
        jsonschema.validate(obj, schema)
    except jsonschema.ValidationError as e:
        return f"schema: {e.message[:200]}"
    return None


def _load_python(ref: str, base: Path) -> Callable[..., Any]:
    spec_str = ref.removeprefix("file://")
    path_str, _, fn_name = spec_str.partition(":")
    path = inside(base, path_str)
    spec = importlib.util.spec_from_file_location(f"whichapiapi_assert_{path.stem}", path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return getattr(mod, fn_name or "get_assert")


def _python_result(a: Assertion, value: Any) -> GradingResult:
    if isinstance(value, dict):
        passed = bool(value.get("pass", value.get("pass_", False)))
        score = float(value.get("score", 1.0 if passed else 0.0))
        return _res(a, passed, score, str(value.get("reason", "")))
    if isinstance(value, bool):
        return _res(a, value)
    if isinstance(value, (int, float)):
        thr = a.threshold if a.threshold is not None else 0.0
        passed = value >= thr if a.threshold is not None else value > thr
        return _res(a, passed, float(value), f"score {value}")
    return _res(a, False, 0.0, f"python assertion returned {type(value).__name__}")


def _run_python(a: Assertion, output: str, ctx: AssertContext) -> GradingResult:
    context = {"vars": ctx.vars, "latency_ms": ctx.latency_ms, "cost": ctx.cost, **ctx.extra}
    code = str(a.value)
    if code.startswith("file://"):
        fn = _load_python(code, ctx.base_dir)
        return _python_result(a, fn(output, context))
    if "\n" not in code.strip() and not code.lstrip().startswith("return "):
        return _python_result(a, eval(code, {"json": json, "re": re}, {"output": output, "context": context}))
    body = textwrap.indent(textwrap.dedent(code), "    ")
    ns: dict[str, Any] = {"json": json, "re": re}
    exec(f"def _fn(output, context):\n{body}", ns)
    return _python_result(a, ns["_fn"](output, context))


async def run_assertion(a: Assertion, output: str, ctx: AssertContext) -> GradingResult:
    t = a.type
    negate = t.startswith("not-")
    base = t.removeprefix("not-")
    try:
        if base == "llm-rubric":
            if ctx.judge is None:  # judging disabled: exclude from the score instead of failing
                return GradingResult(type=t, passed=True, score=1.0, weight=0.0, reason="skipped: no judge")
            return await ctx.judge(render(str(a.value), ctx.vars), output, a)
        if base == "python":
            r = _run_python(a, output, ctx)
        elif base == "equals":
            exp = a.value
            if isinstance(exp, (dict, list)):
                try:
                    ok = extract_json(output) == exp
                except ValueError:
                    ok = False
            else:
                ok = output.strip() == _as_text(exp, ctx.vars).strip()
            r = _res(a, ok, reason="" if ok else "not equal")
        elif base in ("contains", "icontains"):
            needle = _as_text(a.value, ctx.vars)
            ok = needle.lower() in output.lower() if base == "icontains" else needle in output
            r = _res(a, ok, reason="" if ok else f"missing {needle[:60]!r}")
        elif base in ("contains-any", "contains-all"):
            items = [_as_text(v, ctx.vars) for v in (a.value or [])]
            hits = [i for i in items if i in output]
            ok = bool(hits) if base == "contains-any" else len(hits) == len(items)
            r = _res(a, ok, reason=f"{len(hits)}/{len(items)} present")
        elif base == "regex":
            ok = re.search(_as_text(a.value, ctx.vars), output, re.S) is not None
            r = _res(a, ok, reason="" if ok else "no regex match")
        elif base in ("is-json", "contains-json"):
            try:
                obj = json.loads(output) if base == "is-json" else extract_json(output)
                err = _schema_check(obj, a.value)
                r = _res(a, err is None, reason=err or "")
            except (ValueError, json.JSONDecodeError) as e:
                r = _res(a, False, reason=f"invalid JSON: {str(e)[:100]}")
        elif base == "cost":
            if ctx.cost is None:
                r = _res(a, False, 0.0, "cost unknown")
            else:
                ok = a.threshold is None or ctx.cost <= a.threshold
                r = _res(a, ok, reason=f"${ctx.cost:.6f}")
        elif base == "latency":
            ok = a.threshold is None or ctx.latency_ms <= a.threshold
            r = _res(a, ok, reason=f"{ctx.latency_ms:.0f} ms")
        elif base == "wer":
            reference = _as_text(a.value, ctx.vars).strip()
            if not reference:
                r = _res(a, False, 0.0, "empty reference transcript")
            else:
                mode = getattr(a, "normalize", "auto")  # auto | en | basic | none (False = none)
                if mode in (False, "none"):
                    rate = jiwer.wer(reference, output.strip())
                else:
                    ref, hyp = (normalize_for_wer(x, str(mode), reference) for x in (reference, output))
                    rate = jiwer.wer(ref, hyp) if ref else float(bool(hyp))
                ok = a.threshold is None or rate <= a.threshold
                r = _res(a, ok, max(0.0, 1.0 - rate), f"WER {rate:.2%}")
        elif base == "javascript":
            return GradingResult(
                type=t, passed=True, score=1.0, weight=0.0, reason="javascript not supported; skipped"
            )
        else:
            return _res(a, False, 0.0, f"unknown assertion type {t!r}")
    except Exception as e:  # a broken user assertion must not kill the run
        return _res(a, False, 0.0, f"assertion error: {type(e).__name__}: {str(e)[:200]}")
    if negate:
        r.passed = not r.passed
        r.score = 1.0 - r.score
        r.type = t
    return r


def combine(results: list[GradingResult], threshold: float | None = None) -> tuple[bool, float]:
    """Weighted score; pass = all passed, or weighted score >= test threshold if one is set."""
    weighted = [r for r in results if r.weight > 0]
    if not weighted:
        return True, 1.0
    total_w = sum(r.weight for r in weighted)
    score = sum(r.score * r.weight for r in weighted) / total_w
    passed = score >= threshold if threshold is not None else all(r.passed for r in weighted)
    return passed, round(score, 4)
