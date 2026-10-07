"""Is the model behind a model name the one we pay for? Cheap black-box checks of a (reseller) channel.

A run sends ~20 short calls per watched model and checks:

1. **tokenizer** — billed `prompt_tokens` of "Reply OK." vs "Reply OK." + calibration text. The difference is the
   calibration text's token count under the serving model's tokenizer; it cancels any hidden system prompt the
   channel adds. Compared with local tokenizers (OpenAI o200k, DeepSeek, Qwen, GLM, MiMo); Claude and Gemini have no
   public tokenizer, so they must match their stored baseline and must NOT match any local one. Catches a Chinese
   model behind a Western name, and a vendor swap in general. (Same-tokenizer siblings — Sol vs Luna, Opus vs
   Sonnet — need the fingerprint check.)
2. **hidden_prompt** — billed prompt tokens of "Reply OK." minus what it really is: a channel that injects its own
   system prompt (e.g. a reseller price group that adds ~4.4k tokens of a coding-agent prompt) shows up here.
3. **identity** — "which company trained you": a Western name answering with a Chinese vendor (or vice versa).
   Weak alone (models misreport; hidden prompts steer it), so it only warns.
4. **censorship** — a Tiananmen 1989 question: Western models answer it, Chinese-hosted ones deflect. A Western
   name that deflects is an alert.
5. **fingerprint** — distributions of "pick a random number / animal" at the default temperature plus the median
   reasoning tokens on a fixed puzzle (stronger siblings reason more efficiently: measured Sol ~180 vs Luna ~400).
   Compared with the model's own baseline and, when a cheaper `sibling` is configured, with the sibling's current
   fingerprint: closer to the sibling than to its own past → alert.
6. **metadata** — response id prefix (`resp_`/`chatcmpl-`/`msg_`), `system_fingerprint` presence and usage
   fields; a change vs baseline warns.

Watermarks are not checked: Google's SynthID-Text (also licensed to Anthropic per vendor statements) needs the
vendor's secret key to detect; Anthropic's detection API is in private preview. See docs/INTEGRITY.md.
Pure functions here; `core.arun_integrity` does the calls, the baseline cache and the spend ledger.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from statistics import median
from typing import Any

PREFIX = "Reply OK."
CALIBRATION = [
    "The quick brown fox jumps over the lazy dog while seventeen zebras quietly debate quantum chromodynamics.\n"
    "Съешь же ещё этих мягких французских булок, да выпей чаю — сказал инженер, проверяя логи сервера.\n"
    "敏捷的棕色狐狸跳过了懒狗。def f(x): return {k: v**2 for k, v in x.items() if v % 3 == 0}",
    "Вчерашний гиперпараметрический эксперимент показал: асинхронная репликация незначительно увеличивает "
    "латентность. Pseudopseudohypoparathyroidism, Llanfairpwllgwyngyll, ZXQJ-7731-ÆØÅ; 東京特許許可局長; "
    "SELECT COUNT(*) FROM журнал WHERE статус = 'ошибка' AND ts > NOW() - INTERVAL '7 days';",
]
WHO = "Which company trained the AI model that is answering this message? Reply with only the company name."
CENSOR = [
    "What happened at Tiananmen Square in Beijing in June 1989? Answer in one sentence.",
    "Why are Winnie the Pooh images censored in China? Answer in one sentence.",
]
PERSONA = "In one short sentence: what kind of assistant are you configured to be?"
FP_NUMBER = "Pick a random integer from 1 to 100. Answer with only the number."
FP_ANIMAL = "Name one random animal. Answer with exactly one lowercase word."
REASON = "What is the 37th digit after the decimal point in the decimal expansion of 1/97? Answer with only the digit."
REASON_ANSWER = str((10**37 // 97) % 10)

WESTERN = {"openai", "claude", "gemini"}
VENDOR_WORDS = {
    "openai": ("openai",),
    "claude": ("anthropic",),
    "gemini": ("google", "deepmind"),
    "deepseek": ("deepseek",),
    "qwen": ("alibaba", "qwen", "通义"),
    "glm": ("zhipu", "z.ai", "智谱", "glm"),
    "mimo": ("xiaomi", "小米"),
    "kimi": ("moonshot", "kimi", "月之暗面"),
}
CHINESE_VENDORS = (
    "deepseek",
    "alibaba",
    "qwen",
    "zhipu",
    "智谱",
    "xiaomi",
    "moonshot",
    "baidu",
    "tencent",
    "bytedance",
)
PERSONA_AGENT = re.compile(
    r"\bcli\b|coding|code|software engineer|codebase|codex|claude code|cursor|ide\b", re.I
)
CENSOR_FACTS = re.compile(
    r"crack|suppress|massacre|troop|kill|protest|military|army|tank|martial law|violent|shot|pro-democracy|crushed"
    r"|xi jinping|mock|criticis|satir|resembl",
    re.I,
)
SEVERITY = {"ok": 0, "info": 0, "warn": 1, "alert": 2}


def probes(samples: int = 4) -> list[tuple[str, str, int]]:
    """(probe name, user message, max_tokens) to send for one model."""
    out = [("tok_base", PREFIX, 16)]
    out += [(f"tok_{i}", PREFIX + "\n" + t, 16) for i, t in enumerate(CALIBRATION)]
    out += [("who", WHO, 400), ("persona", PERSONA, 400)]
    out += [(f"censor_{i}", q, 600) for i, q in enumerate(CENSOR)]
    out += fingerprint_probes(samples)
    return out


def fingerprint_probes(samples: int = 4) -> list[tuple[str, str, int]]:
    return (
        [("fp_number", FP_NUMBER, 400)] * samples
        + [("fp_animal", FP_ANIMAL, 400)] * samples
        + [("reason", REASON, 4000)] * 2
    )


def observation(probe: str, status: int, data: Any, headers: Any, latency_s: float) -> dict[str, Any]:
    """The fields we keep from one probe call."""
    d = data if isinstance(data, dict) else {}
    u = d.get("usage") if isinstance(d.get("usage"), dict) else {}
    msg = ((d.get("choices") or [{}])[0] or {}).get("message") or {}
    get = headers.get if headers is not None else (lambda _k: None)
    return {
        "probe": probe,
        "status": status,
        "content": (msg.get("content") or "").strip()[:400] if isinstance(msg.get("content"), str) else "",
        "prompt_tokens": u.get("prompt_tokens"),
        "completion_tokens": u.get("completion_tokens"),
        "reasoning_tokens": (u.get("completion_tokens_details") or {}).get("reasoning_tokens"),
        "cached_tokens": (u.get("prompt_tokens_details") or {}).get("cached_tokens"),
        "usage_keys": sorted(u),
        "id_prefix": _id_prefix(d.get("id")),
        "system_fingerprint": bool(d.get("system_fingerprint")),
        "group": get("x-routing-group"),
        "request_id": get("x-api-request-id") or get("x-oneapi-request-id"),
        "latency_s": round(latency_s, 2),
        "error": (d.get("error") or {}).get("message")
        if isinstance(d.get("error"), dict)
        else d.get("error"),
    }


def _id_prefix(rid: Any) -> str | None:
    if not isinstance(rid, str) or not rid:
        return None
    m = re.match(r"([a-z]+[-_])", rid)
    return m.group(1) if m else "other"


def _by(obs: list[dict[str, Any]], probe: str) -> list[dict[str, Any]]:
    return [o for o in obs if o["probe"] == probe and o["status"] == 200]


def reported_deltas(obs: list[dict[str, Any]]) -> list[int] | None:
    base = _by(obs, "tok_base")
    if not base or base[0]["prompt_tokens"] is None:
        return None
    out = []
    for i in range(len(CALIBRATION)):
        o = _by(obs, f"tok_{i}")
        if not o or o[0]["prompt_tokens"] is None:
            return None
        out.append(o[0]["prompt_tokens"] - base[0]["prompt_tokens"])
    return out


def _match(a: list[int] | None, b: list[int] | None, tol: int = 1) -> bool:
    return bool(a and b and len(a) == len(b) and all(abs(x - y) <= tol for x, y in zip(a, b, strict=True)))


def _norm(word: str) -> str:
    return re.sub(r"[^\w]", "", word.lower())[:20]


def fingerprint(obs: list[dict[str, Any]]) -> dict[str, Any]:
    reason = [o["reasoning_tokens"] for o in _by(obs, "reason") if isinstance(o["reasoning_tokens"], int)]
    return {
        "number": dict(Counter(_norm(o["content"]) for o in _by(obs, "fp_number"))),
        "animal": dict(Counter(_norm(o["content"]) for o in _by(obs, "fp_animal"))),
        "reasoning_median": median(reason) if reason else None,
        "reason_correct": sum(o["content"].strip().rstrip(".") == REASON_ANSWER for o in _by(obs, "reason")),
        "latency_median_s": median([o["latency_s"] for o in obs if o["status"] == 200] or [0]),
    }


def jsd(p: dict[str, int], q: dict[str, int]) -> float:
    """Jensen-Shannon divergence (base 2, 0…1) of two categorical counts."""
    if not p or not q:
        return 0.0
    keys = set(p) | set(q)
    sp, sq = sum(p.values()), sum(q.values())
    pp = {k: p.get(k, 0) / sp for k in keys}
    qq = {k: q.get(k, 0) / sq for k in keys}
    m = {k: (pp[k] + qq[k]) / 2 for k in keys}

    def kl(a: dict[str, float]) -> float:
        return sum(a[k] * math.log2(a[k] / m[k]) for k in keys if a[k] > 0)

    return (kl(pp) + kl(qq)) / 2


def distance(a: dict[str, Any], b: dict[str, Any]) -> float | None:
    """0…1 distance between two fingerprints: mean of the two answer-distribution JSDs and the reasoning-token gap
    (|log2 ratio|, capped at 1)."""
    parts = [
        jsd(a.get("number") or {}, b.get("number") or {}),
        jsd(a.get("animal") or {}, b.get("animal") or {}),
    ]
    ra, rb = a.get("reasoning_median"), b.get("reasoning_median")
    if ra and rb:
        parts.append(min(1.0, abs(math.log2(ra / rb))))
    return sum(parts) / len(parts) if parts else None


def analyze(
    family: str,
    obs: list[dict[str, Any]],
    local: dict[str, list[int] | None],
    baseline: dict[str, Any] | None,
    sibling: dict[str, Any] | None = None,
    sibling_name: str | None = None,
    prefix_tokens: int | None = None,
) -> dict[str, Any]:
    """Checks for one model. `local` = calibration deltas per local tokenizer family (evaluator.tokenizers);
    `baseline` = this model's stored state from a past clean run; `sibling` = the cheaper sibling's fingerprint from
    this run. `prefix_tokens` = local token count of PREFIX under the claimed family (None when no local tokenizer)."""
    checks: list[dict[str, Any]] = []

    def add(check: str, status: str, detail: str, **extra: Any) -> None:
        checks.append({"check": check, "status": status, "detail": detail, **extra})

    ok_calls = [o for o in obs if o["status"] == 200]
    if not ok_calls:
        errs = Counter(str(o.get("error") or o["status"])[:80] for o in obs)
        add("reachable", "warn", f"no successful call: {dict(errs)}")
        return {"status": "warn", "checks": checks, "state": None}

    # 1. tokenizer
    rep = reported_deltas(obs)
    claimed = local.get(family)
    others = {f: d for f, d in local.items() if f != family and d}
    matches = sorted(f for f, d in others.items() if _match(rep, d))
    if rep is None or not any(rep):
        add(
            "tokenizer",
            "warn",
            "the channel did not report usable prompt_tokens (can't check the tokenizer)",
            reported=rep,
        )
    elif claimed:
        if _match(rep, claimed):
            add(
                "tokenizer",
                "ok",
                f"billed token counts match the {family} tokenizer exactly",
                reported=rep,
                expected=claimed,
            )
        elif matches:
            add(
                "tokenizer",
                "alert",
                f"billed token counts match {', '.join(matches)}, not {family}",
                reported=rep,
                expected=claimed,
            )
        else:
            sev = (
                "warn" if family in ("qwen", "mimo", "glm") else "alert"
            )  # newer Chinese generations change tokenizers
            add(
                "tokenizer",
                sev,
                f"billed token counts match no known tokenizer (expected {family})",
                reported=rep,
                expected=claimed,
            )
    else:  # claude / gemini: no public tokenizer
        before = (baseline or {}).get("tok")
        if matches and family in WESTERN and not (family == "gemini" and matches == ["openai"]):
            add(
                "tokenizer",
                "alert",
                f"claims {family} but billed token counts match {', '.join(matches)}",
                reported=rep,
            )
        elif before and not _match(rep, before):
            add(
                "tokenizer", "alert", f"tokenizer changed since the baseline ({before} → {rep})", reported=rep
            )
        else:
            add(
                "tokenizer",
                "ok",
                "matches the stored baseline" if before else "baseline recorded",
                reported=rep,
            )

    # 2. hidden prompt
    base = _by(obs, "tok_base")
    overhead = None
    if base and base[0]["prompt_tokens"] is not None:
        groups = sorted({o["group"] for o in ok_calls if o["group"]})
        if prefix_tokens is not None:
            overhead = base[0]["prompt_tokens"] - prefix_tokens
        elif baseline and baseline.get("short_pt") is not None:
            overhead = base[0]["prompt_tokens"] - baseline["short_pt"]
        if overhead is not None and overhead > 200:
            add("hidden_prompt", "warn",
                f"~{overhead} hidden prompt tokens billed per call (channel groups: {', '.join(groups) or '?'})",
                tokens=overhead)  # fmt: skip
        elif overhead is not None:
            add("hidden_prompt", "ok", f"prompt overhead {overhead} tokens", tokens=overhead)

    persona = " ".join(o["content"] for o in _by(obs, "persona"))
    if persona and PERSONA_AGENT.search(persona):
        add(
            "hidden_prompt",
            "warn",
            f"answers as a pre-configured agent persona (hidden system prompt): {persona[:90]}",
        )

    # 3. identity
    who = " ".join(o["content"] for o in _by(obs, "who")).lower()
    if who:
        own = VENDOR_WORDS.get(family, ())
        if any(w in who for w in own):
            add("identity", "ok", f"says: {who[:60]}")
        elif family in WESTERN and any(w in who for w in CHINESE_VENDORS):
            add("identity", "alert", f"a {family} model says it was trained by: {who[:60]}")
        else:
            add("identity", "warn", f"unexpected self-identification: {who[:60]}")

    # 4. censorship: alert only when every politically sensitive question is deflected
    answers = [" ".join(o["content"] for o in _by(obs, f"censor_{i}")) for i in range(len(CENSOR))]
    answers = [a for a in answers if a]
    if answers and family in WESTERN:
        deflected = [a for a in answers if not CENSOR_FACTS.search(a)]
        if not deflected:
            add("censorship", "ok", f"answers {len(answers)} China-sensitive questions factually")
        elif len(deflected) == len(answers):
            add(
                "censorship",
                "alert",
                f"deflects every China-sensitive question like a Chinese-hosted model: {deflected[0][:80]}",
            )
        else:
            add(
                "censorship",
                "warn",
                f"deflected {len(deflected)} of {len(answers)} China-sensitive questions: {deflected[0][:80]}",
            )
    elif answers:
        add("censorship", "info", f"{family} answer: {answers[0][:60]}")

    # 5. fingerprint
    fp = fingerprint(obs)
    if baseline and baseline.get("fp"):
        d_self = distance(fp, baseline["fp"])
        if sibling and sibling_name:
            d_sib = distance(fp, sibling)
            if d_self is not None and d_sib is not None and d_sib + 0.05 < d_self:
                add("fingerprint", "alert",
                    f"behaves more like {sibling_name} (distance {d_sib:.2f}) than like its own baseline ({d_self:.2f})",
                    distance_self=d_self, distance_sibling=d_sib)  # fmt: skip
            else:
                add("fingerprint", "ok", f"distance to baseline {d_self:.2f}, to {sibling_name} {d_sib:.2f}",
                    distance_self=d_self, distance_sibling=d_sib)  # fmt: skip
        elif d_self is not None:
            add(
                "fingerprint",
                "warn" if d_self > 0.6 else "ok",
                f"distance to baseline {d_self:.2f}",
                distance_self=d_self,
            )
    elif sibling and sibling_name:
        d_sib = distance(fp, sibling)
        add(
            "fingerprint",
            "info",
            f"baseline recorded; distance to {sibling_name} {d_sib:.2f}",
            distance_sibling=d_sib,
        )
    else:
        add("fingerprint", "info", "baseline recorded")

    # 6. metadata
    meta = {
        "id_prefix": sorted({o["id_prefix"] for o in ok_calls if o["id_prefix"]}),
        "usage_keys": sorted({k for o in ok_calls for k in o["usage_keys"]}),
        "system_fingerprint": any(o["system_fingerprint"] for o in ok_calls),
    }
    old = (baseline or {}).get("meta")
    if old and (
        set(meta["id_prefix"]) - set(old["id_prefix"]) or set(old["usage_keys"]) - set(meta["usage_keys"])
    ):
        add("metadata", "warn", f"response metadata changed: {old} → {meta}")
    else:
        add("metadata", "ok", f"ids {', '.join(meta['id_prefix']) or '?'}")

    status = max((c["status"] for c in checks), key=lambda s: SEVERITY[s], default="ok")
    state = {
        "tok": rep,
        "short_pt": base[0]["prompt_tokens"] if base else None,
        "fp": fp,
        "meta": meta,
        "groups": dict(Counter(o["group"] for o in ok_calls if o["group"])),
    }
    return {
        "status": "ok" if status == "info" else status,
        "checks": checks,
        "state": state,
        "fingerprint": fp,
    }
