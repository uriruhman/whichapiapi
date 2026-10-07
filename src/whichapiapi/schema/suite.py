"""Eval suite: a compatible subset of promptfoo's config (see ADR-0002) plus `x-whichapiapi` extensions."""

from __future__ import annotations

import contextlib
import json
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, Field

from whichapiapi.schema.offer import Price

Messages = list[dict[str, Any]]


class Assertion(BaseModel):
    model_config = ConfigDict(extra="allow")

    type: str
    value: Any = None
    threshold: float | None = None
    weight: float = 1.0
    metric: str | None = None
    provider: Any = None  # judge override for model-graded assertions


class TestCase(BaseModel):
    model_config = ConfigDict(extra="allow")

    description: str | None = None
    vars: dict[str, Any] = Field(default_factory=dict)
    assert_: list[Assertion] = Field(default_factory=list, alias="assert")
    threshold: float | None = None  # pass if weighted score >= threshold (instead of all-pass)


class Prompt(BaseModel):
    id: str
    label: str
    messages: Messages  # templates; content rendered per test with vars


class Channel(BaseModel):
    """A place to call models: base_url + key env var (+ optional balance probe for measured cost)."""

    base_url: str
    key_env: str
    protocol: str = "openai"  # "openai" (Chat Completions) | "tavily" (search.web) | "openai_stt" | "deepgram" (speech.stt)
    balance: str | None = None  # "newapi" | "openrouter" | None
    pricing: str | None = None  # adapter id to price offers from, e.g. "newapi"
    channel_type: str = "user_added"
    min_balance: float | None = None  # refuse runs that would leave less than this on a shared key
    timeout_s: float = 300.0


class ProviderSpec(BaseModel):
    """One candidate to evaluate."""

    id: str  # "myreseller:gpt-6-luna" | "openai:chat:gpt-4o" | "mock:echo"
    label: str
    channel: str | None = None
    model: str
    config: dict[str, Any] = Field(default_factory=dict)  # temperature, max_tokens, response_format...
    price: Price | None = None  # explicit price overrides store lookup
    current: bool = False  # the user's current choice (baseline for "N % cheaper")


class Judge(BaseModel):
    provider: ProviderSpec
    rubric_prefix: str | None = None


class Budget(BaseModel):
    run_usd: float = 0.25
    safety_factor: float = 1.5
    chars_per_token: float = 3.0
    default_output_tokens: int = 1500
    default_units_per_call: float = 0.0  # planning guess for per-unit APIs, e.g. audio minutes per STT call


class SuiteExt(BaseModel):
    channels: dict[str, Channel] = Field(default_factory=dict)
    judge: dict[str, Any] | None = None
    budget: Budget = Field(default_factory=Budget)
    contains_pii: bool = False
    capability: str = "llm.chat"
    task: str = "general"  # selector task for `eval refresh` (general, coding, math, agents, russian, …)
    weights: dict[str, float] = Field(default_factory=lambda: {"quality": 0.6, "cost": 0.3, "latency": 0.1})
    min_pass_rate: float = 0.5


class Suite(BaseModel):
    description: str = ""
    prompts: list[Prompt]
    providers: list[ProviderSpec]
    tests: list[TestCase]
    default_test: TestCase = Field(default_factory=TestCase)
    ext: SuiteExt = Field(default_factory=SuiteExt)
    base_dir: Path = Path(".")
    raw: dict[str, Any] = Field(default_factory=dict)

    def all_asserts(self, test: TestCase) -> list[Assertion]:
        return [*self.default_test.assert_, *test.assert_]

    def vars_for(self, test: TestCase) -> dict[str, Any]:
        return {**self.default_test.vars, **test.vars}


# ---------------------------------------------------------------- loading


def inside(base: Path, rel: str) -> Path:
    """`base/rel`, refusing anything that resolves outside `base`: a suite must not read (and send to an API) files
    such as ~/.ssh keys or env files."""
    path = (base / rel).resolve()
    if not path.is_relative_to(base.resolve()):
        raise ValueError(f"{rel}: file references must stay inside the suite directory {base}")
    return path


def _resolve_file(ref: Any, base: Path) -> Any:
    """promptfoo-style `file://path` → file contents (parsed for .json/.yaml, text otherwise)."""
    if not (isinstance(ref, str) and ref.startswith("file://")):
        return ref
    path = inside(base, ref.removeprefix("file://"))
    text = path.read_text(encoding="utf-8")
    if path.suffix == ".json":
        return json.loads(text)
    if path.suffix in (".yaml", ".yml"):
        return yaml.safe_load(text)
    return text


def _parse_prompt(i: int, entry: Any, base: Path) -> Prompt:
    label = None
    if isinstance(entry, dict):
        label = entry.get("label")
        entry = entry.get("raw") or entry.get("id")
    ref = entry
    content = _resolve_file(entry, base)
    if isinstance(content, str):
        stripped = content.strip()
        if stripped.startswith("["):
            with contextlib.suppress(json.JSONDecodeError):
                content = json.loads(stripped)
    if isinstance(content, str):
        messages: Messages = [{"role": "user", "content": content}]
    elif isinstance(content, list):
        messages = content
    else:
        raise ValueError(f"prompt #{i}: unsupported prompt format")
    if label is None:
        label = ref.removeprefix("file://") if isinstance(ref, str) and ref.startswith("file://") else f"p{i}"
    return Prompt(id=f"p{i}", label=label, messages=messages)


def _parse_provider(entry: Any, channels: dict[str, Channel]) -> ProviderSpec:
    if isinstance(entry, str):
        entry = {"id": entry}
    pid: str = entry["id"]
    config = dict(entry.get("config") or {})
    ext = entry.get("x-whichapiapi") or {}
    if pid.startswith("openai:"):
        # promptfoo native: openai:chat:<model> with config.apiBaseUrl / apiKeyEnvar
        model = pid.split(":", 2)[-1]
        channel = ext.get("channel")
        if channel is None and "apiBaseUrl" in config:
            channel = f"url:{config['apiBaseUrl']}"
            channels.setdefault(
                channel,
                Channel(base_url=config["apiBaseUrl"], key_env=config.get("apiKeyEnvar", "OPENAI_API_KEY")),
            )
        channel = channel or "openai"
        if channel == "openai":
            channels.setdefault(
                "openai",
                Channel(
                    base_url="https://api.openai.com/v1", key_env="OPENAI_API_KEY", channel_type="official"
                ),
            )
        config.pop("apiBaseUrl", None)
        config.pop("apiKeyEnvar", None)
    elif ":" in pid:
        channel, model = pid.split(":", 1)
        if channel != "mock" and channel not in channels:
            raise ValueError(f"provider {pid!r}: channel {channel!r} not defined in x-whichapiapi.channels")
    else:
        raise ValueError(f"provider {pid!r}: expected '<channel>:<model>'")
    price = Price(**ext["price"]) if "price" in ext else None
    return ProviderSpec(
        id=pid,
        label=entry.get("label") or pid,
        channel=channel,
        model=model,
        config=config,
        price=price,
        current=bool(ext.get("current", False)),
    )


def _parse_test(entry: dict[str, Any], base: Path) -> TestCase:
    entry = dict(entry)
    entry["vars"] = {k: _resolve_file(v, base) for k, v in (entry.get("vars") or {}).items()}
    return TestCase.model_validate(entry)


def load_suite(path: str | Path) -> Suite:
    path = Path(path)
    base = path.parent
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    ext = SuiteExt.model_validate(raw.get("x-whichapiapi") or {})
    channels = dict(ext.channels)

    tests_raw = _resolve_file(raw.get("tests") or [], base)
    tests = [_parse_test(t, base) for t in tests_raw]
    providers = [_parse_provider(p, channels) for p in raw.get("providers") or []]
    ext.channels = channels
    suite = Suite(
        description=raw.get("description", ""),
        prompts=[_parse_prompt(i, p, base) for i, p in enumerate(raw.get("prompts") or [])],
        providers=providers,
        tests=tests,
        default_test=_parse_test(raw.get("defaultTest") or {}, base),
        ext=ext,
        base_dir=base,
        raw=raw,
    )
    if not suite.prompts or not suite.providers or not suite.tests:
        raise ValueError("suite needs at least one prompt, provider and test")
    return suite


def judge_spec(suite: Suite) -> ProviderSpec | None:
    """Judge from x-whichapiapi.judge or defaultTest.options.provider (promptfoo)."""
    j = suite.ext.judge
    if j is None:
        opts = suite.raw.get("defaultTest", {}).get("options", {}) or {}
        j = {"provider": opts.get("provider")} if opts.get("provider") else None
    if not j or not j.get("provider"):
        return None
    channels = dict(suite.ext.channels)
    spec = _parse_provider(j["provider"], channels)
    suite.ext.channels.update(channels)
    return spec
