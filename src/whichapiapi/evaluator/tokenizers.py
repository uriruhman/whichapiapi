"""Local token counters per model family, used to check which tokenizer a channel's billed `prompt_tokens` came from.

- openai: `tiktoken` o200k_base (GPT-4o … GPT-6 family). The BPE file is downloaded once by tiktoken and cached.
- deepseek, qwen, glm, mimo: the model's own `tokenizer.json` from its public Hugging Face repo, read with the
  `tokenizers` package (no torch), cached under `$WHICHAPIAPI_HOME/tokenizers/`.
- claude, gemini: no public tokenizer — checked against a stored baseline and by exclusion (see evaluator/integrity).
"""

from __future__ import annotations

import os
from collections.abc import Callable
from functools import cache
from pathlib import Path

import httpx

HF_REPOS = {
    "deepseek": "deepseek-ai/DeepSeek-V3.1",
    "qwen": "Qwen/Qwen3-8B",
    "glm": "zai-org/GLM-4.6",
    "mimo": "XiaomiMiMo/MiMo-7B-RL",
}
LOCAL_FAMILIES = ("openai", *HF_REPOS)
NO_LOCAL = ("claude", "gemini")


def _home() -> Path:
    return Path(os.environ.get("WHICHAPIAPI_HOME", Path.home() / ".local/share/whichapiapi"))


@cache
def counter(family: str) -> Callable[[str], int] | None:
    """A function text → token count for `family`, or None when it has no public tokenizer or can't be loaded."""
    if family == "openai":
        try:
            import tiktoken

            enc = tiktoken.get_encoding("o200k_base")
        except Exception:
            return None
        return lambda text: len(enc.encode(text, disallowed_special=()))
    repo = HF_REPOS.get(family)
    if not repo:
        return None
    path = _home() / "tokenizers" / f"{family}.json"
    if not path.exists():
        try:
            headers = (
                {"Authorization": f"Bearer {os.environ['HF_TOKEN']}"} if os.environ.get("HF_TOKEN") else {}
            )
            r = httpx.get(
                f"https://huggingface.co/{repo}/resolve/main/tokenizer.json",
                headers=headers,
                follow_redirects=True,
                timeout=60,
            )
            r.raise_for_status()
        except httpx.HTTPError:
            return None
        path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        path.write_bytes(r.content)
    try:
        from tokenizers import Tokenizer

        tok = Tokenizer.from_file(str(path))
    except Exception:
        return None
    return lambda text: len(tok.encode(text, add_special_tokens=False).ids)


def deltas(prefix: str, texts: list[str], family: str) -> list[int] | None:
    """Extra tokens each text adds after `prefix` under `family`'s tokenizer — the same quantity a channel's billed
    prompt_tokens difference measures, independent of any hidden system prompt the channel adds."""
    count = counter(family)
    if count is None:
        return None
    base = count(prefix)
    return [count(prefix + "\n" + t) - base for t in texts]


_FAMILY = [
    (("gpt-", "chatgpt", "o1", "o3", "o4", "codex"), "openai"),
    (("claude",), "claude"),
    (("gemini", "gemma"), "gemini"),
    (("deepseek",), "deepseek"),
    (("qwen", "qwq"), "qwen"),
    (("glm",), "glm"),
    (("mimo",), "mimo"),
    (("kimi", "moonshot"), "kimi"),
]


def family_of(model: str) -> str | None:
    """Tokenizer/vendor family from a model id ("myreseller:gpt-6-sol" → "openai")."""
    name = model.rpartition(":")[2].rpartition("/")[2].lower()
    for prefixes, family in _FAMILY:
        if name.startswith(prefixes):
            return family
    return None


def chat_prompt_tokens(messages: list[dict], family: str) -> int | None:
    """Local estimate of a plain-text chat request's prompt tokens (OpenAI framing: 3 + 4 per message). None when
    the family has no local tokenizer or the messages aren't plain text."""
    count = counter(family) if family == "openai" else None
    if count is None or not messages:
        return None
    total = 3
    for m in messages:
        if not isinstance(m.get("content"), str) or m.get("tool_calls") or m.get("role") == "tool":
            return None
        total += 4 + count(m["content"])
    return total
