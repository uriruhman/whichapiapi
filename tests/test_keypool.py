import json

import httpx

from whichapiapi import keypool
from whichapiapi.surfaces import router


def test_encrypted_store_rotation_and_disable(tmp_path):
    a = keypool.add("groq", "gsk_aaaaaaaaaaaa")
    keypool.add("groq", "gsk_bbbbbbbbbbbb")
    assert keypool.add("groq", "gsk_aaaaaaaaaaaa")["id"] == a["id"]  # deduplicated
    raw = (tmp_path / "home" / "provider-keys.json").read_text()
    assert (
        "gsk_aaaaaaaaaaaa" not in raw
        and oct((tmp_path / "home" / "provider-keys.json").stat().st_mode & 0o777) == "0o600"
    )
    got = {keypool.pick("groq")[1], keypool.pick("groq")[1]}
    assert got == {"gsk_aaaaaaaaaaaa", "gsk_bbbbbbbbbbbb"}
    for _ in range(keypool.AUTH_FAILS_TO_DISABLE):
        keypool.report(a["id"], 401, "invalid api key")
    assert [k["enabled"] for k in keypool.listing()] == [False, True] and keypool.providers() == {"groq"}


def test_parse_env_and_json():
    env = "export GROQ_API_KEY='gsk_123456789'\nGEMINI_API_KEY=AIza12345678 # comment\nDEBUG=1\nHF_TOKEN=hf_abcdefgh\n"  # gitleaks:allow
    assert keypool.parse_file(env) == [
        ("groq", "gsk_123456789"), ("google", "AIza12345678"), ("huggingface", "hf_abcdefgh")]  # fmt: skip
    assert keypool.parse_file(json.dumps({"CEREBRAS_API_KEY": "csk-1234567890", "mistral": "abcdefghij"})) == [  # gitleaks:allow (fake test value)
        ("cerebras", "csk-1234567890"), ("mistral", "abcdefghij")]  # fmt: skip


async def test_router_uses_pool_keys(monkeypatch):
    keypool.add("groq", "gsk_poolkey12345")
    monkeypatch.setattr(router, "candidates", lambda task, preset: ["groq:llama-4-70b"])
    seen = {}

    def handler(request):
        seen["url"], seen["auth"] = str(request.url), request.headers["authorization"]
        return httpx.Response(200, json={"choices": [{"message": {"content": "ok"}}]})

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    status, headers, _ = await router.complete({"model": "auto:general@free", "messages": []}, client)
    assert status == 200 and seen["url"] == "https://api.groq.com/openai/v1/chat/completions"
    assert seen["auth"] == "Bearer gsk_poolkey12345" and headers["x-whichapiapi-route"] == "groq:llama-4-70b"
