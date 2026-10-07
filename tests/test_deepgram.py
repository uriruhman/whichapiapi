import json

import httpx

from whichapiapi.transports.deepgram import DeepgramTransport

RESPONSE = {
    "metadata": {"duration": 30.0, "request_id": "r1", "model_info": {"u": {"name": "general-nova-3"}}},
    "results": {"channels": [{"detected_language": "ru", "alternatives": [{"transcript": "привет мир"}]}]},
}


def _transport(handler, base_dir):
    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    return DeepgramTransport(
        "https://api.deepgram.com/v1", "dg-secret-key-123", base_dir=base_dir, client=client
    )


async def test_call_parses_transcript_and_duration(tmp_path):
    (tmp_path / "a.flac").write_bytes(b"fLaC")
    seen = {}

    def handler(request):
        seen.update(
            path=request.url.path,
            params=dict(request.url.params),
            headers=request.headers,
            body=request.content,
        )
        return httpx.Response(200, json=RESPONSE)

    res = await _transport(handler, tmp_path).call("nova-3", [{"role": "user", "content": "a.flac"}], {})
    assert res.output == "привет мир" and res.usage.units == 0.5 and res.model_reported == "general-nova-3"
    assert seen["path"] == "/v1/listen" and seen["params"]["model"] == "nova-3"
    assert seen["params"]["detect_language"] == "true" and seen["params"]["smart_format"] == "true"
    assert seen["headers"]["authorization"] == "Token dg-secret-key-123"
    assert seen["headers"]["content-type"] == "audio/flac" and seen["body"] == b"fLaC"


async def test_explicit_language_disables_detection(tmp_path):
    (tmp_path / "a.wav").write_bytes(b"RIFF")
    params = {}

    def handler(request):
        params.update(request.url.params)
        return httpx.Response(200, json=RESPONSE)

    await _transport(handler, tmp_path).call(
        "nova-3", [{"role": "user", "content": "a.wav"}], {"language": "ru"}
    )
    assert params["language"] == "ru" and "detect_language" not in params


async def test_errors_are_redacted_and_paths_confined(tmp_path):
    (tmp_path / "a.wav").write_bytes(b"RIFF")
    t = _transport(
        lambda r: httpx.Response(401, text=json.dumps({"err": "bad key dg-secret-key-123"})), tmp_path
    )
    res = await t.call("nova-3", [{"role": "user", "content": "a.wav"}], {})
    assert res.error.startswith("APIStatusError: HTTP 401") and "dg-secret-key-123" not in res.error
    res = await t.call("nova-3", [{"role": "user", "content": "../x.wav"}], {})
    assert res.error.startswith("audio path outside")
    t = _transport(lambda r: httpx.Response(429, text="slow down"), tmp_path)
    assert (await t.call("nova-3", [{"role": "user", "content": "a.wav"}], {})).error.startswith(
        "RateLimitError"
    )
