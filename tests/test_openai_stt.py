import wave

import httpx
import pytest

from whichapiapi.transports.openai_stt import OpenAISTTTransport


def _wav(path, seconds):
    with wave.open(str(path), "wb") as f:
        f.setnchannels(1)
        f.setsampwidth(2)
        f.setframerate(16000)
        f.writeframes(b"\x00\x00" * int(16000 * seconds))
    return path


def _transport(handler, base_dir, max_retries=1):
    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    return OpenAISTTTransport(
        "https://stt.example/v1",
        "k",
        base_dir=base_dir,
        max_retries=max_retries,
        http_client=client,
    )


async def test_tokens_usage(tmp_path):
    _wav(tmp_path / "clip.wav", 1)

    def handler(request):
        body = request.read()
        assert request.url.path == "/v1/audio/transcriptions"
        assert request.headers["authorization"] == "Bearer k"
        assert b'name="model"' in body
        assert b"gpt-4o-mini-transcribe" in body
        assert b'name="language"' in body
        return httpx.Response(
            200,
            json={
                "text": "hello world",
                "usage": {
                    "type": "tokens",
                    "input_tokens": 12,
                    "output_tokens": 3,
                    "total_tokens": 15,
                },
            },
        )

    result = await _transport(handler, tmp_path).call(
        "gpt-4o-mini-transcribe",
        [{"role": "user", "content": "clip.wav"}],
        {"language": "en"},
    )
    assert result.output == "hello world"
    assert result.usage.input_tokens == 12
    assert result.usage.output_tokens == 3
    assert result.usage.units == pytest.approx(1 / 60, rel=0.05)


async def test_duration_usage(tmp_path):
    _wav(tmp_path / "clip.wav", 1)

    def handler(request):
        return httpx.Response(200, json={"text": "hi", "usage": {"type": "duration", "seconds": 30}})

    result = await _transport(handler, tmp_path).call(
        "transcribe", [{"role": "user", "content": "clip.wav"}], {}
    )
    assert result.usage.units == 0.5


async def test_path_traversal_does_not_call_handler(tmp_path):
    suite = tmp_path / "suite"
    suite.mkdir()
    called = False

    def handler(request):
        nonlocal called
        called = True
        return httpx.Response(200)

    result = await _transport(handler, suite).call(
        "transcribe", [{"role": "user", "content": "../outside.wav"}], {}
    )
    assert result.error.startswith("audio path outside")
    assert not called


async def test_missing_file(tmp_path):
    result = await _transport(lambda request: httpx.Response(200), tmp_path).call(
        "transcribe", [{"role": "user", "content": "missing.wav"}], {}
    )
    assert result.error.startswith("audio file not found")


async def test_http_500(tmp_path):
    _wav(tmp_path / "clip.wav", 1)
    result = await _transport(
        lambda request: httpx.Response(500, text="failed"), tmp_path, max_retries=0
    ).call("transcribe", [{"role": "user", "content": "clip.wav"}], {})
    assert result.error.startswith("InternalServerError")


async def test_empty_text(tmp_path):
    _wav(tmp_path / "clip.wav", 1)
    result = await _transport(lambda request: httpx.Response(200, json={"text": ""}), tmp_path).call(
        "transcribe", [{"role": "user", "content": "clip.wav"}], {}
    )
    assert result.error == "empty transcript"
