"""Build a small cached speech-to-text evaluation set from public audio clips.

Run from the repository root:

    uv run python examples/speech_stt/build_cases.py --en 5 --ru 3

Audio files are downloaded into `.cache/` (git-ignored), while the generated
case definitions are written to `.cache/tests.live.yaml`.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import Any

import httpx
import yaml

HERE = Path(__file__).parent
ROWS_URL = "https://datasets-server.huggingface.co/rows"

SOURCES: dict[str, dict[str, str]] = {
    "en": {
        "dataset": "hf-internal-testing/librispeech_asr_dummy",
        "config": "clean",
        "split": "validation",
        "text": "text",
        "id": "id",
        "license": "CC BY 4.0 (LibriSpeech, openslr.org/12)",
        "language": "en",
    },
    "ru": {
        "dataset": "google/fleurs",
        "config": "ru_ru",
        "split": "validation",
        "text": "transcription",
        "id": "id",
        "license": "CC BY 4.0 (FLEURS, Google)",
        "language": "ru",
    },
}

EXTENSIONS = {
    "audio/flac": ".flac",
    "audio/wav": ".wav",
    "audio/mpeg": ".mp3",
}


def fail(message: str) -> None:
    print(f"error: {message}", file=sys.stderr)
    raise SystemExit(1)


def safe_id(value: Any, row_idx: Any) -> str:
    result = re.sub(r"[^A-Za-z0-9_-]", "_", str(value))
    return result or f"row-{row_idx}"


def fetch_rows(
    client: httpx.Client, source: dict[str, str], offset: int, length: int
) -> list[dict[str, Any]]:
    try:
        response = client.get(
            ROWS_URL,
            params={
                "dataset": source["dataset"],
                "config": source["config"],
                "split": source["split"],
                "offset": offset,
                "length": length,
            },
        )
        response.raise_for_status()
        payload = response.json()
    except (httpx.HTTPError, ValueError) as exc:
        fail(f"request for {source['dataset']} failed: {exc}")

    rows = payload.get("rows")
    if not isinstance(rows, list):
        fail(f"request for {source['dataset']} returned no valid rows list")
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--en", type=int, default=5)
    parser.add_argument("--ru", type=int, default=3)
    parser.add_argument("--offset", type=int, default=0)
    args = parser.parse_args()

    if args.en < 0 or args.ru < 0 or args.offset < 0:
        fail("--en, --ru, and --offset must be non-negative")

    cache = HERE / ".cache"
    audio_dir = cache / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)

    tests: list[dict[str, Any]] = []
    counts = {"en": args.en, "ru": args.ru}

    with httpx.Client(timeout=60, follow_redirects=True) as client:
        for lang, count in counts.items():
            if count == 0:
                continue

            source = SOURCES[lang]
            rows = fetch_rows(client, source, args.offset, count)
            if len(rows) < count:
                fail(f"{source['dataset']} returned {len(rows)} rows, but {count} were requested")

            for item in rows:
                row = item.get("row")
                row_idx = item.get("row_idx")
                if not isinstance(row, dict):
                    fail(f"{source['dataset']} row {row_idx!r} has no row object")

                row_id = row.get(source["id"])
                text = row.get(source["text"])
                audio = row.get("audio")
                if row_id is None:
                    fail(f"{source['dataset']} row {row_idx!r} lacks {source['id']!r}")
                if not isinstance(text, str) or not text.strip():
                    fail(f"{source['dataset']} row {row_idx!r} lacks non-empty transcript")
                if not isinstance(audio, list) or not audio or not isinstance(audio[0], dict):
                    fail(f"{source['dataset']} row {row_idx!r} lacks audio")

                audio_info = audio[0]
                url = audio_info.get("src")
                content_type = audio_info.get("type")
                if not isinstance(url, str) or not url:
                    fail(f"{source['dataset']} row {row_idx!r} audio lacks src")
                if not isinstance(content_type, str):
                    fail(f"{source['dataset']} row {row_idx!r} audio lacks type")

                filename = f"{lang}-{safe_id(row_id, row_idx)}"
                filename += EXTENSIONS.get(content_type, ".bin")
                audio_path = audio_dir / filename

                if audio_path.exists() and audio_path.stat().st_size > 0:
                    action = "cached"
                else:
                    try:
                        download = client.get(url)
                        download.raise_for_status()
                    except httpx.HTTPError as exc:
                        fail(f"audio download for {lang} {row_id} failed: {exc}")
                    audio_path.write_bytes(download.content)
                    action = "downloaded"

                relative_audio = f".cache/audio/{filename}"
                tests.append(
                    {
                        "description": f"{lang} {row_id}",
                        "vars": {
                            "audio": relative_audio,
                            "reference": text,
                            "language": source["language"],
                        },
                    }
                )
                print(f"{lang} {row_id}: {action} {relative_audio}")

    header = (
        "# Generated by build_cases.py; this file is generated and lives in .cache/.\n"
        "# Sources and licenses:\n"
        "# - hf-internal-testing/librispeech_asr_dummy: "
        "CC BY 4.0 (LibriSpeech, openslr.org/12)\n"
        "# - google/fleurs: CC BY 4.0 (FLEURS, Google)\n"
    )
    output = cache / "tests.live.yaml"
    output.write_text(
        header + yaml.safe_dump(tests, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )
    print(f"wrote {len(tests)} clips and {output}")


if __name__ == "__main__":
    main()
