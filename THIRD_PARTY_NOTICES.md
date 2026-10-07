# Third-party notices

## FreeLLMAPI — https://github.com/tashfeenahmed/freellmapi (MIT)

What Which API API uses from it:

- **Free-tier catalog data**: `adapters/freellmapi.py` reads FreeLLMAPI's public free catalog
  (`https://api.freellmapi.co/v1/latest`, the 30-day-delayed "monthly" snapshot) — free-tier limits, reset times and
  provider quirks — and verifies its Ed25519 signature with the public key published in their source
  (`server/src/services/catalog-sync.ts`). Offers built from it carry the source in their provenance
  ("… free tier via FreeLLMAPI catalog"). Credit: FreeLLMAPI / freellmapi.co.
- **Ideas and parameters, re-implemented in Python** (no code copied): the router's error classes and cooldown ladder,
  decaying penalties and first-token deadline (`route_health.py`, `surfaces/router.py`), tool-call rescue
  (`surfaces/rescue.py`), response cache / idempotency / stickiness (`surfaces/router_memory.py`), the encrypted key
  pool (`keypool.py`) and the coding-agent `setup` / `launch` / `doctor` commands (`agents.py`).

Their licence:

```text
MIT License

Copyright (c) 2026 Tashfeen Ahmed

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

## OpenAI Whisper text normalizers (MIT)

Vendored byte-identical in `src/whichapiapi/evaluator/whisper_normalizers/` with their own `LICENSE` file.
