import json

from whichapiapi.adapters.inferindex import InferIndexAdapter
from whichapiapi.adapters.litellm_conditions import LiteLLMConditionsAdapter
from whichapiapi.adapters.lmarena import LMArenaAdapter
from whichapiapi.adapters.models_dev import ModelsDevAdapter
from whichapiapi.adapters.newapi import NewApiAdapter


def test_newapi_parse(fix):
    pricing = json.loads((fix / "newapi_pricing.json").read_text())
    status = json.loads((fix / "newapi_status.json").read_text())["data"]
    offers = list(NewApiAdapter("https://api.reseller.test/v1", "reseller").parse(pricing, status))
    base = next(o for o in offers if o.model == "gpt-6-luna" and o.group is None)
    # model_ratio 0.05 -> $0.10 / 1M in, completion_ratio 5 -> $0.50 out (matches official list)
    assert base.price.input_per_1m == 0.1 and base.price.output_per_1m == 0.5
    assert base.price.cache_read_per_1m == 0.01
    assert (
        base.id == "reseller/gpt-6-luna" and base.visibility == "user" and base.channel_type == "user_added"
    )
    grp = next(o for o in offers if o.model == "gpt-6-luna" and o.group == "Group-A1")
    assert grp.price.input_per_1m == round(0.1 * pricing["group_ratio"]["Group-A1"], 6)
    assert base.endpoint.base_url == "https://api.reseller.test/v1"
    assert "Crypto Pay" in base.billing.payment_methods and base.billing.min_topup == 1
    assert base.vendor == "openai"


def test_models_dev_parse():
    data = {
        "deepseek": {
            "id": "deepseek", "api": "https://api.deepseek.com", "npm": "@ai-sdk/openai-compatible",
            "env": ["DEEPSEEK_API_KEY"],
            "models": {
                "deepseek-chat": {"cost": {"input": 0.27, "output": 1.1, "cache_read": 0.07},
                                  "limit": {"context": 128000, "output": 8192}, "tool_call": True,
                                  "family": "deepseek"},
                "free": {"name": "no cost field"},
            },
        },
        "requesty": {"id": "requesty", "models": {"x/y": {"cost": {"input": 1, "output": 2}}}},
    }  # fmt: skip
    offers = list(ModelsDevAdapter().parse(data))
    assert len(offers) == 2
    ds = offers[0]
    assert ds.channel_type == "official" and ds.endpoint.protocol == "openai" and "tools" in ds.features
    assert ds.endpoint.key_env == "DEEPSEEK_API_KEY"
    assert offers[1].channel_type == "aggregator" and offers[1].vendor == "x"


def test_lmarena_parse():
    rows = [
        {"model_name": "claude-opus-5.5-high", "organization": "anthropic", "rating": 1517.79, "rank": 1.0,
         "vote_count": 2307.0, "category": "overall", "leaderboard_publish_date": "2026-09-25"},
        {"model_name": "deepseek/DeepSeek-V4.1-Flash", "organization": "deepseek", "rating": 1300.111,
         "rank": 50.0, "vote_count": 500.0, "category": "overall",
         "leaderboard_publish_date": "2026-09-25"},
        {"model_name": "glm-4.5", "organization": "zhipu", "rating": 1600.0, "rank": 3.0, "vote_count": 100.0,
         "category": "chinese", "leaderboard_publish_date": "2026-09-25"},  # other category: dropped
        {"model_name": "no-rating-model", "organization": "x", "rating": None, "rank": 9.0,
         "vote_count": 1.0, "category": "overall", "leaderboard_publish_date": "2026-09-25"},
    ]  # fmt: skip
    scores = LMArenaAdapter().parse(rows, "text")
    assert [(sc.board, sc.model_name) for sc in scores] == [
        ("text/overall", "claude-opus-5.5-high"),
        ("text/overall", "deepseek/DeepSeek-V4.1-Flash"),
        ("text/chinese", "glm-4.5"),
    ]
    ds = scores[1]
    assert ds.score == 1300.11 and ds.rank == 50 and ds.votes == 500 and ds.id == "lmarena/text/overall"


def test_litellm_conditions_parse():
    data = {
        "gpt-4o": {  # batch fields on the same entry, no off-peak
            "input_cost_per_token": 2.5e-06,
            "output_cost_per_token": 1e-05,
            "input_cost_per_token_batches": 1.25e-06,
            "output_cost_per_token_batches": 5e-06,
            "litellm_provider": "openai",
        },
        "deepseek-flash": {  # off-peak window, no batch fields
            "input_cost_per_token": 3e-07,
            "output_cost_per_token": 1.2e-06,
            "off_peak_pricing": {
                "input_cost_per_token": 1.5e-07,
                "output_cost_per_token": 6e-07,
                "windows": [{"hours_utc": ["00:00-01:00", "10:00-00:00"], "weekdays": [1, 2, 3, 4, 5]}],
            },
        },
        "no-cost-model": {"litellm_provider": "x"},  # no cost fields at all -> nothing derivable
        "sample_spec": {"comment": "not a real model"},
        "openai/gpt-4o-mini": {  # sparser entry for the same model_key, added first
            "input_cost_per_token": 1.5e-07,
            "output_cost_per_token": 6e-07,
            "input_cost_per_token_batches": 7.5e-08,
        },
        "azure/gpt-4o-mini": {  # richer entry (batch + off-peak) for the same model_key
            "input_cost_per_token": 1.5e-07,
            "output_cost_per_token": 6e-07,
            "input_cost_per_token_batches": 7.5e-08,
            "off_peak_pricing": {
                "input_cost_per_token": 7.5e-08,
                "windows": [{"hours_utc": "00:00-06:00"}],
            },
        },
    }  # fmt: skip
    out = LiteLLMConditionsAdapter().parse(data)
    assert set(out) == {"gpt-4o", "deepseek-flash", "gpt-4o-mini"}

    assert out["gpt-4o"]["batch_discount"] == 0.5
    assert "off_peak" not in out["gpt-4o"]

    ds = out["deepseek-flash"]
    assert "batch_discount" not in ds
    assert ds["off_peak"]["discount"] == 0.5 and ds["off_peak"]["tz"] == "UTC"
    assert "00:00-01:00" in ds["off_peak"]["window"] and "weekdays" in ds["off_peak"]["window"]

    # the richer (batch + off-peak) variant won the model_key collision
    mini = out["gpt-4o-mini"]
    assert mini["batch_discount"] == 0.5 and "off_peak" in mini


def test_inferindex_parse():
    data = {
        "query": "deepseek/deepseek-v3.2",
        "model": {"id": "deepseek/deepseek-v3.2", "name": "DeepSeek: DeepSeek V3.2"},
        "cheapest": {
            "provider": "Nous Portal", "via": "nous", "blended_per_1M": 0.234, "input_per_1M": 0.2088,
            "output_per_1M": 0.3096, "cache_read_per_1M": 0.0216, "currency": "USD",
            "context_length": 163840, "promo": False,
        },
        "fetched_at": "2026-09-28T00:00:00Z",
    }  # fmt: skip
    row = InferIndexAdapter().parse(data)
    assert row["resolved_model_id"] == "deepseek/deepseek-v3.2"
    assert row["input_per_1m"] == 0.2088 and row["blended_per_1m"] == 0.234
    assert row["promo"] is False and row["price_before_promo"] is None
    assert row["checked_at"] == "2026-09-28T00:00:00Z"


def test_inferindex_parse_promo():
    data = {
        "model": {"id": "x/y", "name": "X Y"},
        "cheapest": {
            "provider": "P", "via": "direct", "input_per_1M": 1.0, "promo": True,
            "price_before_promo": 2.0, "confidence": "high",
        },
    }  # fmt: skip
    row = InferIndexAdapter().parse(data)
    assert row["promo"] is True and row["price_before_promo"] == 2.0 and row["promo_confidence"] == "high"


def test_inferindex_parse_empty():
    assert InferIndexAdapter().parse({"query": "unknown"}) == {}


def test_store_history(store, fix):
    pricing = json.loads((fix / "newapi_pricing.json").read_text())
    ad = NewApiAdapter("https://x.example", "rs")
    offers = list(ad.parse(pricing, {}))
    assert store.upsert_offers(offers)[0] == len(offers)
    assert store.upsert_offers(offers) == (0, 0)  # unchanged -> no new versions
    pricing["data"][0]["model_ratio"] *= 2
    changed = list(ad.parse(pricing, {}))
    new, ch = store.upsert_offers(changed)
    assert new == 0 and ch > 0
    oid = changed[0].id
    hist = store.offer_history(oid)
    assert len(hist) == 2 and hist[1]["price"]["input_per_1m"] == 2 * hist[0]["price"]["input_per_1m"]
    assert store.find_offers(channel="rs", include_user=False) == []


def test_infer_capability():
    from whichapiapi.schema.capability import infer_capability

    assert infer_capability("whisper-large-v3", ["audio"], ["text"]) == "speech.stt"
    assert infer_capability("mimo-v2.5-tts", ["text"], ["audio"]) == "speech.tts"
    assert infer_capability("gpt-4o", ["text", "audio", "image"], ["text"]) == "llm.chat"
    assert infer_capability("qwen3-embedding-4b", ["text"], ["text"]) == "llm.embeddings"
    # name fallback when a catalog has no modalities (new-api resellers)
    assert infer_capability("gpt-4o-mini-transcribe") == "speech.stt"
    assert infer_capability("nvidia/parakeet-tdt-0.6b-v3") == "speech.stt"
    assert infer_capability("stealth/pixel-canary") == "llm.chat"
    assert infer_capability("gpt-4o-mini-tts") == "speech.tts"
    assert infer_capability("gpt-image-2") == infer_capability("mj_imagine") == "image.gen"
    assert infer_capability("gemini-3.1-flash-image-preview") == "image.gen"
    assert infer_capability("veo_3_1-fast") == infer_capability("doubao-seedance-2-5-260628") == "video.gen"
    assert infer_capability("flux-kontext", ["text", "image"], ["image"]) == "image.gen"
    assert infer_capability("gemini-3.5-flash", ["text", "image"], ["text", "image"]) == "llm.chat"


def test_huggingface_router_parse():
    from whichapiapi.adapters.hf_router import HFRouterAdapter

    data = {
        "object": "list",
        "data": [
            {
                "id": "Qwen/Qwen3.8-27B",
                "owned_by": "Qwen",
                "architecture": {
                    "input_modalities": ["text", "image"],
                    "output_modalities": ["text"],
                },
                "providers": [
                    {
                        "provider": "novita",
                        "status": "live",
                        "context_length": 1000000,
                        "pricing": {"input": 0.42, "output": 3},
                        "is_free": False,
                        "supports_tools": True,
                        "supports_structured_output": False,
                        "first_token_latency_ms": 120.0,
                        "throughput": 32.28,
                    },
                    {
                        "provider": "cerebras",
                        "status": "live",
                        "context_length": 65536,
                        "pricing": {"input": 0.99, "output": 1.49},
                        "is_free": False,
                        "supports_tools": True,
                        "supports_structured_output": True,
                        "first_token_latency_ms": 80.0,
                        "throughput": 317.86,
                    },
                    {
                        "provider": "featherless-ai",
                        "status": "offline",
                        "is_free": False,
                    },
                ],
            },
            {
                "id": "deepseek-ai/DeepSeek-V4.1-Flash",
                "owned_by": "deepseek-ai",
                "architecture": {
                    "input_modalities": ["text"],
                    "output_modalities": ["text"],
                },
                "providers": [
                    {
                        "provider": "deepinfra",
                        "status": "live",
                        "context_length": 1048576,
                        "pricing": {"input": 0.2, "output": 0.6},
                        "is_free": False,
                        "supports_tools": True,
                        "supports_structured_output": True,
                        "first_token_latency_ms": 95.0,
                        "throughput": 67.8,
                    },
                ],
            },
        ],
    }

    offers = list(HFRouterAdapter().parse(data))

    assert len(offers) == 5
    first = [offer for offer in offers if offer.model == "Qwen/Qwen3.8-27B"]
    assert len(first) == 3
    assert all(offer.group != "featherless-ai" for offer in first)
    cheapest = next(offer for offer in first if offer.group is None)
    assert cheapest.endpoint.model_id.endswith(":cheapest")
    assert cheapest.price.input_per_1m == 0.99
    assert cheapest.price.output_per_1m == 1.49

    grouped = next(offer for offer in first if offer.group == "novita")
    assert grouped.reliability.ttft_ms == 120.0
    assert grouped.reliability.throughput_tps == 32.28
    assert "vision" in grouped.features
    assert "tools" in grouped.features

    second = [offer for offer in offers if offer.model == "deepseek-ai/DeepSeek-V4.1-Flash"]
    assert len(second) == 2
    assert any(offer.group == "deepinfra" for offer in second)
    assert any(offer.group is None for offer in second)


def test_deepinfra_parse():
    from pytest import approx

    from whichapiapi.adapters.deepinfra import DeepInfraAdapter

    def record(model_name, model_type, pricing, max_tokens=None, **extra):
        return {
            "model_name": model_name,
            "type": model_type,
            "reported_type": model_type,
            "pricing": pricing,
            "max_tokens": max_tokens,
            "replaced_by": None,
            "deprecated": None,
            "private": 0,
            **extra,
        }

    data = [
        record(
            "meta-llama/chat",
            "text-generation",
            {
                "type": "tokens",
                "cents_per_input_token": 0.01,
                "cents_per_output_token": 0.02,
                "rate_per_input_token_cached": 0.2,
                "discount": 0.5,
            },
            8192,
        ),
        record(
            "nvidia/asr",
            "automatic-speech-recognition",
            {"type": "input_length", "cents_per_input_sec": 0.000333},
        ),
        record(
            "openai/whisper",
            "automatic-speech-recognition",
            {"type": "time", "cents_per_sec": 0.05},
        ),
        record(
            "inworld/tts",
            "text-to-speech",
            {"type": "input_character_length", "cents_per_input_chars": 0.0035},
        ),
        record(
            "thenlper/gte-base",
            "embeddings",
            {"type": "input_tokens", "cents_per_input_token": 0.01},
            512,
        ),
        record(
            "old/chat",
            "text-generation",
            {
                "type": "tokens",
                "cents_per_input_token": 0.01,
                "cents_per_output_token": 0.02,
            },
            deprecated=1,
        ),
        record(
            "soon/chat",
            "text-generation",
            {"type": "tokens", "cents_per_input_token": 0.01, "cents_per_output_token": 0.02},
            deprecated=4102444800,  # 2100-01-01: still served until then
            replaced_by="new/chat",
        ),
        record(
            "qwen/reranker",
            "reranker",
            {"type": "input_tokens", "cents_per_input_token": 0.01},
        ),
    ]

    offers = list(DeepInfraAdapter().parse(data))
    by_model = {offer.model: offer for offer in offers}

    assert "replaced by new/chat" in by_model.pop("soon/chat").provenance.note
    assert set(by_model) == {
        "meta-llama/chat",
        "nvidia/asr",
        "openai/whisper",
        "inworld/tts",
        "thenlper/gte-base",
    }

    chat = by_model["meta-llama/chat"]
    assert chat.capability == "llm.chat"
    assert chat.price.input_per_1m == approx(100.0)
    assert chat.price.output_per_1m == approx(200.0)
    assert chat.price.cache_read_per_1m == approx(20.0)  # rate_* is a multiplier on the input price
    assert chat.channel == "deepinfra"
    assert chat.channel_type == "official"
    assert chat.endpoint.protocol == "openai"

    stt = by_model["nvidia/asr"]
    assert stt.capability == "speech.stt"
    assert stt.price.per_unit == approx(0.0001998)
    assert stt.price.unit == "minute"

    timed = by_model["openai/whisper"]
    assert timed.price.per_unit == approx(0.0005)
    assert timed.price.unit == "exec_second"

    tts = by_model["inworld/tts"]
    assert tts.capability == "speech.tts"
    assert tts.price.per_unit == approx(0.035)
    assert tts.price.unit == "1k_chars"

    embeddings = by_model["thenlper/gte-base"]
    assert embeddings.capability == "llm.embeddings"
    assert embeddings.price.input_per_1m == approx(100.0)
    assert embeddings.context_window == 512
    assert embeddings.endpoint.protocol == "openai"


def test_edenai_parse():
    from whichapiapi.adapters.edenai import EdenAIAdapter

    def record(feature, subfeature, provider, model, price, quantity, unit, working=True):
        return {
            "pricings": [
                {
                    "model_name": model,
                    "price": price,
                    "price_unit_quantity": quantity,
                    "price_unit_type": unit,
                }
            ],
            "is_working": working,
            "provider": {"name": provider},
            "feature": {"name": feature},
            "subfeature": {"name": subfeature},
        }

    data = [
        record("audio", "speech_to_text_async", "gladia", "default", "0.0052", 60, "seconde"),
        record("audio", "speech_to_text_async", "other", "default", "0.0001734", 1, "seconde"),
        record("audio", "tts", "deepgram", "aura", "0.015", 1_000_000, "char"),
        record("ocr", "ocr", "mistral", "default", "4", 1000, "page"),
        record("ocr", "ocr", "broken", "default", "1", 1, "page", working=False),
        record("llm", "chat", "minimax", "model", "1", 1, "token"),
        record("audio", "tts", "skip", "default", "1", 1, "token"),
        record("ocr", "ocr_async", "mistral", "default", "5", 1, "page"),
        record("ocr", "ocr", "mistral", "default", "2", 1, "page"),
    ]

    offers = list(EdenAIAdapter().parse(data))
    assert len(offers) == 5

    stt = next(o for o in offers if o.vendor == "gladia")
    assert stt.price.per_unit == 0.0052
    assert stt.price.unit == "minute"

    other = next(o for o in offers if o.vendor == "other")
    assert other.price.per_unit == 0.010404
    assert other.price.unit == "minute"

    tts = next(o for o in offers if o.vendor == "deepgram")
    assert tts.price.per_unit == 0.000015
    assert tts.price.unit == "1k_chars"

    ocr = next(o for o in offers if o.model == "mistral/ocr")  # duplicate model: the cheaper price wins
    assert ocr.price.per_unit == 0.004
    assert ocr.price.unit == "page"
    assert ocr.endpoint.model_id == ocr.model
    assert ocr.endpoint.key_env == "EDENAI_API_KEY"
    assert next(o for o in offers if o.model == "mistral/ocr_async").price.per_unit == 5.0
    assert stt.model == "gladia/speech_to_text_async" and tts.model == "deepgram/aura"


def test_open_asr_parse():
    from whichapiapi.adapters.open_asr import OpenASRAdapter

    rows = [
        {"model": "nvidia/parakeet-tdt-0.6b-v3", "avg": 6.34, "RTFx": 3332.7, "LS Clean WER": 1.93, "AMI WER": None},
        {"model": "assemblyai/universal-3-5-pro", "avg": 4.34125, "RTFx": None, "LS Clean WER": 1.1},
        {"model": "broken/no-avg", "avg": float("nan"), "RTFx": 10.0},
    ]  # fmt: skip
    scores = OpenASRAdapter().parse(rows)
    by = {(s.model_name, s.board): s for s in scores}
    avg = by[("nvidia/parakeet-tdt-0.6b-v3", "avg_wer")]
    assert avg.score == 6.34 and not avg.higher_is_better and avg.rank == 2 and avg.unit == "wer"
    assert by[("assemblyai/universal-3-5-pro", "avg_wer")].rank == 1
    assert by[("nvidia/parakeet-tdt-0.6b-v3", "rtfx")].higher_is_better
    assert by[("nvidia/parakeet-tdt-0.6b-v3", "wer/ls_clean")].score == 1.93
    assert ("nvidia/parakeet-tdt-0.6b-v3", "wer/ami") not in by  # missing value
    assert ("assemblyai/universal-3-5-pro", "rtfx") not in by
    assert ("broken/no-avg", "avg_wer") not in by and by[("broken/no-avg", "rtfx")].score == 10.0


def test_swebench_adapter_parses_leaderboards():
    from whichapiapi.adapters.swebench import SWEBenchAdapter

    data = {
        "leaderboards": [
            {
                "name": "Verified",
                "results": [
                    {
                        "model_display": "Claude 4.5 Opus",
                        "model_org": "Anthropic",
                        "resolved": 79.2,
                        "date": "2025-12-05",
                    },
                    {
                        "model_display": "Multiple",
                        "model_org": "Anthropic",
                        "resolved": 44.25,
                        "date": "2025-10-27",
                    },
                ],
            },
            {
                "name": "Test",
                "results": [
                    {
                        "model_display": "Claude 4.5 Opus",
                        "model_org": "Anthropic",
                        "resolved": 52.62,
                        "date": "2025-12-19",
                    },
                    {
                        "model_display": "Warned Agent",
                        "model_org": "Example",
                        "resolved": 61.5,
                        "date": "2025-11-01",
                        "warning": "under review",
                    },
                ],
            },
        ]
    }

    scores = SWEBenchAdapter().parse(data)

    assert [(score.board, score.score) for score in scores] == [
        ("verified", 79.2),
        ("test", 52.62),
    ]
    assert [score.date for score in scores] == ["2025-12-05", "2025-12-19"]
    assert [score.model_name for score in scores] == ["Claude 4.5 Opus", "Claude 4.5 Opus"]


def test_bfcl_parse():
    from whichapiapi.adapters.bfcl import BFCLAdapter

    text = """Rank,Overall Acc,Model,Non-Live AST Acc,Live Acc,Multi Turn Acc,Organization
1,77.47%,Claude-Opus-4-5-20251101 (FC),88.58%,79.79%,68.38%,Anthropic
2,72.51%,Gemini-3-Pro-Preview (Prompt),90.65%,83.12%,60.75%,Google
3,70.00%,Example-Model (FC thinking),80.00%,,55.00%,Example
"""

    scores = BFCLAdapter().parse(text)

    assert {(score.model_name, score.board) for score in scores} == {
        ("Claude-Opus-4-5-20251101", "overall/fc"),
        ("Claude-Opus-4-5-20251101", "live/fc"),
        ("Claude-Opus-4-5-20251101", "multi_turn/fc"),
        ("Claude-Opus-4-5-20251101", "non_live/fc"),
        ("Gemini-3-Pro-Preview", "overall/prompt"),
        ("Gemini-3-Pro-Preview", "live/prompt"),
        ("Gemini-3-Pro-Preview", "multi_turn/prompt"),
        ("Gemini-3-Pro-Preview", "non_live/prompt"),
        ("Example-Model", "overall/fc"),
        ("Example-Model", "multi_turn/fc"),
        ("Example-Model", "non_live/fc"),
    }

    by = {(score.model_name, score.board): score for score in scores}
    opus = "Claude-Opus-4-5-20251101"
    assert by[(opus, "overall/fc")].score == 77.47 and by[(opus, "overall/fc")].rank == 1
    assert by[(opus, "live/fc")].score == 79.79
    assert by[("Gemini-3-Pro-Preview", "overall/prompt")].score == 72.51
    assert ("Example-Model", "live/fc") not in by  # empty cell


def test_artificial_analysis_parse():
    from whichapiapi.adapters.artificial_analysis import ArtificialAnalysisAdapter

    data = {
        "status": 200,
        "data": [
            {"slug": "claude-opus-5-5", "release_date": "2026-09-22", "model_creator": {"slug": "anthropic"},
             "evaluations": {"artificial_analysis_intelligence_index": 57.6, "artificial_analysis_coding_index": None,
                             "hle": 0.614, "lcr": 0.846666666666667},
             "median_output_tokens_per_second": 96.114, "median_time_to_first_token_seconds": 0},
        ],
    }  # fmt: skip
    by = {s.board: s for s in ArtificialAnalysisAdapter(api_key="k").parse(data)}
    assert set(by) == {"intelligence_index", "bench/hle", "bench/lcr", "speed_tps"}  # ttft 0 = not measured
    assert (
        by["intelligence_index"].score == 57.6
        and by["bench/hle"].score == 61.4
        and by["bench/lcr"].score == 84.67
    )
    assert by["speed_tps"].higher_is_better and by["bench/hle"].date == "2026-09-22"
    assert by["bench/hle"].organization == "anthropic"


def test_lmarena_agent_arena_score_columns():
    rows = [{"model_name": "Claude Fable 5.1 (Max)", "organization": "anthropic", "score": 0.140647, "score_ci_lower": 0.1229,
             "score_ci_upper": 0.1583, "session_count": 15125.0, "rank": 1, "category": "overall"}]  # fmt: skip
    (sc,) = LMArenaAdapter().parse(rows, "agent")
    assert sc.board == "agent/overall" and sc.score == 0.1406 and sc.unit == "score" and sc.votes == 15125
