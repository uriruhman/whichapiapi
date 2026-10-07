import json

from whichapiapi.implementer.audit import audit_n8n, to_markdown
from whichapiapi.implementer.n8n import _cron_per_month, extract, load, runs_per_month


def _wf(fix):
    return json.loads((fix / "n8n_workflow.json").read_text())


def test_extract_finds_ai_calls_models_and_fallbacks(fix):
    ex = extract(_wf(fix))
    by = {u.node: u for u in ex.uses}
    assert set(by) == {
        "Main model",
        "Backup model",
        "Lonely model",
        "Runtime model",
        "Embed",
        "Call OpenAI",
        "Transcribe",
        "Local",
    }
    assert (
        by["Main model"].fallback_model
        and by["Backup model"].fallback_model
        and not by["Lonely model"].has_fallback
    )
    assert by["Lonely model"].model == "openai/gpt-4.1-mini"  # the node's default when unset
    assert (
        by["Runtime model"].model == "deepseek/deepseek-v4.1-flash"
        and by["Runtime model"].model_is_expression
    )
    assert by["Embed"].capability == "llm.embeddings" and by["Embed"].model == "text-embedding-3-small"
    assert (
        by["Call OpenAI"].model == "gpt-4o-mini"
        and by["Call OpenAI"].retry
        and by["Call OpenAI"].provider == "openai"
    )
    assert by["Transcribe"].capability == "speech.stt" and by["Transcribe"].model == "openai/whisper-large-v3"
    assert by["Local"].local and by["Local"].provider == "ollama"
    assert all(u.runs_per_month == 120.0 for u in ex.uses)
    assert [s.node for s in ex.secrets] == ["Call OpenAI"] and "1234567890" not in ex.secrets[0].preview


def test_load_shapes_and_schedules(fix):
    wf = _wf(fix)
    assert len(load([wf, wf])) == 2 and len(load({"data": [wf]})) == 1 and len(load(json.dumps(wf))) == 1
    assert _cron_per_month("0 8 * * 1") == 4.35 and _cron_per_month("0 9 1 * *") == 1
    assert _cron_per_month("*/30 * * * *") == 2 * 24 * 30
    daily = {
        "nodes": [
            {
                "type": "n8n-nodes-base.scheduleTrigger",
                "parameters": {"rule": {"interval": [{"triggerAtHour": 7}]}},
            }
        ]
    }
    assert runs_per_month(daily) == 30 and runs_per_month({"nodes": []}) is None


def test_audit_reports_overpay_fallback_and_secret(fix, store):
    from whichapiapi.schema.offer import Offer, Price, Provenance

    def offer(channel, model, i, o):
        return Offer(
            model=model,
            channel=channel,
            price=Price(input_per_1m=i, output_per_1m=o),
            provenance=Provenance(source="t"),
        )

    store.upsert_offers(
        [offer("openai", "gpt-4o-mini", 0.15, 0.6), offer("cheapco", "gpt-4o-mini", 0.05, 0.2)]
    )
    rep = audit_n8n(_wf(fix), calls_per_month=10000, store=store)
    kinds = {(f["node"], f["kind"]) for f in rep["findings"]}
    assert ("Call OpenAI", "hardcoded_secret") in kinds and rep["findings"][0]["kind"] == "hardcoded_secret"
    assert ("Call OpenAI", "overpay") in kinds and rep["potential_saving_monthly"] > 0
    assert ("Lonely model", "no_fallback") in kinds and ("Main model", "no_fallback") not in kinds
    assert ("Call OpenAI", "no_fallback") not in kinds  # retry on fail counts
    assert ("Local", "self_hosted") in kinds and ("Runtime model", "runtime_model") in kinds
    assert "cheapco/gpt-4o-mini" in to_markdown(rep)


def test_free_route_flags_training_data(store):
    from whichapiapi.schema.offer import Offer, Price, Provenance

    store.upsert_offers(
        [
            Offer(
                model="some/model:free",
                channel="openrouter",
                price=Price(input_per_1m=0.0, output_per_1m=0.0),
                provenance=Provenance(source="t"),
            )
        ]
    )
    wf = {
        "name": "w",
        "nodes": [
            {
                "name": "LM",
                "type": "@n8n/n8n-nodes-langchain.lmChatOpenRouter",
                "parameters": {"model": "some/model:free"},
            }
        ],
    }
    kinds = {f["kind"] for f in audit_n8n(wf, store=store)["findings"]}
    assert "training_data" in kinds


def test_build_suite_from_node(fix, tmp_path):
    import yaml

    from whichapiapi.implementer.suite_gen import build_suite, to_template
    from whichapiapi.schema.suite import load_suite

    names: list[str] = []
    assert (
        to_template("=Hi {{ $json.name }}, {{ $json['order id'] }} {{ 1 + 2 }}", names)
        == "Hi {{name}}, {{order_id}} {{var_3}}"
    )
    files = build_suite(_wf(fix), "Main model", alternatives=["openrouter:qwen/qwen3.8-flash"])
    prompt = json.loads(files["prompt.json"])
    assert prompt[0]["role"] == "system" and "triage" in prompt[0]["content"]
    assert prompt[-1]["content"] == "Classify this ticket: {{ticket_text}} from {{customer}}"
    suite = yaml.safe_load(files["suite.yaml"])
    assert [p["id"] for p in suite["providers"]] == [
        "openrouter:deepseek/deepseek-v4.1-flash",
        "openrouter:qwen/qwen3.8-flash",
    ]
    assert suite["defaultTest"]["assert"][0]["type"] == "is-json"  # the prompt asks for JSON
    for name, text in files.items():
        (tmp_path / name).write_text(text)
    loaded = load_suite(tmp_path / "suite.yaml")  # the skeleton is a valid suite
    assert set(loaded.tests[0].vars) == {"ticket_text", "customer"}


def test_make_blueprint_and_code_scan(tmp_path, store):
    from whichapiapi.implementer.audit import audit_extraction
    from whichapiapi.implementer.sources import extract_code, extract_make

    bp = {
        "name": "Leads",
        "flow": [
            {"id": 1, "module": "openai-gpt-3:CreateCompletion", "mapper": {"model": "gpt-4o-mini"}},
            {"id": 2, "module": "builtin:BasicRouter", "routes": [{"flow": [
                {"id": 3, "module": "http:ActionSendData", "mapper": {"url": "https://openrouter.ai/api/v1/chat/completions",
                 "data": '{"model": "deepseek/deepseek-v4.1-flash"}', "headers": [{"value": "Bearer sk-or-v1-abcdef1234567890abcdef"}]}},
                {"id": 4, "module": "slack:CreateMessage", "mapper": {}},
            ]}]},
        ],
    }  # fmt: skip
    ex = extract_make(bp)
    assert [(u.provider, u.model) for u in ex.uses] == [
        ("openai", "gpt-4o-mini"),
        ("openrouter", "deepseek/deepseek-v4.1-flash"),
    ]
    assert len(ex.secrets) == 1
    (tmp_path / "bot.py").write_text(
        "from openai import OpenAI\nclient = OpenAI(base_url='https://api.deepinfra.com/v1/openai')\n"
        "r = client.chat.completions.create(model='meta-llama/Llama-3.3-70B-Instruct', messages=[])\n"
    )
    (tmp_path / "util.py").write_text("print('no ai here')\n")
    code = extract_code(tmp_path)
    assert [(u.workflow, u.provider, u.model, u.node) for u in code.uses] == [
        ("bot.py", "deepinfra", "meta-llama/Llama-3.3-70B-Instruct", "line 3")
    ]
    rep = audit_extraction(code, store=store)
    assert rep["api_calls"] == 1 and any(f["kind"] == "no_fallback" for f in rep["findings"])


def test_cases_from_executions(fix):
    import yaml

    from whichapiapi.implementer.suite_gen import build_suite

    run = lambda items, parent=None: {  # noqa: E731 — the shape a real n8n execution has (verified live)
        "source": [{"previousNode": parent}] if parent else [],
        "data": {"main": [[{"json": j} for j in items]]},
    }
    execution = {
        "id": "42",
        "data": {"resultData": {"runData": {
            "Load": [run([{"customer": "ACME"}])],
            "Prep": [run([{"ticket_text": "printer on fire"}, {"ticket_text": "refund please"}], "Load")],
            "Chain": [run([{"text": "ok"}], "Prep")],
        }}},
    }  # fmt: skip
    files = build_suite(_wf(fix), "Main model", executions=[execution])
    tests = yaml.safe_load(files["tests.yaml"])
    assert [t["vars"] for t in tests] == [
        {"ticket_text": "printer on fire", "customer": "ACME"},
        {"ticket_text": "refund please", "customer": "ACME"},
    ]


def test_draft_suite_writes_a_runnable_suite(tmp_path):
    from whichapiapi.implementer.suite_draft import draft_suite
    from whichapiapi.schema.suite import load_suite

    res = draft_suite(
        tmp_path / "triage",
        "Support ticket triage",
        "Classify this ticket into billing, bug or other. Ticket: {{ticket}}",
        [
            {"ticket": "I was charged twice", "expected": "billing"},
            {"vars": {"ticket": "App crashes on login"}},
        ],
        ["reseller:gpt-6-luna", "reseller:deepseek-v4.1-flash"],
        criteria="Picks the right category.",
        json_output=True,
    )
    suite = load_suite(res["suite"])
    assert [p.model for p in suite.providers] == ["gpt-6-luna", "deepseek-v4.1-flash"] and suite.providers[
        0
    ].current
    assert (
        suite.tests[0].vars == {"ticket": "I was charged twice"}
        and suite.tests[0].assert_[0].type == "icontains"
    )
    assert {a.type for a in suite.default_test.assert_} == {"is-json", "llm-rubric"} and res[
        "unknown_channels"
    ] == []
    import pytest

    with pytest.raises(ValueError, match="no value"):
        draft_suite(tmp_path / "x", "t", "{{a}} {{b}}", [{"a": 1}], ["reseller:m"])
