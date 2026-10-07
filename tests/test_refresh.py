import yaml

from whichapiapi import core
from whichapiapi.implementer import refresh
from whichapiapi.schema.suite import load_suite

SUITE = """# generated header
description: t
prompts: ["hi {{x}}"]
providers:
  - id: reseller:gpt-6-luna
    label: luna
    config: {temperature: 0}
    x-whichapiapi: {current: true}
tests: [{vars: {x: a}}]
x-whichapiapi:
  task: coding
  channels:
    reseller: {base_url: https://api.reseller.test/v1, key_env: RESELLER_API_KEY}
"""


def test_new_candidates_skip_models_already_in_suite(tmp_path, monkeypatch):
    path = tmp_path / "suite.yaml"
    path.write_text(SUITE)
    seen = {}

    def fake(task, n, preset, store=None):
        seen["task"] = task
        return ["openrouter:gpt-6-luna", "deepinfra:glm-5.3", "reseller:deepseek-v4.1-flash"]

    monkeypatch.setattr(core, "suggest_candidates", fake)
    found = refresh.new_candidates(path)
    assert seen["task"] == "coding"
    assert found["new"] == ["deepinfra:glm-5.3", "reseller:deepseek-v4.1-flash"]

    assert refresh.add_providers(path, found["new"]) == []
    text = path.read_text()
    assert text.startswith("# generated header\n")
    suite = load_suite(path)
    assert [p.id for p in suite.providers][1:] == found["new"]
    assert suite.providers[0].current and suite.ext.task == "coding"
    assert suite.providers[2].config == {"temperature": 0}
    assert "deepinfra" in yaml.safe_load(text)["x-whichapiapi"]["channels"]


def test_non_chat_suite_is_skipped(tmp_path):
    path = tmp_path / "suite.yaml"
    path.write_text(SUITE.replace("task: coding", "capability: speech.stt"))
    assert refresh.new_candidates(path)["skipped"]


def test_record_winner_reports_only_changes(tmp_path):
    path = tmp_path / "suite.yaml"
    path.write_text(SUITE)
    assert refresh.record_winner(path, "a", "b") is None
    assert refresh.record_winner(path, "a", "c") is None
    assert refresh.record_winner(path, "d", "a")["primary"] == "a"
