import json

import yaml

from whichapiapi import agents

URL, KEY = "https://router.test/v1", "wa-secretsecret123"


def test_render_every_tool_and_idempotence():
    claude = json.loads(agents.render("claude", '{"permissions": {"allow": ["Bash"]}}', URL, KEY))
    assert claude["permissions"] == {"allow": ["Bash"]}
    assert (
        claude["env"]["ANTHROPIC_BASE_URL"] == "https://router.test"
        and claude["env"]["ANTHROPIC_MODEL"] == "auto:coding"
    )
    codex = agents.render("codex", 'model = "gpt-5"\n', URL, KEY)
    assert codex.startswith('model = "gpt-5"') and 'wire_api = "responses"' in codex and KEY not in codex
    assert (
        agents.render("codex", codex, URL, KEY) == codex
    )  # re-running replaces the block, doesn't duplicate it
    aider = yaml.safe_load(agents.render("aider", "dark-mode: true\n", URL, KEY))
    assert aider == {"dark-mode": True, "openai-api-base": "https://router.test/v1", "openai-api-key": KEY,
                     "model": "openai/auto:coding"}  # fmt: skip
    oc = json.loads(agents.render("opencode", "", URL, KEY))
    assert oc["model"] == "whichapiapi/auto:coding" and oc["provider"]["whichapiapi"]["options"][
        "baseURL"
    ].endswith("/v1")
    cont = yaml.safe_load(agents.render("continue", agents.render("continue", "", URL, KEY), URL, KEY))
    assert [m["name"] for m in cont["models"]].count("Which API API") == 1


def test_setup_writes_with_backup_and_redacted_diff(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    cfg = tmp_path / ".aider.conf.yml"
    cfg.write_text("dark-mode: true\n")
    dry = agents.setup("aider", URL, KEY, dry_run=True)
    assert dry["changed"] and KEY not in dry["diff"] and "wa-se…" in dry["diff"]
    assert cfg.read_text() == "dark-mode: true\n"
    res = agents.setup("aider", URL, KEY)
    assert "backup" in res and (tmp_path / res["backup"]).read_text() == "dark-mode: true\n"
    assert oct(cfg.stat().st_mode & 0o777) == "0o600"
    assert agents.setup("aider", URL, KEY)["changed"] is False


def test_claude_project_scope_and_doctor_shadowing(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("CLAUDE_CONFIG_DIR", str(tmp_path / ".claude"))
    monkeypatch.delenv("ANTHROPIC_BASE_URL", raising=False)
    proj = tmp_path / "proj"
    agents.setup("claude", URL, KEY, project=proj)
    assert (proj / ".claude/settings.local.json").exists() and not (
        tmp_path / ".claude/settings.json"
    ).exists()
    monkeypatch.setattr(agents, "_probe", lambda url, key: "routed" if key == KEY else "degraded")
    assert agents.doctor(URL, proj)[0]["verdict"] == "routed"
    other = tmp_path / "other"  # ours only in the user file, a project file points elsewhere → shadowed
    agents.setup("claude", URL, KEY, global_=True)
    (other / ".claude").mkdir(parents=True)
    (other / ".claude/settings.json").write_text(
        '{"env": {"ANTHROPIC_BASE_URL": "https://api.anthropic.com"}}'
    )
    assert agents.doctor(URL, other)[0]["verdict"] == "shadowed"


def test_launch_env_strips_host_keys(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-host")
    env, args = agents.launch_env("claude", URL, KEY)
    assert "ANTHROPIC_API_KEY" not in env and env["ANTHROPIC_AUTH_TOKEN"] == KEY and args == []
    env, args = agents.launch_env("codex", URL, KEY)
    assert env["WHICHAPIAPI_KEY"] == KEY and 'model_providers.whichapiapi.wire_api="responses"' in args
