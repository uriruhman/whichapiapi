from whichapiapi.schema.suite import judge_spec, load_suite


def test_load_suite(fix):
    s = load_suite(fix / "suite/suite.yaml")
    assert len(s.prompts) == 1 and s.prompts[0].messages[0]["role"] == "system"
    assert [p.label for p in s.providers] == ["good", "bad", "broken"]
    assert s.providers[0].current and s.providers[0].price.input_per_1m == 1.0
    assert s.providers[0].model == 'const:{"answer": "Paris"}'
    assert len(s.tests) == 2 and s.tests[0].assert_[0].type == "python"
    assert len(s.all_asserts(s.tests[0])) == 3
    assert judge_spec(s).model == "judge"


def test_promptfoo_native_provider(tmp_path):
    (tmp_path / "s.yaml").write_text(
        "prompts: ['Hi {{x}}']\n"
        "providers:\n  - id: openai:chat:gpt-x\n    config: {apiBaseUrl: 'https://r.example/v1', apiKeyEnvar: RK, temperature: 0}\n"
        "tests: [{vars: {x: 1}}]\n"
    )
    s = load_suite(tmp_path / "s.yaml")
    p = s.providers[0]
    assert p.model == "gpt-x" and p.config == {"temperature": 0}
    ch = s.ext.channels[p.channel]
    assert ch.base_url == "https://r.example/v1" and ch.key_env == "RK"
