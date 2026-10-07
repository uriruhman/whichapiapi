import pytest

from whichapiapi.evaluator.assertions import AssertContext, combine, extract_json, run_assertion
from whichapiapi.schema.suite import Assertion


def ctx(tmp_path, **kw):
    return AssertContext(vars={"name": "Bob"}, base_dir=tmp_path, **kw)


@pytest.mark.parametrize(
    ("atype", "value", "output", "expected"),
    [
        ("equals", "hi Bob", " hi Bob ", True),
        ("equals", "{{name}}", "Bob", True),
        ("contains", "Bob", "hello Bob", True),
        ("icontains", "bob", "hello BOB", True),
        ("not-contains", "Bob", "hello Bob", False),
        ("contains-any", ["x", "Bob"], "Bob", True),
        ("contains-all", ["x", "Bob"], "Bob", False),
        ("regex", r"\d{3}", "code 123", True),
        ("not-regex", r"\d{3}", "code 12", True),
        ("is-json", None, '{"a": 1}', True),
        ("is-json", None, "nope", False),
        ("contains-json", {"type": "object", "required": ["a"]}, 'text ```json\n{"a": 1}\n```', True),
        ("contains-json", {"type": "object", "required": ["b"]}, '{"a": 1}', False),
        ("python", "len(output) > 3", "long enough", True),
        ("python", "return 0.2 if 'x' in output else 0.0\n", "x", True),
        ("bogus-type", None, "x", False),
    ],
)
async def test_assertions(tmp_path, atype, value, output, expected):
    r = await run_assertion(Assertion(type=atype, value=value), output, ctx(tmp_path))
    assert r.passed is expected, r.reason


async def test_cost_latency(tmp_path):
    c = ctx(tmp_path, cost=0.002, latency_ms=900)
    assert (await run_assertion(Assertion(type="cost", threshold=0.01), "", c)).passed
    assert not (await run_assertion(Assertion(type="latency", threshold=500), "", c)).passed


async def test_wer(tmp_path):
    c = ctx(tmp_path)
    exact = await run_assertion(
        Assertion(type="wer", value="the cat sat on the mat"), "the cat sat on the mat", c
    )
    assert exact.passed and exact.score == 1.0 and "0.00%" in exact.reason

    one_word_off = await run_assertion(
        Assertion(type="wer", value="the cat sat on the mat", threshold=0.2), "the cat sat on a mat", c
    )
    assert one_word_off.passed and 0.0 < one_word_off.score < 1.0  # 1/6 words wrong, under the 20% cutoff

    too_different = await run_assertion(
        Assertion(type="wer", value="the cat sat on the mat", threshold=0.1),
        "completely different words here",
        c,
    )
    assert not too_different.passed

    no_threshold = await run_assertion(Assertion(type="wer", value="hello"), "goodbye", c)
    assert no_threshold.passed  # no threshold given -> reports the rate, doesn't fail on it

    empty_ref = await run_assertion(Assertion(type="wer", value=""), "anything", c)
    assert not empty_ref.passed and "empty reference" in empty_ref.reason


async def test_broken_python_does_not_raise(tmp_path):
    r = await run_assertion(Assertion(type="python", value="1/0"), "x", ctx(tmp_path))
    assert not r.passed and "ZeroDivisionError" in r.reason


def test_extract_json():
    assert extract_json('pre {"a": [1, 2]} post') == {"a": [1, 2]}


def test_combine(tmp_path):
    from whichapiapi.evaluator.assertions import GradingResult

    rs = [
        GradingResult(type="a", passed=True, score=1.0, weight=1),
        GradingResult(type="b", passed=False, score=0.5, weight=3),
    ]
    assert combine(rs) == (False, 0.625)
    assert combine(rs, threshold=0.6) == (True, 0.625)


async def test_wer_normalizes_like_stt_benchmarks(tmp_path):
    from whichapiapi.evaluator.assertions import AssertContext, run_assertion
    from whichapiapi.schema.suite import Assertion

    c = AssertContext(vars={}, base_dir=tmp_path)
    ref = "MISTER QUILTER'S MANNER SEVEN THOUSAND ISLANDS"
    hyp = "Mr. Quilter's manner, 7000 islands."
    assert (await run_assertion(Assertion(type="wer", value=ref), hyp, c)).score == 1.0
    raw = await run_assertion(Assertion(type="wer", value=ref, normalize="none"), hyp, c)
    assert raw.score < 0.5
    ru = await run_assertion(Assertion(type="wer", value="ёлка в японии"), "Елка в Японии!", c)
    assert ru.score == 1.0
