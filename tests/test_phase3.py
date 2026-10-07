"""Phase 3 scaffold: search.web and speech.stt run through the same evaluator/selector as llm.chat.
Both example suites use `mock:const:...` providers (no network, no spend) — see examples/{search,speech_stt}.
"""

from whichapiapi.evaluator.runner import EvalRunner
from whichapiapi.schema.suite import load_suite
from whichapiapi.selector.pareto import recommend
from whichapiapi.store.db import Store


async def test_search_scaffold_runs_and_picks_relevant_provider(tmp_path):
    suite = load_suite("examples/search/suite.yaml")
    assert suite.ext.capability == "search.web"
    runner = EvalRunner(
        suite, Store(f"sqlite:///{tmp_path / 't.db'}"), suite_path="examples/search/suite.yaml"
    )
    rep = await runner.run()
    by = {s.label: s for s in rep.summaries}
    assert by["good"].pass_rate == 1.0 and by["bad"].pass_rate == 0.0
    rec = recommend(rep.summaries, suite.ext.weights, suite.ext.min_pass_rate)
    assert (
        rec.primary
        == "mock:const:1. Da Nang weather forecast - accuweather.com\\n2. CVE-2026-1234 self-hosting advisory - thehackernews.com"
    )


async def test_speech_stt_scaffold_runs_and_wer_differentiates(tmp_path):
    suite = load_suite("examples/speech_stt/suite.yaml")
    assert suite.ext.capability == "speech.stt"
    runner = EvalRunner(
        suite, Store(f"sqlite:///{tmp_path / 't.db'}"), suite_path="examples/speech_stt/suite.yaml"
    )
    rep = await runner.run()
    by = {s.label: s for s in rep.summaries}
    assert by["good"].n == 2 and by["so-so"].n == 2
    # "good" nails case 1 exactly (WER 0) and both get the numbers case wrong -> good scores no worse
    assert by["good"].score >= by["so-so"].score
