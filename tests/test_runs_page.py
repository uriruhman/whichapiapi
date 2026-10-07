import json

from typer.testing import CliRunner

from whichapiapi.surfaces.cli import app


def test_runs_page_embeds_reports_and_escapes_script_end(tmp_path):
    run = tmp_path / "runs" / "abc123"
    run.mkdir(parents=True)
    report = {"title": "t", "summaries": [], "cases": [{"output": "</script><b>x" + "y" * 1000}]}
    (run / "report.json").write_text(json.dumps({"report": report}))
    out = tmp_path / "page.html"
    res = CliRunner().invoke(app, ["runs-page", str(out), "--dir", str(tmp_path / "runs")])
    assert res.exit_code == 0, res.output
    html = out.read_text()
    assert "/*__RUNS__*/" not in html and "abc123" in html
    data = html.split('id="runs-data" type="application/json">', 1)[1].split("</script>", 1)[0]
    assert "</" not in data
    assert len(json.loads(data)[0]["report"]["cases"][0]["output"]) == 600
