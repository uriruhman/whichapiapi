"""Compare two runs of the same suite (e.g. two judges): model ranking agreement and per-case verdict agreement."""

from __future__ import annotations

import json
import statistics
from pathlib import Path
from typing import Any


def _load(run_dir: Path) -> dict[str, Any]:
    return json.loads((run_dir / "report.json").read_text())["report"]


def _rubric(case: dict[str, Any]) -> dict[str, Any] | None:
    return next((g for g in case["grades"] if g["type"] == "llm-rubric" and g["weight"] > 0), None)


def _ranks(values: dict[str, float]) -> dict[str, float]:
    order = sorted(values, key=lambda k: -values[k])
    return {k: i + 1 for i, k in enumerate(order)}


def spearman(a: dict[str, float], b: dict[str, float]) -> float | None:
    keys = [k for k in a if k in b]
    n = len(keys)
    if n < 3:
        return None
    ra, rb = _ranks({k: a[k] for k in keys}), _ranks({k: b[k] for k in keys})
    d2 = sum((ra[k] - rb[k]) ** 2 for k in keys)
    return round(1 - 6 * d2 / (n * (n * n - 1)), 3)


def compare_runs(a_dir: Path, b_dir: Path) -> dict[str, Any]:
    a, b = _load(a_dir), _load(b_dir)
    by_case_b = {(c["provider"], c["test_idx"]): c for c in b["cases"]}
    per_model: dict[str, dict[str, list[float]]] = {}
    agree = total = 0
    for c in a["cases"]:
        other = by_case_b.get((c["provider"], c["test_idx"]))
        ga, gb = _rubric(c), _rubric(other) if other else None
        if not ga or not gb:
            continue
        m = per_model.setdefault(c["provider"], {"a": [], "b": [], "sa": [], "sb": []})
        m["a"].append(ga["score"])
        m["b"].append(gb["score"])
        m["sa"].append(c["score"])
        m["sb"].append(other["score"])
        total += 1
        agree += ga["passed"] == gb["passed"]
    rows = {
        p: {
            "rubric_a": round(statistics.fmean(v["a"]), 3),
            "rubric_b": round(statistics.fmean(v["b"]), 3),
            "score_a": round(statistics.fmean(v["sa"]), 3),
            "score_b": round(statistics.fmean(v["sb"]), 3),
            "n": len(v["a"]),
        }
        for p, v in per_model.items()
    }
    return {
        "a": {"run": a["run_id"], "judge": a.get("judge_label")},
        "b": {"run": b["run_id"], "judge": b.get("judge_label")},
        "models": rows,
        "rank_correlation_rubric": spearman(
            {p: r["rubric_a"] for p, r in rows.items()}, {p: r["rubric_b"] for p, r in rows.items()}
        ),
        "rank_correlation_total": spearman(
            {p: r["score_a"] for p, r in rows.items()}, {p: r["score_b"] for p, r in rows.items()}
        ),
        "case_verdict_agreement": round(agree / total, 3) if total else None,
        "cases_compared": total,
    }


def to_markdown(res: dict[str, Any]) -> str:
    a, b = res["a"]["judge"], res["b"]["judge"]
    lines = [
        f"| Model | Rubric ({a}) | Rubric ({b}) | Total ({a}) | Total ({b}) |",
        "|---|---:|---:|---:|---:|",
    ]
    for p, r in sorted(res["models"].items(), key=lambda kv: -kv[1]["score_a"]):
        lines.append(
            f"| {p.split(':', 1)[-1]} | {r['rubric_a']:.2f} | {r['rubric_b']:.2f} | {r['score_a']:.2f} | {r['score_b']:.2f} |"
        )
    lines += [
        "",
        f"- Rank correlation (Spearman) on rubric: **{res['rank_correlation_rubric']}**, "
        f"on total score: **{res['rank_correlation_total']}**",
        f"- Per-case pass/fail agreement of the two judges: **{res['case_verdict_agreement']:.0%}** "
        f"over {res['cases_compared']} cases",
    ]
    return "\n".join(lines) + "\n"
