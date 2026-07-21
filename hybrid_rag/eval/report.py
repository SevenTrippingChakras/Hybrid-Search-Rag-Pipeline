"""Static HTML report for one eval run.

Renders an ``EvalResults`` into a self-contained HTML file: metric means, a
per-category breakdown, and a drill-down into the lowest-scoring questions. The
shared render helpers are reused by ``compare.py``.
"""

import argparse
import html
from pathlib import Path

from hybrid_rag.eval.results import EvalResults, QuestionResult

METRIC_ORDER = [
    "retrieval_relevance",
    "citation_accuracy",
    "faithfulness",
    "answer_correctness",
    "abstention_accuracy",
]
METRIC_LABELS = {
    "retrieval_relevance": "Retrieval relevance",
    "citation_accuracy": "Citation accuracy",
    "faithfulness": "Faithfulness",
    "answer_correctness": "Answer correctness",
    "abstention_accuracy": "Abstention accuracy",
}

STYLE = """
body { font: 15px/1.5 -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
       color: #1a1a1a; background: #fafafa; margin: 0; padding: 2rem; }
h1 { font-size: 1.5rem; margin: 0 0 .25rem; }
h2 { font-size: 1.15rem; margin: 2rem 0 .75rem;
     border-bottom: 2px solid #eee; padding-bottom: .3rem; }
.sub { color: #666; margin: 0 0 1.5rem; font-size: .9rem; }
table { border-collapse: collapse; width: 100%; background: #fff;
        margin-bottom: 1rem; }
th, td { text-align: left; padding: .5rem .7rem; border-bottom: 1px solid #eee; }
th { background: #f4f4f6; font-weight: 600; }
td.num, th.num { text-align: right; font-variant-numeric: tabular-nums; }
td.best { background: #e6f7ed; font-weight: 700; color: #0a6b34; }
.card { background: #fff; border: 1px solid #eee; border-radius: 8px;
        padding: 1rem 1.2rem; margin-bottom: 1rem; }
.card h3 { margin: 0 0 .5rem; font-size: 1rem; }
.q { color: #333; font-weight: 600; }
.ans { white-space: pre-wrap; background: #f8f8fa; border-radius: 6px;
       padding: .5rem .7rem; margin: .4rem 0; }
.gold { color: #0a6b34; } .gen { color: #1a4fa0; }
.tag { display: inline-block; font-size: .75rem; padding: .1rem .5rem;
       border-radius: 10px; background: #eef; margin-left: .5rem; }
.miss { color: #b00020; } .hit { color: #0a6b34; }
.scores span { display: inline-block; margin-right: .8rem; font-size: .85rem;
               color: #444; }
.n { color: #999; font-weight: 400; font-size: .8em; }
details { margin-bottom: 1rem; }
summary { cursor: pointer; font-weight: 600; font-size: 1.15rem;
          margin: 2rem 0 .75rem;
          border-bottom: 2px solid #eee; padding-bottom: .3rem; }
.chunks { margin: .4rem 0; }
.chunk { font-size: .82rem; color: #444; background: #f8f8fa;
         border-left: 3px solid #ccd;
         border-radius: 4px; padding: .35rem .6rem; margin: .3rem 0; }
.chunk b { color: #556; }
"""


def esc(text) -> str:
    return html.escape(str(text))


def fmt(value) -> str:
    return f"{value:.2f}" if isinstance(value, (int, float)) else "-"


def ordered_metrics(present) -> list[str]:
    """Canonical metric order, with any unknown metrics appended alphabetically."""
    present = set(present)
    known = [m for m in METRIC_ORDER if m in present]
    return known + sorted(present - set(METRIC_ORDER))


def label(metric: str) -> str:
    return METRIC_LABELS.get(metric, metric.replace("_", " ").capitalize())


def _question_mean(result: QuestionResult) -> float:
    values = list(result.scores.values())
    return sum(values) / len(values) if values else 0.0


def metric_counts(results: list[QuestionResult]) -> dict[str, int]:
    """How many questions actually carry each metric (its averaging denominator)."""
    counts: dict[str, int] = {}
    for result in results:
        for name in result.scores:
            counts[name] = counts.get(name, 0) + 1
    return counts


def _headline_table(aggregate: dict, results: list[QuestionResult]) -> str:
    metrics = ordered_metrics(k for k in aggregate if k != "by_category")
    counts = metric_counts(results)
    rows = "".join(
        f"<tr><td>{esc(label(m))}</td><td class='num'>{fmt(aggregate[m])}</td>"
        f"<td class='num n'>{counts.get(m, 0)}</td></tr>"
        for m in metrics
    )
    return (
        "<table><tr><th>Metric</th><th class='num'>Mean</th>"
        f"<th class='num'>n</th></tr>{rows}</table>"
    )


def _category_table(aggregate: dict) -> str:
    by_cat = aggregate.get("by_category", {})
    if not by_cat:
        return ""
    metrics = ordered_metrics({m for cat in by_cat.values() for m in cat})
    head = "".join(f"<th class='num'>{esc(label(m))}</th>" for m in metrics)
    rows = ""
    for cat, scores in by_cat.items():
        cells = "".join(f"<td class='num'>{fmt(scores.get(m))}</td>" for m in metrics)
        rows += f"<tr><td>{esc(cat)}</td>{cells}</tr>"
    return f"<h2>By category</h2><table><tr><th>Category</th>{head}</tr>{rows}</table>"


def _sources_line(result: QuestionResult) -> str:
    retrieved = set(result.retrieved_sources)
    if not result.gold_sources:
        return (
            f"<div>Retrieved: {esc(', '.join(result.retrieved_sources) or '-')}</div>"
        )
    parts = []
    for src in result.gold_sources:
        css = "hit" if src in retrieved else "miss"
        mark = "OK" if src in retrieved else "MISS"
        parts.append(f"<span class='{css}'>{esc(src)} [{mark}]</span>")
    return f"<div>Gold sources: {' '.join(parts)}</div>"


def _chunks_block(result: QuestionResult, limit_chars: int = 300) -> str:
    """The retrieved chunk texts, so a low score can be read against the evidence."""
    if not result.retrieved_contexts:
        return ""
    items = ""
    for chunk in result.retrieved_contexts:
        text = str(chunk.get("text", ""))
        if len(text) > limit_chars:
            text = text[:limit_chars].rstrip() + "..."
        items += (
            f"<div class='chunk'><b>{esc(chunk.get('source', '?'))}</b>: "
            f"{esc(text)}</div>"
        )
    return f"<div class='chunks'>{items}</div>"


def _question_card(result: QuestionResult) -> str:
    scores = " ".join(
        f"<span>{esc(label(m))}: <b>{fmt(result.scores[m])}</b></span>"
        for m in ordered_metrics(result.scores)
    )
    abstained = "<span class='tag'>abstained</span>" if result.abstained else ""
    answer = result.answer if result.answer else "(no answer generated)"
    return (
        f"<div class='card'><h3>{esc(result.id)} "
        f"<span class='tag'>{esc(result.category)}</span>{abstained}</h3>"
        f"<div class='q'>{esc(result.question)}</div>"
        f"<div class='ans gold'>Golden: {esc(result.golden_answer)}</div>"
        f"<div class='ans gen'>Generated: {esc(answer)}</div>"
        f"{_sources_line(result)}"
        f"{_chunks_block(result)}"
        f"<div class='scores'>{scores}</div></div>"
    )


def drilldown_section(results: list[QuestionResult], title: str, n: int = 5) -> str:
    """Collapsible cards for the ``n`` lowest-scoring questions (by metric mean)."""
    worst = sorted(results, key=_question_mean)[:n]
    cards = "".join(_question_card(r) for r in worst)
    return f"<details><summary>{esc(title)}</summary>{cards}</details>"


def render_report(results: EvalResults) -> str:
    """Full single-run HTML."""
    run = results.run
    body = (
        f"<h1>Eval report - {esc(run.chunking_strategy)} chunking</h1>"
        f"<p class='sub'>sample {esc(run.sample)} - {run.n} questions"
        f" - {esc(run.timestamp)}</p>"
        f"<h2>Overall</h2>{_headline_table(results.aggregate, results.results)}"
        f"{_category_table(results.aggregate)}"
        f"{drilldown_section(results.results, 'Lowest-scoring questions')}"
    )
    return (
        f"<!doctype html><html><head><meta charset='utf-8'><style>{STYLE}</style>"
        f"</head><body>{body}</body></html>"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Render a single eval run to HTML.")
    parser.add_argument("results", help="Path to a results.json.")
    parser.add_argument("--out", required=True, help="Path to write the HTML report.")
    args = parser.parse_args()

    results = EvalResults.load(args.results)
    Path(args.out).write_text(render_report(results), encoding="utf-8")
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
