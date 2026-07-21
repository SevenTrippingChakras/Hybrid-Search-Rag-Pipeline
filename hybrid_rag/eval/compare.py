"""The combined chunking-strategy comparison report.

Reads several ``results.json`` files (one per strategy) and renders one HTML page:
a metric-by-strategy table with each row's winner highlighted, a per-category
breakdown, and each strategy's lowest-scoring questions for drill-down.
"""

import argparse
from pathlib import Path

from hybrid_rag.eval.report import (
    STYLE,
    drilldown_section,
    esc,
    fmt,
    label,
    metric_counts,
    ordered_metrics,
)
from hybrid_rag.eval.results import EvalResults


def _best_indices(values: list[float | None]) -> set[int]:
    """Indices sharing the highest (rounded) value; empty on a full tie."""
    present = [
        (i, round(v, 2)) for i, v in enumerate(values) if isinstance(v, (int, float))
    ]
    if not present:
        return set()
    top = max(v for _, v in present)
    winners = {i for i, v in present if v == top}
    return set() if len(winners) == len(present) else winners


def _unique_winner(values: list[float | None]) -> int | None:
    """The single best index, or ``None`` on a tie -- ties count for nobody."""
    winners = _best_indices(values)
    return next(iter(winners)) if len(winners) == 1 else None


def _metric_n_label(metric: str, runs: list[EvalResults], n_pick) -> str:
    """Metric name plus its averaging denominator; a range when it varies by run."""
    counts = [c for c in (metric_counts(n_pick(r)).get(metric, 0) for r in runs) if c]
    if not counts:
        return esc(label(metric))
    lo, hi = min(counts), max(counts)
    n = f"n={lo}" if lo == hi else f"n={lo}-{hi}"
    return f"{esc(label(metric))} <span class='n'>{n}</span>"


def _comparison_table(runs: list[EvalResults], title: str, pick, n_pick=None) -> str:
    """One row per metric, one column per strategy, best cell(s) highlighted.

    ``pick`` returns the score dict to read (overall or a category slice);
    ``n_pick`` returns the results the denominator is counted over.
    """
    strategies = [r.run.chunking_strategy for r in runs]
    score_dicts = [pick(r) for r in runs]
    metrics = ordered_metrics({m for d in score_dicts for m in d})

    head = "".join(f"<th class='num'>{esc(s)}</th>" for s in strategies)
    rows = ""
    for metric in metrics:
        values = [d.get(metric) for d in score_dicts]
        winners = _best_indices(values)
        cells = ""
        for i, value in enumerate(values):
            cls = "num best" if i in winners else "num"
            cells += f"<td class='{cls}'>{fmt(value)}</td>"
        name = _metric_n_label(metric, runs, n_pick) if n_pick else esc(label(metric))
        rows += f"<tr><td>{name}</td>{cells}</tr>"
    return f"<h2>{esc(title)}</h2><table><tr><th>Metric</th>{head}</tr>{rows}</table>"


def _overall(results: EvalResults) -> dict:
    return {k: v for k, v in results.aggregate.items() if k != "by_category"}


def _category(results: EvalResults, category: str) -> dict:
    return results.aggregate.get("by_category", {}).get(category, {})


def _winner_summary(runs: list[EvalResults]) -> str:
    """Count clear (non-tied) metric wins per strategy to name a leader."""
    strategies = [r.run.chunking_strategy for r in runs]
    overalls = [_overall(r) for r in runs]
    metrics = ordered_metrics({m for d in overalls for m in d})
    wins = dict.fromkeys(strategies, 0)
    ties = 0
    for metric in metrics:
        winner = _unique_winner([d.get(metric) for d in overalls])
        if winner is None:
            ties += 1
        else:
            wins[strategies[winner]] += 1
    parts = ", ".join(f"{s}: {n}" for s, n in wins.items())
    top = max(wins.values())
    leaders = [s for s, n in wins.items() if n == top and n > 0]
    tie_note = f" ({ties} tied)" if ties else ""
    if len(leaders) == 1:
        verdict = f"Leader: <b>{esc(leaders[0])}</b>"
    elif leaders:
        verdict = f"No clear leader ({esc(', '.join(leaders))} level)"
    else:
        verdict = "No metric had a clear winner"
    return f"<p class='sub'>Metric wins - {esc(parts)}{tie_note}. {verdict}.</p>"


def render_comparison(runs: list[EvalResults]) -> str:
    """Full comparison HTML across the given per-strategy runs."""
    categories = sorted({c for r in runs for c in r.aggregate.get("by_category", {})})
    body = [
        "<h1>Chunking strategy comparison</h1>",
        _winner_summary(runs),
        _comparison_table(runs, "Overall", _overall, n_pick=lambda r: r.results),
    ]
    for category in categories:
        body.append(
            _comparison_table(
                runs,
                f"By category: {category}",
                lambda r, c=category: _category(r, c),
                n_pick=lambda r, c=category: [x for x in r.results if x.category == c],
            )
        )
    for run in runs:
        body.append(
            drilldown_section(
                run.results, f"Lowest-scoring - {run.run.chunking_strategy}"
            )
        )
    return (
        f"<!doctype html><html><head><meta charset='utf-8'><style>{STYLE}</style>"
        f"</head><body>{''.join(body)}</body></html>"
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Render a combined chunking-strategy comparison report."
    )
    parser.add_argument("results", nargs="+", help="Per-strategy results.json files.")
    parser.add_argument("--out", required=True, help="Path to write the HTML report.")
    args = parser.parse_args()

    runs = [EvalResults.load(p) for p in args.results]
    Path(args.out).write_text(render_comparison(runs), encoding="utf-8")
    print(f"Wrote {args.out} ({len(runs)} strategies)")


if __name__ == "__main__":
    main()
