"""Tests for the HTML report and the combined comparison report.

Everything renders from in-memory EvalResults -- no files, no LLM, no browser.
"""

from app.rag.eval.compare import render_comparison
from app.rag.eval.report import render_report
from app.rag.eval.results import EvalResults, QuestionResult, RunInfo


def _q(id_, category, scores, retrieved, gold, abstained=False, contexts=None):
    return QuestionResult(
        id=id_,
        category=category,
        question=f"question {id_}?",
        golden_answer="gold",
        answer=None if abstained else "generated answer",
        retrieved_sources=retrieved,
        gold_sources=gold,
        abstained=abstained,
        scores=scores,
        retrieved_contexts=contexts or [],
    )


def _run(strategy, rr, correctness):
    results = [
        _q(
            "q1",
            "multi_hop",
            {
                "retrieval_relevance": rr,
                "citation_accuracy": 0.7,
                "answer_correctness": correctness,
            },
            ["a.md"],
            ["a.md", "b.md"],
            contexts=[{"source": "a.md", "text": "the retrieved chunk evidence text"}],
        ),
        _q(
            "q2",
            "no_answer",
            {"abstention_accuracy": 1.0},
            ["x.md"],
            [],
            abstained=True,
        ),
    ]
    run = RunInfo(
        chunking_strategy=strategy,
        sample="golden_30",
        n=len(results),
        timestamp="2026-07-21T00:00:00Z",
    )
    return EvalResults(run=run, results=results)


def test_single_report_contains_metrics_drilldown_and_chunks():
    html = render_report(_run("fixed", rr=0.5, correctness=0.7))

    assert "Eval report - fixed chunking" in html
    assert "Retrieval relevance" in html
    assert "0.50" in html
    assert "Lowest-scoring questions" in html
    assert "MISS" in html  # b.md gold source was not retrieved
    # #4: retrieved chunk text shows in the drill-down
    assert "the retrieved chunk evidence text" in html
    # #2: the n (denominator) column renders
    assert "<th class='num'>n</th>" in html


def test_comparison_names_a_clear_leader_when_one_strategy_wins():
    runs = [
        _run("fixed", rr=0.4, correctness=0.6),
        _run("header", rr=0.9, correctness=0.9),  # wins both differing metrics
        _run("semantic", rr=0.5, correctness=0.8),
    ]

    html = render_comparison(runs)

    assert "Chunking strategy comparison" in html
    assert "Leader: <b>header</b>" in html
    assert html.count("class='num best'") >= 2
    assert "By category: multi_hop" in html
    assert "By category: no_answer" in html
    assert "Lowest-scoring - header" in html


def test_comparison_does_not_highlight_or_credit_full_ties():
    # All three identical on citation_accuracy and abstention -> full ties.
    runs = [
        _run("fixed", rr=0.9, correctness=0.5),  # wins retrieval
        _run("header", rr=0.5, correctness=0.9),  # wins correctness
        _run("semantic", rr=0.5, correctness=0.5),
    ]

    html = render_comparison(runs)

    # citation_accuracy is 0.70 for every strategy: a full tie -> no clear leader
    assert "No clear leader" in html
    # the tie is disclosed
    assert "tied)" in html
