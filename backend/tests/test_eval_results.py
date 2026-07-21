"""Offline tests for the results.json contract: aggregate math and round-trip."""

from app.rag.eval.results import EvalResults, QuestionResult, RunInfo


def _q(id_, category, scores):
    return QuestionResult(
        id=id_,
        category=category,
        question="q",
        golden_answer="a",
        answer="a",
        retrieved_sources=[],
        gold_sources=[],
        abstained=False,
        scores=scores,
    )


def test_aggregate_means_overall_and_by_category():
    results = [
        _q("1", "multi_hop", {"faithfulness": 1.0, "context_recall": 0.5}),
        _q("2", "multi_hop", {"faithfulness": 0.0, "context_recall": 1.0}),
        _q("3", "no_answer", {"abstention_accuracy": 1.0}),
    ]
    agg = EvalResults(run=_run(), results=results).aggregate

    assert agg["faithfulness"] == 0.5  # (1.0 + 0.0) / 2
    assert agg["context_recall"] == 0.75  # (0.5 + 1.0) / 2
    assert agg["abstention_accuracy"] == 1.0
    assert agg["by_category"]["multi_hop"]["faithfulness"] == 0.5
    assert agg["by_category"]["no_answer"]["abstention_accuracy"] == 1.0
    # a metric only present in one category doesn't leak into the other
    assert "abstention_accuracy" not in agg["by_category"]["multi_hop"]


def test_save_load_round_trip(tmp_path):
    original = EvalResults(
        run=_run(), results=[_q("1", "multi_hop", {"faithfulness": 1.0})]
    )
    path = tmp_path / "results.json"
    original.save(path)
    loaded = EvalResults.load(path)

    assert loaded.run.chunking_strategy == "header"
    assert loaded.results[0].id == "1"
    assert loaded.results[0].scores == {"faithfulness": 1.0}
    assert loaded.aggregate == original.aggregate


def _run():
    return RunInfo(
        chunking_strategy="header",
        sample="golden_sample.json",
        n=3,
        timestamp="2026-07-21T00:00:00",
    )
