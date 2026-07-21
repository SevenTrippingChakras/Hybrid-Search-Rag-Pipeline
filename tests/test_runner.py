"""Tests for the eval runner, driven by a fake pipeline and fake RAGAS scorer.

No OpenSearch, no OpenAI: the runner's orchestration and the written results.json
are exercised entirely offline.
"""

import asyncio
import json

from hybrid_rag.eval.results import EvalResults
from hybrid_rag.eval.runner import run_eval
from hybrid_rag.models import Answer, CitationCheck
from hybrid_rag.pipeline import PipelineResult
from hybrid_rag.stores import QueryHit


def _hit(source, text="chunk text"):
    return QueryHit(
        id=f"{source}:0", document=text, metadata={"source": source}, score=1.0
    )


def _answered_result(question, sources):
    hits = [_hit(s) for s in sources]
    answer = Answer(query=question, text="An answer [1].", citations=[])
    checks = [CitationCheck(claim="c", number=1, supported=True, reason="")]
    return PipelineResult(query=question, hits=hits, answer=answer, checks=checks)


def _abstained_result(question, sources):
    from hybrid_rag.models import NoAnswer

    hits = [_hit(s) for s in sources]
    no_answer = NoAnswer(
        query=question,
        retrieval_confidence=0.1,
        message="idk",
        found=[],
        suggested_sources=[],
    )
    return PipelineResult(query=question, hits=hits, no_answer=no_answer)


class FakePipeline:
    def __init__(self, by_question):
        self._by_question = by_question

    def answer(self, question):
        return self._by_question[question]


class FakeScorer:
    async def score(self, question, answer, contexts, reference):
        return {"faithfulness": 1.0, "answer_correctness": 0.8}


def _golden(tmp_path):
    data = [
        {
            "id": "q-mh",
            "question": "multi hop question?",
            "answer": "the gold answer",
            "category": "multi_hop",
            "subtype": "inference_query",
            "gold_sources": ["a.md", "b.md"],
        },
        {
            "id": "q-na",
            "question": "unanswerable question?",
            "answer": "Insufficient information",
            "category": "no_answer",
            "subtype": "null_query",
            "gold_sources": [],
        },
    ]
    path = tmp_path / "golden.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    return path


def test_runner_scores_answered_and_abstained(tmp_path):
    golden = _golden(tmp_path)
    pipeline = FakePipeline(
        {
            "multi hop question?": _answered_result(
                "multi hop question?",
                ["a.md", "x.md"],  # 1 of 2 gold retrieved
            ),
            "unanswerable question?": _abstained_result(
                "unanswerable question?", ["y.md"]
            ),
        }
    )

    results = asyncio.run(
        run_eval(golden, strategy="fixed", pipeline=pipeline, scorer=FakeScorer())
    )

    by_id = {r.id: r for r in results.results}

    mh = by_id["q-mh"]
    assert mh.scores["retrieval_relevance"] == 0.5
    assert mh.scores["citation_accuracy"] == 1.0
    assert mh.scores["faithfulness"] == 1.0
    assert mh.scores["answer_correctness"] == 0.8
    assert "abstention_accuracy" not in mh.scores

    na = by_id["q-na"]
    assert na.abstained is True
    assert na.scores["abstention_accuracy"] == 1.0
    # No answer generated -> no LLM-judged metrics on an abstention
    assert "faithfulness" not in na.scores
    assert "retrieval_relevance" not in na.scores


def test_runner_output_roundtrips_through_results_file(tmp_path):
    golden = _golden(tmp_path)
    pipeline = FakePipeline(
        {
            "multi hop question?": _answered_result(
                "multi hop question?", ["a.md", "b.md"]
            ),
            "unanswerable question?": _abstained_result("unanswerable question?", []),
        }
    )
    results = asyncio.run(
        run_eval(golden, strategy="header", pipeline=pipeline, scorer=FakeScorer())
    )

    out = tmp_path / "results.json"
    results.save(out)
    loaded = EvalResults.load(out)

    assert loaded.run.chunking_strategy == "header"
    assert loaded.run.n == 2
    assert loaded.aggregate["retrieval_relevance"] == 1.0
    assert loaded.aggregate["abstention_accuracy"] == 1.0
    assert "multi_hop" in loaded.aggregate["by_category"]
    assert "no_answer" in loaded.aggregate["by_category"]
