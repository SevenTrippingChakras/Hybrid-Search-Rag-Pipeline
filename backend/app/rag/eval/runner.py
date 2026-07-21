"""The eval runner: score the pipeline over a golden set into results.json.

For each golden question it runs the pipeline, scores every metric, and writes an
``EvalResults`` file. One run scores one chunking strategy, so the index must
already hold that strategy's chunks.
"""

import argparse
import asyncio
import json
from datetime import UTC, datetime
from pathlib import Path

from app.rag.eval.metrics import (
    abstention_accuracy,
    citation_accuracy,
    retrieval_relevance,
)
from app.rag.eval.ragas_scorer import RagasScorer
from app.rag.eval.results import EvalResults, QuestionResult, RunInfo
from app.rag.pipeline import Pipeline
from app.rag.stores import index_for_strategy


def _retrieved_sources(hits) -> list[str]:
    """Unique source docs among the retrieved chunks, retrieval order preserved."""
    seen: list[str] = []
    for hit in hits:
        src = hit.metadata.get("source")
        if src and src not in seen:
            seen.append(src)
    return seen


async def _score_question(question: dict, pipeline, scorer) -> QuestionResult:
    """Run one golden question through the pipeline and score every metric."""
    result = pipeline.answer(question["question"])
    retrieved = _retrieved_sources(result.hits)
    answer_text = result.answer.text if result.answer else None

    scores: dict[str, float] = {}
    local_scores = {
        "retrieval_relevance": retrieval_relevance(retrieved, question["gold_sources"]),
        "citation_accuracy": citation_accuracy(result.checks),
        "abstention_accuracy": abstention_accuracy(
            question["category"], result.abstained
        ),
    }
    for name, value in local_scores.items():
        if value is not None:
            scores[name] = value

    if answer_text and not result.abstained:
        scores.update(
            await scorer.score(
                question=question["question"],
                answer=answer_text,
                contexts=[hit.document for hit in result.hits],
                reference=question["answer"],
            )
        )

    return QuestionResult(
        id=question["id"],
        category=question["category"],
        question=question["question"],
        golden_answer=question["answer"],
        answer=answer_text,
        retrieved_sources=retrieved,
        gold_sources=question["gold_sources"],
        abstained=result.abstained,
        scores=scores,
        retrieved_contexts=[
            {"source": hit.metadata.get("source", ""), "text": hit.document}
            for hit in result.hits
        ],
    )


async def run_eval(
    golden_path: str | Path,
    strategy: str,
    sample: str = "golden_30",
    pipeline=None,
    scorer=None,
    limit: int | None = None,
) -> EvalResults:
    """Score every golden question and return a saveable ``EvalResults``."""
    questions = json.loads(Path(golden_path).read_text(encoding="utf-8"))
    if limit is not None:
        questions = questions[:limit]
    pipeline = pipeline or Pipeline(index=index_for_strategy(strategy))
    scorer = scorer or RagasScorer()

    results = []
    for i, question in enumerate(questions, start=1):
        results.append(await _score_question(question, pipeline, scorer))
        print(f"[{i}/{len(questions)}] {question['id']} scored")

    run = RunInfo(
        chunking_strategy=strategy,
        sample=sample,
        n=len(results),
        timestamp=datetime.now(UTC).isoformat(),
    )
    return EvalResults(run=run, results=results)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the RAG eval over a golden set.")
    parser.add_argument("--golden", default="data/eval_test/golden_30.json")
    parser.add_argument(
        "--strategy", required=True, choices=["fixed", "header", "semantic"]
    )
    parser.add_argument("--out", required=True, help="Where to write results.json.")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    results = asyncio.run(run_eval(args.golden, args.strategy, limit=args.limit))
    results.save(args.out)
    print(f"\nSaved {args.out}")
    print(json.dumps(results.aggregate, indent=2))


if __name__ == "__main__":
    main()
