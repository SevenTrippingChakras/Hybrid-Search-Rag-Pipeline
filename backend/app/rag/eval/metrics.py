"""The three local (no-LLM) eval metrics.

Faithfulness and answer correctness are LLM-judged by RAGAS (see
``ragas_scorer.py``). Each metric here returns ``None`` when it does not apply to
a question; the aggregate skips ``None``s.
"""

from app.rag.models import CitationCheck


def retrieval_relevance(
    retrieved_sources: list[str], gold_sources: list[str]
) -> float | None:
    """Fraction of the question's gold docs found in the retrieved sources.

    ``None`` when there are no gold docs (no-answer questions).
    """
    if not gold_sources:
        return None
    retrieved = set(retrieved_sources)
    found = sum(1 for src in gold_sources if src in retrieved)
    return found / len(gold_sources)


def citation_accuracy(checks: list[CitationCheck]) -> float | None:
    """Fraction of the answer's citations whose passage supports its claim.

    ``None`` when the answer made no citations (nothing to verify).
    """
    if not checks:
        return None
    return sum(1 for c in checks if c.supported) / len(checks)


def abstention_accuracy(category: str, abstained: bool) -> float | None:
    """1.0 if a no-answer question was correctly refused, else 0.0.

    ``None`` for answerable questions (scored only on the no-answer set).
    """
    if category != "no_answer":
        return None
    return 1.0 if abstained else 0.0
