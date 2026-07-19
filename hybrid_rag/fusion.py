"""Reciprocal Rank Fusion: merge several ranked lists into one.

RRF scores a document by its *rank position* in each list, not by each list's
raw score, so incomparable scales (cosine vs. BM25) combine cleanly. A document
at rank ``r`` (1-based) in a list contributes ``weight / (k + r)``; contributions
sum across lists. ``k`` damps the curve so top ranks dominate without a single
first place swamping the rest.
"""

from hybrid_rag.stores import QueryHit


def reciprocal_rank_fusion(
    ranked_lists: list[tuple[list[QueryHit], float]],
    k: int = 60,
    top_k: int = 10,
) -> list[QueryHit]:
    """Fuse ``(hits, weight)`` lists into one ranked list of top-``top_k`` hits.

    Each returned ``QueryHit`` carries its fused RRF score (not the original
    cosine/BM25 score) and the metadata from where the document was first seen.
    """
    scores: dict[str, float] = {}
    hits: dict[str, QueryHit] = {}
    for ranked, weight in ranked_lists:
        for rank, hit in enumerate(ranked, start=1):
            scores[hit.id] = scores.get(hit.id, 0.0) + weight / (k + rank)
            hits.setdefault(hit.id, hit)
    fused = [
        QueryHit(id_, hits[id_].document, hits[id_].metadata, score)
        for id_, score in scores.items()
    ]
    fused.sort(key=lambda h: h.score, reverse=True)
    return fused[:top_k]
