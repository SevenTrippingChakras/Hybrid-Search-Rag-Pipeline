"""Build the golden Q&A dataset from MultiHop-RAG.

Downloads the public MultiHop-RAG dataset (609 real news documents + 2,556
human-verified queries) and turns it into two things this pipeline can consume:

- ``data/multihop/corpus/`` — one markdown file per document, ingestible by the
  existing ``ingest_file`` loader (title becomes the ``#`` heading).
- ``data/multihop/golden.json`` — the golden Q&A set: each query with its answer,
  category, and the corpus files that hold the supporting evidence (gold docs
  for the retrieval-relevance metric). ``golden_sample.json`` is a small,
  type-balanced subset for cheap eval runs.

Everything under ``data/`` is gitignored and regenerable — this script is the
committed source of truth. Run: ``uv run python scripts/build_multihop_dataset.py``.

Source: https://huggingface.co/datasets/yixuantt/MultiHopRAG (ODC-by).
"""

import json
import urllib.request
from pathlib import Path

_BASE = "https://huggingface.co/datasets/yixuantt/MultiHopRAG/resolve/main"
_OUT = Path("data/multihop")
_RAW = _OUT / "raw"
_CORPUS_DIR = _OUT / "corpus"

# MultiHop-RAG question types mapped to our categories.
_CATEGORY = {
    "comparison_query": "multi_hop",
    "inference_query": "multi_hop",
    "temporal_query": "multi_hop",
    "null_query": "no_answer",
}
_SAMPLE_PER_SUBTYPE = 15


def _download(name: str) -> Path:
    """Fetch a raw dataset file into data/multihop/raw/ if not already there."""
    _RAW.mkdir(parents=True, exist_ok=True)
    dest = _RAW / name
    if not dest.exists():
        print(f"downloading {name} ...")
        urllib.request.urlretrieve(f"{_BASE}/{name}", dest)
    return dest


def _write_corpus(corpus: list[dict]) -> dict[str, str]:
    """Write each document as a markdown file. Return title -> filename map."""
    _CORPUS_DIR.mkdir(parents=True, exist_ok=True)
    title_to_file: dict[str, str] = {}
    for i, doc in enumerate(corpus):
        name = f"doc_{i:04d}.md"
        (_CORPUS_DIR / name).write_text(
            f"# {doc['title']}\n\n{doc['body']}", encoding="utf-8"
        )
        title_to_file[doc["title"]] = name
    return title_to_file


def _build_golden(queries: list[dict], title_to_file: dict[str, str]) -> list[dict]:
    """Turn raw queries into golden records with resolved gold-source files."""
    golden = []
    for i, q in enumerate(queries):
        subtype = q["question_type"]
        gold_sources = sorted({title_to_file[e["title"]] for e in q["evidence_list"]})
        golden.append(
            {
                "id": f"mhrag-{i:04d}",
                "question": q["query"],
                "answer": q["answer"],
                "category": _CATEGORY[subtype],
                "subtype": subtype,
                "gold_sources": gold_sources,
                "evidence_facts": [e["fact"] for e in q["evidence_list"]],
            }
        )
    return golden


def _stratified_sample(golden: list[dict]) -> list[dict]:
    """A deterministic, type-balanced subset for cost-controlled eval runs."""
    by_subtype: dict[str, list[dict]] = {}
    for rec in golden:
        by_subtype.setdefault(rec["subtype"], []).append(rec)
    sample = []
    for recs in by_subtype.values():
        sample.extend(recs[:_SAMPLE_PER_SUBTYPE])
    return sample


def main() -> None:
    corpus = json.loads(_download("corpus.json").read_text(encoding="utf-8"))
    queries = json.loads(_download("MultiHopRAG.json").read_text(encoding="utf-8"))

    title_to_file = _write_corpus(corpus)
    golden = _build_golden(queries, title_to_file)
    sample = _stratified_sample(golden)

    (_OUT / "golden.json").write_text(
        json.dumps(golden, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (_OUT / "golden_sample.json").write_text(
        json.dumps(sample, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    counts: dict[str, int] = {}
    for rec in golden:
        counts[rec["category"]] = counts.get(rec["category"], 0) + 1
    print(f"corpus files: {len(title_to_file)} -> {_CORPUS_DIR}")
    print(f"golden: {len(golden)} ({counts}) -> {_OUT / 'golden.json'}")
    print(f"sample: {len(sample)} -> {_OUT / 'golden_sample.json'}")


if __name__ == "__main__":
    main()
