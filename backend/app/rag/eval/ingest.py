"""Full-corpus ingest driver.

Walks a corpus folder, loads every supported document (md/txt/html/pdf), chunks
each with the chosen strategy, and indexes all chunks into the OpenSearch store.
"""

import argparse
from dataclasses import dataclass
from pathlib import Path

from app.rag.chunking import chunk
from app.rag.index import Index
from app.rag.loaders import SUPPORTED_EXTENSIONS, load_file
from app.rag.stores import index_for_strategy


@dataclass
class IngestReport:
    """Totals from an ingest run."""

    files: int
    chunks_added: int
    chunks_skipped: int


def find_documents(corpus_dir: str | Path) -> list[Path]:
    """Every supported document under the corpus folder, sorted for determinism."""
    corpus_dir = Path(corpus_dir)
    files = [
        p
        for p in corpus_dir.rglob("*")
        if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS
    ]
    return sorted(files)


def ingest_corpus(
    corpus_dir: str | Path,
    strategy: str = "fixed",
    limit: int | None = None,
    index: Index | None = None,
) -> IngestReport:
    """Load, chunk, embed, and index every supported document in the corpus.

    Each strategy is written to its own index (``chunks_<strategy>``) so the three
    coexist without wiping. Documents are processed one at a time so dedup catches
    cross-document duplicates. ``limit`` ingests only the first N documents.
    """
    index = index or Index(index=index_for_strategy(strategy))
    files = find_documents(corpus_dir)
    if limit is not None:
        files = files[:limit]

    added = skipped = 0
    for i, path in enumerate(files, start=1):
        chunks = chunk(load_file(path), strategy=strategy)
        result = index.add(chunks)
        added += len(result.added)
        skipped += len(result.skipped)
        print(
            f"[{i}/{len(files)}] {path.name}: "
            f"+{len(result.added)} chunks ({len(result.skipped)} dup)"
        )

    return IngestReport(files=len(files), chunks_added=added, chunks_skipped=skipped)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Ingest a document corpus into the hybrid store (OpenSearch)."
    )
    parser.add_argument(
        "--corpus",
        default="data/multihop/corpus",
        help="Corpus folder to walk (default: data/multihop/corpus).",
    )
    parser.add_argument(
        "--strategy",
        default="fixed",
        choices=["fixed", "header", "semantic"],
        help="Chunking strategy to index (default: fixed).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Only ingest the first N documents (smoke test).",
    )
    args = parser.parse_args()

    report = ingest_corpus(args.corpus, strategy=args.strategy, limit=args.limit)
    print(
        f"\nDone. {report.files} files, {report.chunks_added} chunks indexed, "
        f"{report.chunks_skipped} duplicates skipped."
    )


if __name__ == "__main__":
    main()
