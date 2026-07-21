"""Build the reduced eval_test corpus + question set.

Selects a balanced 30-question subset (22 multi-hop across the three reasoning
types + 8 no-answer) and the 300-doc corpus that serves it: every gold doc the
questions need, plus fixed-seed filler as retrieval distractors.

Outputs data/eval_test/corpus/ and data/eval_test/golden_30.json.
"""

import json
import random
import shutil
from pathlib import Path

SOURCE_CORPUS = Path("data/multihop/corpus")
SOURCE_GOLDEN = Path("data/multihop/golden_sample.json")
OUT_DIR = Path("data/eval_test")
OUT_CORPUS = OUT_DIR / "corpus"
OUT_GOLDEN = OUT_DIR / "golden_30.json"

TOTAL_DOCS = 300
MULTI_HOP_PER_TYPE = {"comparison_query": 8, "inference_query": 7, "temporal_query": 7}
NUM_NO_ANSWER = 8
SEED = 7


def _smallest_footprint_first(questions: list[dict]) -> list[dict]:
    """Fewest gold sources first -- keeps the required-doc set tight."""
    return sorted(questions, key=lambda q: len(q["gold_sources"]))


def select_questions(golden: list[dict]) -> list[dict]:
    """Balanced, deterministic 22 multi-hop + 8 no-answer."""
    chosen: list[dict] = []
    for subtype, n in MULTI_HOP_PER_TYPE.items():
        pool = [q for q in golden if q["subtype"] == subtype]
        chosen.extend(_smallest_footprint_first(pool)[:n])

    no_answer = [q for q in golden if q["category"] == "no_answer"]
    chosen.extend(no_answer[:NUM_NO_ANSWER])
    return chosen


def select_docs(questions: list[dict], all_docs: list[str]) -> list[str]:
    """Every gold doc the questions need, padded with filler distractors to 300."""
    gold = sorted({s for q in questions for s in q["gold_sources"]})
    filler_pool = sorted(set(all_docs) - set(gold))
    random.Random(SEED).shuffle(filler_pool)
    filler = filler_pool[: TOTAL_DOCS - len(gold)]
    return sorted(gold) + sorted(filler)


def main() -> None:
    golden = json.loads(SOURCE_GOLDEN.read_text(encoding="utf-8"))
    all_docs = [p.name for p in SOURCE_CORPUS.glob("*.md")]

    questions = select_questions(golden)
    docs = select_docs(questions, all_docs)
    gold_docs = {s for q in questions for s in q["gold_sources"]}

    OUT_CORPUS.mkdir(parents=True, exist_ok=True)
    for name in docs:
        shutil.copy2(SOURCE_CORPUS / name, OUT_CORPUS / name)

    OUT_GOLDEN.write_text(
        json.dumps(questions, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    from collections import Counter

    cats = Counter(q["category"] for q in questions)
    subs = Counter(q["subtype"] for q in questions)
    print(f"Questions: {len(questions)}  {dict(cats)}")
    print(f"  subtypes: {dict(subs)}")
    filler = len(docs) - len(gold_docs)
    print(f"Docs: {len(docs)}  ({len(gold_docs)} gold + {filler} filler)")
    print(f"Wrote {OUT_CORPUS}/ and {OUT_GOLDEN}")


if __name__ == "__main__":
    main()
