"""The ``results.json`` contract: the eval's source of truth.

The runner fills these structures per question; the report only reads the written
file. Aggregates are derived from the per-question results here, so the runner and
report can never disagree on how a headline number was computed.
"""

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path


@dataclass
class QuestionResult:
    """One golden question scored end to end."""

    id: str
    category: str
    question: str
    golden_answer: str
    answer: str | None
    retrieved_sources: list[str]
    gold_sources: list[str]
    abstained: bool
    scores: dict[str, float] = field(default_factory=dict)


@dataclass
class RunInfo:
    """What configuration produced a run -- so results are comparable."""

    chunking_strategy: str
    sample: str
    n: int
    timestamp: str


@dataclass
class EvalResults:
    """A whole run: config, derived aggregate, and per-question drill-down."""

    run: RunInfo
    results: list[QuestionResult]
    aggregate: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.aggregate:
            self.aggregate = _compute_aggregate(self.results)

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "run": asdict(self.run),
            "aggregate": self.aggregate,
            "results": [asdict(r) for r in self.results],
        }
        path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), "utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "EvalResults":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(
            run=RunInfo(**data["run"]),
            results=[QuestionResult(**r) for r in data["results"]],
            aggregate=data["aggregate"],
        )


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _means_over(results: list[QuestionResult]) -> dict[str, float]:
    """Mean of each metric across the results that carry it (missing skipped)."""
    names = sorted({name for r in results for name in r.scores})
    return {n: _mean([r.scores[n] for r in results if n in r.scores]) for n in names}


def _compute_aggregate(results: list[QuestionResult]) -> dict:
    """Overall metric means plus a per-category breakdown."""
    aggregate: dict = _means_over(results)
    by_category: dict = {}
    for category in sorted({r.category for r in results}):
        by_category[category] = _means_over(
            [r for r in results if r.category == category]
        )
    aggregate["by_category"] = by_category
    return aggregate
