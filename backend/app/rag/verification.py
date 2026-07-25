"""Citation verification with an LLM-as-judge.

A ``[n]`` marker asserts a citation but does not prove it. ``CitationVerifier``
splits an answer into claim/citation pairs and asks a judge, for each pair,
whether the cited passage actually supports the claim; unsupported ones are flagged.
"""

import re
from concurrent.futures import ThreadPoolExecutor

from pydantic import BaseModel

from app.config import settings
from app.rag.llm import LLM, build_llm
from app.rag.models import Answer, CitationCheck

JUDGE_SYSTEM = (
    "You are a strict fact-checker. Given a CLAIM and a single source PASSAGE, "
    "decide whether the passage directly supports the claim.\n"
    "- Supported means the passage states or clearly entails the claim.\n"
    "- If the passage is merely related, off-topic, or only partially covers the "
    "claim, it is not supported.\n"
    "- Judge only against the passage. Never use outside knowledge.\n"
    "Give a one-sentence reason for your verdict."
)

_MARKER = re.compile(r"\s*\[(\d+)\]")
_SENTENCE = re.compile(r"(?<=[.!?])\s+")


def split_claims(text: str) -> list[str]:
    """Answer prose split into claims (sentences), each stripped of ``[n]`` markers.

    Shared with confidence scoring so a claim string produced here matches the
    ``claim`` on the ``CitationCheck`` it corresponds to.
    """
    claims = []
    for sentence in _SENTENCE.split(text.strip()):
        claim = _MARKER.sub("", sentence).strip()
        if claim:
            claims.append(claim)
    return claims


class _Verdict(BaseModel):
    """The judge's reply: whether the passage supports the claim, and why."""

    supported: bool
    reason: str


class CitationVerifier:
    """Verifies each claim/citation pair in an ``Answer`` with an LLM judge."""

    def __init__(self, llm: LLM | None = None) -> None:
        self._llm = llm or build_llm()

    def verify(self, answer: Answer) -> list[CitationCheck]:
        """Judge every cited claim in ``answer`` against its backing passage.

        Each pair is an independent LLM round-trip, so they run concurrently on a
        bounded thread pool. ``map`` yields results in input order, keeping the
        returned checks deterministic regardless of which call finishes first.
        """
        backing = {c.number: c.text for c in answer.citations}
        pairs = [
            (claim, number, backing[number])
            for claim, number in self._claim_citation_pairs(answer.text)
            if number in backing  # skip markers with no resolved citation
        ]
        if not pairs:
            return []

        workers = min(settings.verification_max_workers, len(pairs)) or 1
        with ThreadPoolExecutor(max_workers=workers) as pool:
            return list(pool.map(self._judge_one, pairs))

    def _judge_one(self, pair: tuple[str, int, str]) -> CitationCheck:
        """Judge a single (claim, number, passage) pair."""
        claim, number, passage = pair
        verdict = self._llm.parse(
            JUDGE_SYSTEM, self._build_prompt(claim, passage), _Verdict
        )
        return CitationCheck(
            claim=claim,
            number=number,
            supported=verdict.supported,
            reason=verdict.reason,
        )

    @staticmethod
    def _claim_citation_pairs(text: str) -> list[tuple[str, int]]:
        """Split into sentences, pairing each with every number it cites."""
        pairs = []
        for sentence in _SENTENCE.split(text.strip()):
            claim = _MARKER.sub("", sentence).strip()
            for number in _MARKER.findall(sentence):
                pairs.append((claim, int(number)))
        return pairs

    @staticmethod
    def _build_prompt(claim: str, passage: str) -> str:
        return f"CLAIM: {claim}\n\nPASSAGE:\n{passage}"
