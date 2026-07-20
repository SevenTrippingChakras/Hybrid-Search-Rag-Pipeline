"""Citation verification with an LLM-as-judge.

A ``[n]`` marker asserts a citation but does not prove it. ``CitationVerifier``
splits an answer into claim/citation pairs and asks a judge, for each pair,
whether the cited passage actually supports the claim; unsupported ones are flagged.
"""

import re

from pydantic import BaseModel

from hybrid_rag.llm import LLM, build_llm
from hybrid_rag.models import Answer, CitationCheck

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


class _Verdict(BaseModel):
    """The judge's reply: whether the passage supports the claim, and why."""

    supported: bool
    reason: str


class CitationVerifier:
    """Verifies each claim/citation pair in an ``Answer`` with an LLM judge."""

    def __init__(self, llm: LLM | None = None) -> None:
        self._llm = llm or build_llm()

    def verify(self, answer: Answer) -> list[CitationCheck]:
        """Judge every cited claim in ``answer`` against its backing passage."""
        backing = {c.number: c.text for c in answer.citations}
        checks = []
        for claim, number in self._claim_citation_pairs(answer.text):
            passage = backing.get(number)
            if passage is None:
                continue  # marker with no resolved citation (dropped upstream)
            verdict = self._llm.parse(
                JUDGE_SYSTEM, self._build_prompt(claim, passage), _Verdict
            )
            checks.append(
                CitationCheck(
                    claim=claim,
                    number=number,
                    supported=verdict.supported,
                    reason=verdict.reason,
                )
            )
        return checks

    @staticmethod
    def _claim_citation_pairs(text: str) -> list[tuple[str, int]]:
        """Split into sentences, pairing each with every number it cites."""
        pairs = []
        for sentence in _SENTENCE.split(text.strip()):
            numbers = [int(n) for n in _MARKER.findall(sentence)]
            claim = _MARKER.sub("", sentence).strip()
            for number in numbers:
                pairs.append((claim, number))
        return pairs

    @staticmethod
    def _build_prompt(claim: str, passage: str) -> str:
        return f"CLAIM: {claim}\n\nPASSAGE:\n{passage}"
