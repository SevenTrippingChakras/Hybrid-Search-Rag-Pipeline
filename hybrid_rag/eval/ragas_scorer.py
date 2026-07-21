"""The two LLM-judged metrics: faithfulness and answer correctness.

Wraps RAGAS's ``Faithfulness`` and ``AnswerCorrectness`` behind one async
``score`` call, built on the shared OpenAI judge from ``judge.py``.
"""

from ragas.metrics.collections import AnswerCorrectness, Faithfulness

from hybrid_rag.eval.judge import build_judge_embeddings, build_judge_llm


class RagasScorer:
    """Faithfulness + answer correctness for one answered question."""

    def __init__(self, faithfulness=None, correctness=None) -> None:
        if faithfulness is None or correctness is None:
            llm = build_judge_llm()
            faithfulness = faithfulness or Faithfulness(llm=llm)
            correctness = correctness or AnswerCorrectness(
                llm=llm, embeddings=build_judge_embeddings()
            )
        self._faithfulness = faithfulness
        self._correctness = correctness

    async def score(
        self, question: str, answer: str, contexts: list[str], reference: str
    ) -> dict[str, float]:
        """Grade one answer; keys merge into the question's ``scores``."""
        faith = await self._faithfulness.ascore(
            user_input=question, response=answer, retrieved_contexts=contexts
        )
        corr = await self._correctness.ascore(
            user_input=question, response=answer, reference=reference
        )
        return {"faithfulness": faith.value, "answer_correctness": corr.value}
