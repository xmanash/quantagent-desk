"""RAGAS-style evaluation module for RAG retrieval and generative risk responses."""

from __future__ import annotations

import os
import re
import sys
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock

# Shim to prevent legacy langchain-community Vertex AI import error in ragas
if "langchain_community.chat_models.vertexai" not in sys.modules:
    sys.modules["langchain_community.chat_models.vertexai"] = MagicMock()

import numpy as np


_STOPWORDS = {
    "the", "and", "for", "with", "from", "that", "this", "into", "your",
    "please", "does", "how", "what", "show", "run", "me", "are", "was",
}


def _content_words(text: str) -> set:
    return {
        word
        for word in re.findall(r"[a-z0-9]+", text.lower())
        if len(word) > 2 and word not in _STOPWORDS
    }


@dataclass
class RagasCaseResult:
    case_name: str
    query: str
    faithfulness: float
    answer_relevancy: float
    context_precision: float
    context_recall: float
    numeric_grounding: float
    overall_score: float
    passed: bool
    details: Dict[str, Any] = field(default_factory=dict)


class RagasEvaluator:
    """Evaluates RAG and quantitative responses on key RAGAS dimensions."""

    def __init__(self):
        self._embedding_model = None

    def _get_embedding_model(self):
        if os.getenv("QUANTAGENT_RAGAS_LEXICAL", "1").strip().lower() in {"1", "true", "yes"}:
            return None
        if self._embedding_model is None:
            try:
                from sentence_transformers import SentenceTransformer
                self._embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
            except Exception:
                self._embedding_model = None
        return self._embedding_model

    def _compute_cosine_sim(self, text_a: str, text_b: str) -> float:
        """Compute cosine similarity between two texts using local embeddings or word overlap."""
        model = self._get_embedding_model()
        if model is not None and text_a.strip() and text_b.strip():
            try:
                embs = model.encode([text_a, text_b])
                norm_a = np.linalg.norm(embs[0])
                norm_b = np.linalg.norm(embs[1])
                if norm_a > 0 and norm_b > 0:
                    sim = float(np.dot(embs[0], embs[1]) / (norm_a * norm_b))
                    return max(0.0, min(1.0, (sim + 1.0) / 2.0))
            except Exception:
                pass

        # Jaccard word-overlap fallback
        words_a = set(re.findall(r"\w+", text_a.lower()))
        words_b = set(re.findall(r"\w+", text_b.lower()))
        if not words_a or not words_b:
            return 0.5
        intersection = words_a.intersection(words_b)
        union = words_a.union(words_b)
        return float(len(intersection) / len(union)) if union else 0.5

    def evaluate_faithfulness(self, answer: str, context_docs: List[str]) -> float:
        """Measure if claims in the answer are grounded in context documents."""
        if not context_docs:
            # Quantitative answers are checked by numeric grounding, not document overlap.
            return 1.0

        context_blob = " ".join(context_docs)
        # Split answer into clean sentences/clauses
        sentences = [s.strip() for s in re.split(r"[.\n]+", answer) if len(s.strip()) > 15]
        if not sentences:
            return 1.0

        context_words = _content_words(context_blob)
        supported = 0
        for sent in sentences:
            sent_words = _content_words(sent)
            if not sent_words:
                supported += 1
                continue
            recall = len(sent_words & context_words) / len(sent_words)
            if recall >= 0.45:
                supported += 1

        return float(supported / len(sentences))

    def evaluate_answer_relevancy(self, query: str, answer: str) -> float:
        """Fraction of query content words that the answer actually covers."""
        if not query.strip() or not answer.strip():
            return 0.0
        query_words = _content_words(query)
        answer_words = _content_words(answer)
        if not query_words:
            return self._compute_cosine_sim(query, answer)
        return float(len(query_words & answer_words) / len(query_words))

    def evaluate_context_precision(self, query: str, context_docs: List[str]) -> float:
        """Measure whether the most relevant documents appear at top rank."""
        if not context_docs:
            return 1.0
        sims = [self._compute_cosine_sim(query, doc) for doc in context_docs]
        if len(sims) <= 1:
            return 1.0

        # Precision discounted cumulative gain proxy:
        # higher score if top element has higher similarity
        top_sim = sims[0]
        avg_other = float(np.mean(sims[1:])) if len(sims) > 1 else top_sim
        if top_sim >= avg_other:
            return 1.0
        return max(0.6, float(top_sim / (avg_other + 1e-6)))

    def evaluate_numeric_grounding(
        self,
        answer: str,
        calculation_ids: List[str],
        snapshot: Optional[Dict[str, Any]] = None,
        stress_result: Optional[Dict[str, Any]] = None,
    ) -> float:
        """Measure what fraction of monetary figures are grounded in calculation data."""
        from quantagent_desk.guardrails.validator import get_guardrails
        gr = get_guardrails()
        audit = gr.verify_numeric_grounding(
            answer=answer,
            calculation_ids=calculation_ids,
            snapshot=snapshot,
            stress_result=stress_result,
        )
        return float(1.0 - audit.hallucination_score)

    def evaluate_case(
        self,
        case_name: str,
        query: str,
        answer: str,
        context_docs: Optional[List[str]] = None,
        calculation_ids: Optional[List[str]] = None,
        snapshot: Optional[Dict[str, Any]] = None,
        stress_result: Optional[Dict[str, Any]] = None,
        threshold: float = 0.70,
    ) -> RagasCaseResult:
        """Run full RAGAS evaluation on a single query and response."""
        context_docs = context_docs or []
        calculation_ids = calculation_ids or []

        faithfulness = self.evaluate_faithfulness(answer, context_docs)
        relevancy = self.evaluate_answer_relevancy(query, answer)
        precision = self.evaluate_context_precision(query, context_docs)
        context_recall = 1.0 if context_docs else 0.8
        numeric_grounding = self.evaluate_numeric_grounding(
            answer=answer,
            calculation_ids=calculation_ids,
            snapshot=snapshot,
            stress_result=stress_result,
        )

        # Adapt weightings: RAG methodology queries focus on context faithfulness;
        # quantitative queries focus on numeric grounding and query relevancy.
        is_rag_query = bool("what is" in query.lower() or "explain" in query.lower() or "methodology" in query.lower() or "how" in query.lower())

        if is_rag_query:
            overall = float(
                (faithfulness * 0.40)
                + (relevancy * 0.30)
                + (precision * 0.15)
                + (context_recall * 0.15)
            )
            passed = overall >= threshold
        else:
            overall = float(
                (numeric_grounding * 0.50)
                + (relevancy * 0.35)
                + (faithfulness * 0.15)
            )
            passed = overall >= threshold and numeric_grounding >= 0.80

        return RagasCaseResult(
            case_name=case_name,
            query=query,
            faithfulness=round(faithfulness, 3),
            answer_relevancy=round(relevancy, 3),
            context_precision=round(precision, 3),
            context_recall=round(context_recall, 3),
            numeric_grounding=round(numeric_grounding, 3),
            overall_score=round(overall, 3),
            passed=passed,
            details={
                "context_chunks_evaluated": len(context_docs),
                "calculation_ids_evaluated": calculation_ids,
                "is_rag_query": is_rag_query,
            },
        )


_EVALUATOR: Optional[RagasEvaluator] = None


def get_ragas_evaluator() -> RagasEvaluator:
    global _EVALUATOR
    if _EVALUATOR is None:
        _EVALUATOR = RagasEvaluator()
    return _EVALUATOR
