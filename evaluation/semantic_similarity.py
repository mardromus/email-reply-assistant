"""
Semantic Similarity Metric (Weight: 20%).

Computes a combined similarity score between the generated and reference
email replies using:

1. **Cosine similarity** – via sentence-transformers embeddings.
2. **BERTScore F1** – contextualised token-level similarity.

Combined score = 0.6 × cosine_sim + 0.4 × bertscore_f1

The metric leverages the project's shared ``EmbeddingModel`` for embedding
consistency and falls back gracefully if the ``bert-score`` library is
unavailable.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

from evaluation.metrics import BaseMetric, MetricResult, clamp

logger = logging.getLogger(__name__)

# Weights for the composite score
_COSINE_WEIGHT: float = 0.60
_BERTSCORE_WEIGHT: float = 0.40


class SemanticSimilarityMetric(BaseMetric):
    """Measures semantic similarity between generated and reference replies.

    Attributes:
        name: ``"semantic_similarity"``
    """

    name: str = "semantic_similarity"

    def __init__(self) -> None:
        from retrieval.embedding import get_embedding_model

        self._embedding_model = get_embedding_model()
        logger.info("SemanticSimilarityMetric initialised.")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def evaluate(
        self,
        generated: str,
        reference: str,
        email: str,
        context: list[str] | None = None,
    ) -> MetricResult:
        """Compute cosine similarity and BERTScore between *generated* and *reference*.

        Args:
            generated: AI-generated reply text.
            reference: Ground-truth reply text.
            email: Original incoming email (unused by this metric).
            context: Retrieved snippets (unused by this metric).

        Returns:
            ``MetricResult`` with combined score and breakdown in ``details``.
        """
        if not generated.strip() or not reference.strip():
            return MetricResult(
                name=self.name,
                score=0.0,
                reasoning="Empty generated or reference text – similarity is 0.",
                details={"cosine_sim": 0.0, "bertscore_f1": 0.0},
            )

        cosine_sim = self._compute_cosine_similarity(generated, reference)
        bertscore_metrics = self._compute_bertscore(generated, reference)

        bertscore_f1 = bertscore_metrics.get("f1", cosine_sim)  # fallback

        combined = clamp(
            _COSINE_WEIGHT * cosine_sim + _BERTSCORE_WEIGHT * bertscore_f1
        )

        reasoning = self._build_reasoning(cosine_sim, bertscore_f1, combined)

        return MetricResult(
            name=self.name,
            score=round(combined, 4),
            details={
                "cosine_sim": round(cosine_sim, 4),
                "bertscore_precision": round(bertscore_metrics.get("precision", 0.0), 4),
                "bertscore_recall": round(bertscore_metrics.get("recall", 0.0), 4),
                "bertscore_f1": round(bertscore_f1, 4),
                "cosine_weight": _COSINE_WEIGHT,
                "bertscore_weight": _BERTSCORE_WEIGHT,
            },
            reasoning=reasoning,
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _compute_cosine_similarity(self, text_a: str, text_b: str) -> float:
        """Embed both texts and return their cosine similarity."""
        try:
            embeddings = self._embedding_model.encode([text_a, text_b])
            vec_a, vec_b = embeddings[0], embeddings[1]
            dot = float(np.dot(vec_a, vec_b))
            norm_a = float(np.linalg.norm(vec_a))
            norm_b = float(np.linalg.norm(vec_b))
            if norm_a == 0 or norm_b == 0:
                return 0.0
            return clamp(dot / (norm_a * norm_b))
        except Exception as exc:
            logger.warning("Cosine similarity computation failed: %s", exc)
            return 0.0

    @staticmethod
    def _compute_bertscore(generated: str, reference: str) -> dict[str, float]:
        """Compute BERTScore (precision, recall, F1).

        Falls back to empty dict if the ``bert_score`` package is missing.
        """
        try:
            from bert_score import score as bert_score_fn

            precision, recall, f1 = bert_score_fn(
                [generated],
                [reference],
                lang="en",
                verbose=False,
                rescale_with_baseline=True,
            )
            return {
                "precision": clamp(float(precision[0])),
                "recall": clamp(float(recall[0])),
                "f1": clamp(float(f1[0])),
            }
        except ImportError:
            logger.warning(
                "bert-score not installed – BERTScore will be approximated by cosine sim."
            )
            return {}
        except Exception as exc:
            logger.warning("BERTScore computation failed: %s", exc)
            return {}

    @staticmethod
    def _build_reasoning(
        cosine_sim: float,
        bertscore_f1: float,
        combined: float,
    ) -> str:
        """Generate human-readable explanation of the similarity scores."""
        level = (
            "very high"
            if combined >= 0.85
            else "high"
            if combined >= 0.70
            else "moderate"
            if combined >= 0.50
            else "low"
            if combined >= 0.30
            else "very low"
        )
        return (
            f"Semantic similarity is {level} (combined={combined:.2f}). "
            f"Cosine similarity={cosine_sim:.2f} captures overall meaning overlap; "
            f"BERTScore F1={bertscore_f1:.2f} measures contextualised token alignment. "
            f"Weights: cosine={_COSINE_WEIGHT}, BERTScore={_BERTSCORE_WEIGHT}."
        )
