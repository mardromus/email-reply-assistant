"""
Retrieval Grounding Metric (Weight: 15%).

Measures how well the generated response is **grounded** in the retrieved
context examples.  A high grounding score indicates the reply is faithful
to the information surfaced by the retrieval pipeline.

Two signals are combined:

1. **Embedding similarity** – max cosine similarity between the generated
   reply and each retrieved context snippet.
2. **Key-phrase overlap** – proportion of distinctive n-grams from the
   retrieved context that also appear in the generated reply.

Combined: 0.7 × max_embedding_sim + 0.3 × phrase_overlap
"""

from __future__ import annotations

import logging
import re
from collections import Counter
from typing import Any

import numpy as np

from evaluation.metrics import BaseMetric, MetricResult, clamp

logger = logging.getLogger(__name__)

_EMBEDDING_WEIGHT: float = 0.70
_PHRASE_WEIGHT: float = 0.30
_NGRAM_SIZE: int = 3  # trigrams for phrase overlap


class GroundingMetric(BaseMetric):
    """Evaluates faithfulness of the generated reply to retrieved context.

    Attributes:
        name: ``"grounding"``
    """

    name: str = "grounding"

    def __init__(self) -> None:
        from retrieval.embedding import get_embedding_model

        self._embedding_model = get_embedding_model()
        logger.info("GroundingMetric initialised.")

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
        """Compute grounding of *generated* in *context*.

        Args:
            generated: AI-generated reply text.
            reference: Ground-truth reply (unused directly).
            email: Original incoming email (unused directly).
            context: Retrieved context snippets to ground against.

        Returns:
            ``MetricResult`` with embedding similarity and phrase overlap.
        """
        if not context:
            return MetricResult(
                name=self.name,
                score=0.5,
                reasoning=(
                    "No retrieved context provided – grounding cannot be "
                    "measured. Returning neutral score."
                ),
                details={"context_count": 0},
            )

        if not generated.strip():
            return MetricResult(
                name=self.name,
                score=0.0,
                reasoning="Empty generated text – cannot measure grounding.",
            )

        max_sim, per_snippet_sims = self._max_embedding_similarity(
            generated, context
        )
        phrase_overlap = self._phrase_overlap(generated, context)

        combined = clamp(
            _EMBEDDING_WEIGHT * max_sim + _PHRASE_WEIGHT * phrase_overlap
        )

        confidence_level = (
            "strongly grounded"
            if combined >= 0.75
            else "moderately grounded"
            if combined >= 0.50
            else "weakly grounded"
            if combined >= 0.25
            else "not grounded"
        )

        reasoning = (
            f"Response is {confidence_level} in retrieved context "
            f"(combined={combined:.2f}). "
            f"Max embedding similarity={max_sim:.2f} across {len(context)} "
            f"snippet(s). Key-phrase overlap={phrase_overlap:.2f}."
        )

        return MetricResult(
            name=self.name,
            score=round(combined, 4),
            details={
                "max_embedding_similarity": round(max_sim, 4),
                "per_snippet_similarities": [
                    round(s, 4) for s in per_snippet_sims
                ],
                "phrase_overlap": round(phrase_overlap, 4),
                "context_count": len(context),
                "embedding_weight": _EMBEDDING_WEIGHT,
                "phrase_weight": _PHRASE_WEIGHT,
            },
            reasoning=reasoning,
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _max_embedding_similarity(
        self,
        generated: str,
        context: list[str],
    ) -> tuple[float, list[float]]:
        """Return (max_sim, per_snippet_sims) using embedding cosine sim."""
        try:
            texts = [generated] + context
            embeddings = self._embedding_model.encode(texts)
            gen_vec = embeddings[0]
            sims: list[float] = []
            for ctx_vec in embeddings[1:]:
                dot = float(np.dot(gen_vec, ctx_vec))
                norm_g = float(np.linalg.norm(gen_vec))
                norm_c = float(np.linalg.norm(ctx_vec))
                if norm_g == 0 or norm_c == 0:
                    sims.append(0.0)
                else:
                    sims.append(clamp(dot / (norm_g * norm_c)))
            return (max(sims) if sims else 0.0, sims)
        except Exception as exc:
            logger.warning("Embedding grounding computation failed: %s", exc)
            return 0.0, []

    @staticmethod
    def _phrase_overlap(generated: str, context: list[str]) -> float:
        """Compute trigram overlap between generated text and context."""
        def _ngrams(text: str, n: int) -> Counter[tuple[str, ...]]:
            tokens = re.findall(r"\w+", text.lower())
            return Counter(
                tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)
            )

        gen_ngrams = _ngrams(generated, _NGRAM_SIZE)
        if not gen_ngrams:
            return 0.0

        # Aggregate n-grams across all context snippets
        ctx_ngrams: Counter[tuple[str, ...]] = Counter()
        for snippet in context:
            ctx_ngrams.update(_ngrams(snippet, _NGRAM_SIZE))

        if not ctx_ngrams:
            return 0.0

        # Proportion of generated n-grams also found in context
        overlap_count = sum(
            min(gen_ngrams[ng], ctx_ngrams[ng])
            for ng in gen_ngrams
            if ng in ctx_ngrams
        )
        return clamp(overlap_count / sum(gen_ngrams.values()))
