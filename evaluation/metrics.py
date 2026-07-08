"""
Base classes and shared utilities for the evaluation framework.

Defines the canonical ``MetricResult`` data model and the ``BaseMetric``
abstract base class that every evaluation dimension must implement.

All metric implementations should inherit from ``BaseMetric`` and return a
``MetricResult`` from their ``evaluate`` method.

Example::

    class MyMetric(BaseMetric):
        name: str = "my_metric"

        def evaluate(self, generated, reference, email, context=None):
            score = ...
            return MetricResult(name=self.name, score=score, reasoning="...")
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Any

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------

class MetricResult(BaseModel):
    """Result returned by a single evaluation metric.

    Attributes:
        name: Human-readable metric name (e.g. ``"semantic_similarity"``).
        score: Normalised score in [0, 1]. Higher is better.
        details: Arbitrary structured data specific to the metric.
        reasoning: Free-text explanation of how the score was determined.
        penalized: Whether the score was reduced due to a penalty rule.
        penalty_reason: Explanation of any penalty applied.
    """

    name: str
    score: float = Field(
        ge=0.0,
        le=1.0,
        description="Normalised score between 0 (worst) and 1 (best).",
    )
    details: dict[str, Any] = Field(default_factory=dict)
    reasoning: str = ""
    penalized: bool = False
    penalty_reason: str = ""

    class Config:
        json_schema_extra = {
            "example": {
                "name": "semantic_similarity",
                "score": 0.85,
                "details": {"cosine_sim": 0.88, "bertscore_f1": 0.81},
                "reasoning": "High semantic overlap between generated and reference.",
                "penalized": False,
                "penalty_reason": "",
            }
        }


# ---------------------------------------------------------------------------
# Abstract base metric
# ---------------------------------------------------------------------------

class BaseMetric(ABC):
    """Abstract base class for all evaluation metrics.

    Subclasses **must** set the ``name`` class attribute and implement the
    ``evaluate`` method.

    The ``safe_evaluate`` wrapper catches any unexpected exception so that a
    single failing metric does not prevent other metrics from executing.
    """

    name: str = "base_metric"

    @abstractmethod
    def evaluate(
        self,
        generated: str,
        reference: str,
        email: str,
        context: list[str] | None = None,
    ) -> MetricResult:
        """Evaluate *generated* against *reference* for the given *email*.

        Args:
            generated: The AI-generated email reply.
            reference: The ground-truth / expected reply.
            email: The original incoming email being replied to.
            context: Optional list of retrieved context snippets used during
                generation.

        Returns:
            A ``MetricResult`` with a normalised score and supporting detail.
        """
        ...

    def safe_evaluate(
        self,
        generated: str,
        reference: str,
        email: str,
        context: list[str] | None = None,
    ) -> MetricResult:
        """Run ``evaluate`` with a safety net for graceful degradation.

        If the underlying ``evaluate`` call raises, a ``MetricResult`` with
        ``score=0.0`` and the error details is returned instead of propagating
        the exception.
        """
        try:
            return self.evaluate(generated, reference, email, context)
        except Exception as exc:
            logger.exception(
                "Metric '%s' failed – returning zero score.",
                self.name,
            )
            return MetricResult(
                name=self.name,
                score=0.0,
                reasoning=f"Metric evaluation failed: {exc}",
                details={"error": str(exc), "error_type": type(exc).__name__},
                penalized=True,
                penalty_reason="Metric raised an exception during evaluation.",
            )


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def clamp(value: float, lo: float = 0.0, hi: float = 1.0) -> float:
    """Clamp *value* to the closed interval [lo, hi]."""
    return max(lo, min(hi, value))


def safe_json_parse(text: str) -> dict[str, Any] | None:
    """Attempt to parse JSON from *text*, stripping markdown fences.

    Returns ``None`` on failure instead of raising.
    """
    import json
    import re

    # Strip markdown code fences (```json ... ```)
    cleaned = re.sub(r"```(?:json)?\s*", "", text).strip()
    cleaned = cleaned.rstrip("`").strip()

    try:
        return json.loads(cleaned)
    except (json.JSONDecodeError, ValueError) as exc:
        logger.warning("Failed to parse JSON from LLM response: %s", exc)
        return None
