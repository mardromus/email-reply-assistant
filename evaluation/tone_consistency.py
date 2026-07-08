"""
Tone Consistency Metric (Weight: 10%).

Uses an LLM judge to rate both the generated reply and the reference reply
on three tone dimensions:

- **Formality** (1-5)
- **Professionalism** (1-5)
- **Empathy** (1-5)

The score measures how closely the generated reply's tone matches the
reference reply's tone.  A large mismatch (e.g., casual when the reference
is formal) incurs a penalty.
"""

from __future__ import annotations

import logging
from typing import Any

from evaluation.metrics import BaseMetric, MetricResult, clamp, safe_json_parse

logger = logging.getLogger(__name__)

# Maximum possible distance per dimension (rating range 1-5)
_MAX_DISTANCE_PER_DIM: float = 4.0
_TONE_DIMENSIONS = ("formality", "professionalism", "empathy")

# ---------------------------------------------------------------------------
# Prompt template
# ---------------------------------------------------------------------------

_TONE_PROMPT = """\
You are a tone analysis expert. Rate each of the following email texts on \
three dimensions using a 1–5 integer scale.

Dimensions:
- **formality**: 1 = very casual/slang, 5 = very formal/corporate
- **professionalism**: 1 = unprofessional, 5 = highly professional
- **empathy**: 1 = cold/dismissive, 5 = very empathetic/caring

Return ONLY a JSON object with this exact schema (no markdown fences):
{{
  "generated": {{
    "formality": <int 1-5>,
    "professionalism": <int 1-5>,
    "empathy": <int 1-5>,
    "reasoning": "<brief justification for ratings>"
  }},
  "reference": {{
    "formality": <int 1-5>,
    "professionalism": <int 1-5>,
    "empathy": <int 1-5>,
    "reasoning": "<brief justification for ratings>"
  }}
}}

--- GENERATED REPLY ---
{generated}

--- REFERENCE REPLY ---
{reference}

Return ONLY the JSON object.
"""


class ToneConsistencyMetric(BaseMetric):
    """Evaluates whether the generated reply matches the reference's tone.

    Attributes:
        name: ``"tone_consistency"``
    """

    name: str = "tone_consistency"

    def __init__(self) -> None:
        from config import get_settings
        from generator.llm import CerebrasLLM

        settings = get_settings()
        self._llm = CerebrasLLM(
            model=settings.eval_llm_model,
            temperature=settings.eval_llm_temperature,
        )
        logger.info("ToneConsistencyMetric initialised.")

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
        """Rate tone dimensions for *generated* and *reference*, then compute match.

        Args:
            generated: AI-generated reply text.
            reference: Ground-truth reply text.
            email: Original incoming email (unused).
            context: Retrieved context snippets (unused).

        Returns:
            ``MetricResult`` with per-dimension ratings and distance info.
        """
        if not generated.strip() or not reference.strip():
            return MetricResult(
                name=self.name,
                score=0.0,
                reasoning="Empty text – tone cannot be assessed.",
            )

        prompt = _TONE_PROMPT.format(generated=generated, reference=reference)
        messages = [{"role": "user", "content": prompt}]
        raw_response = self._llm.generate(messages)
        parsed = safe_json_parse(raw_response)

        if (
            parsed is None
            or "generated" not in parsed
            or "reference" not in parsed
        ):
            logger.warning(
                "ToneConsistency: LLM returned unparseable response."
            )
            return MetricResult(
                name=self.name,
                score=0.5,
                reasoning="Could not parse LLM tone analysis; returning neutral score.",
                details={"raw_llm_response": raw_response[:500]},
            )

        gen_ratings: dict[str, Any] = parsed["generated"]
        ref_ratings: dict[str, Any] = parsed["reference"]

        # Compute per-dimension distance and aggregate
        distances: dict[str, float] = {}
        total_distance = 0.0
        for dim in _TONE_DIMENSIONS:
            g_val = self._safe_rating(gen_ratings.get(dim, 3))
            r_val = self._safe_rating(ref_ratings.get(dim, 3))
            dist = abs(g_val - r_val)
            distances[dim] = dist
            total_distance += dist

        max_total_distance = _MAX_DISTANCE_PER_DIM * len(_TONE_DIMENSIONS)
        score = clamp(1.0 - (total_distance / max_total_distance))

        # Penalty flag if any single dimension differs by ≥ 3 points
        penalized = any(d >= 3.0 for d in distances.values())
        penalty_reason = ""
        if penalized:
            bad_dims = [d for d, v in distances.items() if v >= 3.0]
            penalty_reason = (
                f"Large tone mismatch on: {', '.join(bad_dims)}."
            )

        reasoning = (
            f"Generated ratings: {gen_ratings}. "
            f"Reference ratings: {ref_ratings}. "
            f"Per-dimension distance: {distances}. "
            f"Aggregate score={score:.2f}."
        )
        if penalized:
            reasoning += f" PENALTY: {penalty_reason}"

        return MetricResult(
            name=self.name,
            score=round(score, 4),
            details={
                "generated_ratings": gen_ratings,
                "reference_ratings": ref_ratings,
                "distances": distances,
                "total_distance": total_distance,
            },
            reasoning=reasoning,
            penalized=penalized,
            penalty_reason=penalty_reason,
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _safe_rating(value: Any) -> int:
        """Coerce a value to an integer in [1, 5]."""
        try:
            v = int(value)
        except (TypeError, ValueError):
            v = 3
        return max(1, min(5, v))
