"""
LLM-as-Judge Holistic Assessment.

Provides a **cross-validation** layer by having the LLM rate the generated
reply on five dimensions:

- **Correctness** (1-5)
- **Completeness** (1-5)
- **Professionalism** (1-5)
- **Helpfulness** (1-5)
- **Naturalness** (1-5)

The average of all dimensions is normalised to [0, 1] and reported alongside
per-dimension reasoning.  This score is used as an independent sanity check
against the specialised metrics.
"""

from __future__ import annotations

import logging
from typing import Any

from evaluation.metrics import BaseMetric, MetricResult, clamp, safe_json_parse

logger = logging.getLogger(__name__)

_DIMENSIONS = (
    "correctness",
    "completeness",
    "professionalism",
    "helpfulness",
    "naturalness",
)

# ---------------------------------------------------------------------------
# Prompt template
# ---------------------------------------------------------------------------

_LLM_JUDGE_PROMPT = """\
You are an expert email quality evaluator. Rate the generated reply to the \
email below on five dimensions using a 1–5 integer scale.

Dimensions:
- **correctness**: Is the information accurate and non-contradictory? (1=wrong, 5=perfect)
- **completeness**: Does it address all questions/requests? (1=missing everything, 5=fully complete)
- **professionalism**: Is the tone and language appropriate for business? (1=unprofessional, 5=exemplary)
- **helpfulness**: Does it actually help the sender? (1=useless, 5=extremely helpful)
- **naturalness**: Does it read like a human wrote it? (1=robotic, 5=indistinguishable from human)

For each dimension, provide a brief **reasoning** explaining the rating.

Return ONLY a JSON object with this exact schema (no markdown fences):
{{
  "ratings": {{
    "correctness": {{"score": <int 1-5>, "reasoning": "<explanation>"}},
    "completeness": {{"score": <int 1-5>, "reasoning": "<explanation>"}},
    "professionalism": {{"score": <int 1-5>, "reasoning": "<explanation>"}},
    "helpfulness": {{"score": <int 1-5>, "reasoning": "<explanation>"}},
    "naturalness": {{"score": <int 1-5>, "reasoning": "<explanation>"}}
  }},
  "overall_comment": "<brief holistic comment>"
}}

--- ORIGINAL EMAIL ---
{email}

--- REFERENCE REPLY ---
{reference}

--- GENERATED REPLY ---
{generated}

Return ONLY the JSON object.
"""


class LLMJudgeMetric(BaseMetric):
    """Holistic LLM-as-Judge evaluation across five quality dimensions.

    Attributes:
        name: ``"llm_judge"``
    """

    name: str = "llm_judge"

    def __init__(self) -> None:
        from config import get_settings
        from generator.llm import CerebrasLLM

        settings = get_settings()
        self._llm = CerebrasLLM(
            model=settings.eval_llm_model,
            temperature=settings.eval_llm_temperature,
        )
        logger.info("LLMJudgeMetric initialised.")

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
        """Get holistic quality ratings from the LLM judge.

        Args:
            generated: AI-generated reply text.
            reference: Ground-truth reply text.
            email: Original incoming email.
            context: Retrieved context (unused in prompt but available).

        Returns:
            ``MetricResult`` with per-dimension ratings and normalised average.
        """
        if not generated.strip():
            return MetricResult(
                name=self.name,
                score=0.0,
                reasoning="Empty generated reply – cannot evaluate.",
            )

        prompt = _LLM_JUDGE_PROMPT.format(
            email=email,
            reference=reference,
            generated=generated,
        )
        messages = [{"role": "user", "content": prompt}]
        raw_response = self._llm.generate(messages)
        parsed = safe_json_parse(raw_response)

        if parsed is None or "ratings" not in parsed:
            logger.warning("LLMJudge: LLM returned unparseable response.")
            return MetricResult(
                name=self.name,
                score=0.5,
                reasoning="Could not parse LLM judge response; returning neutral score.",
                details={"raw_llm_response": raw_response[:500]},
            )

        ratings_raw: dict[str, Any] = parsed["ratings"]
        overall_comment: str = parsed.get("overall_comment", "")

        # Extract scores and reasoning per dimension
        dimension_scores: dict[str, int] = {}
        dimension_reasoning: dict[str, str] = {}
        for dim in _DIMENSIONS:
            dim_data = ratings_raw.get(dim, {})
            if isinstance(dim_data, dict):
                raw_score = dim_data.get("score", 3)
                reasoning = dim_data.get("reasoning", "")
            else:
                raw_score = dim_data
                reasoning = ""
            dimension_scores[dim] = self._safe_rating(raw_score)
            dimension_reasoning[dim] = reasoning

        # Average normalised to [0, 1] (ratings are 1-5)
        avg_rating = sum(dimension_scores.values()) / len(dimension_scores)
        normalised_score = clamp((avg_rating - 1.0) / 4.0)

        reasoning_lines: list[str] = [
            f"LLM Judge average={avg_rating:.2f}/5 "
            f"(normalised={normalised_score:.2f}).",
        ]
        for dim in _DIMENSIONS:
            reasoning_lines.append(
                f"  {dim.capitalize()}: {dimension_scores[dim]}/5 – "
                f"{dimension_reasoning[dim]}"
            )
        if overall_comment:
            reasoning_lines.append(f"Overall: {overall_comment}")

        return MetricResult(
            name=self.name,
            score=round(normalised_score, 4),
            details={
                "dimension_scores": dimension_scores,
                "dimension_reasoning": dimension_reasoning,
                "average_rating": round(avg_rating, 2),
                "overall_comment": overall_comment,
            },
            reasoning="\n".join(reasoning_lines),
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
