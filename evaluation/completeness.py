"""
Completeness Metric (Weight: 20%).

Uses an LLM judge to assess whether every question, request, or action item
in the incoming email has been **fully**, **partially**, or **not at all**
answered in the generated reply.

Score = proportion of fully addressed items.  Partial answers receive 0.5
credit and deferred answers receive 0.25 credit.
"""

from __future__ import annotations

import logging
from typing import Any

from evaluation.metrics import BaseMetric, MetricResult, clamp, safe_json_parse

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Prompt template
# ---------------------------------------------------------------------------

_COMPLETENESS_PROMPT = """\
You are an expert email reviewer. Analyse the incoming email and the reply below.

1. List every **question** or **request** in the incoming email.
2. For each, classify the reply's coverage as one of:
   - "fully_addressed": The reply directly and completely answers it.
   - "partially_addressed": The reply mentions it but the answer is incomplete.
   - "deferred": The reply acknowledges it but defers to a later time or another person.
   - "missing": The reply does not address it at all.
3. Provide a brief **explanation** for each classification.

Return ONLY a JSON object with this exact schema (no markdown fences):
{{
  "questions": [
    {{
      "question": "<short description>",
      "status": "fully_addressed" | "partially_addressed" | "deferred" | "missing",
      "explanation": "<why you classified it this way>"
    }}
  ]
}}

--- INCOMING EMAIL ---
{email}

--- GENERATED REPLY ---
{generated}

Return ONLY the JSON object.
"""

# Credit weights for each status
_STATUS_CREDIT: dict[str, float] = {
    "fully_addressed": 1.0,
    "partially_addressed": 0.5,
    "deferred": 0.25,
    "missing": 0.0,
}


class CompletenessMetric(BaseMetric):
    """Evaluates whether all questions / requests are answered.

    Attributes:
        name: ``"completeness"``
    """

    name: str = "completeness"

    def __init__(self) -> None:
        from config import get_settings
        from generator.llm import CerebrasLLM

        settings = get_settings()
        self._llm = CerebrasLLM(
            model=settings.eval_llm_model,
            temperature=settings.eval_llm_temperature,
        )
        logger.info("CompletenessMetric initialised.")

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
        """Judge completeness of *generated* reply to *email*.

        Args:
            generated: AI-generated reply text.
            reference: Ground-truth reply (not used by prompt but kept for
                API consistency).
            email: Original incoming email.
            context: Retrieved context snippets (unused).

        Returns:
            ``MetricResult`` with per-question status breakdown.
        """
        if not email.strip():
            return MetricResult(
                name=self.name,
                score=1.0,
                reasoning="Empty email – nothing to address.",
            )
        if not generated.strip():
            return MetricResult(
                name=self.name,
                score=0.0,
                reasoning="Empty reply – nothing is addressed.",
            )

        prompt = _COMPLETENESS_PROMPT.format(email=email, generated=generated)
        messages = [{"role": "user", "content": prompt}]
        raw_response = self._llm.generate(messages)
        parsed = safe_json_parse(raw_response)

        if parsed is None or "questions" not in parsed:
            logger.warning(
                "Completeness: LLM returned unparseable response. "
                "Returning neutral score."
            )
            return MetricResult(
                name=self.name,
                score=0.5,
                reasoning="Could not parse LLM completeness analysis; returning neutral score.",
                details={"raw_llm_response": raw_response[:500]},
            )

        questions: list[dict[str, Any]] = parsed["questions"]
        if not questions:
            return MetricResult(
                name=self.name,
                score=1.0,
                reasoning="No questions or requests detected – trivially complete.",
                details={"questions": []},
            )

        # Compute weighted score
        total_credit = sum(
            _STATUS_CREDIT.get(q.get("status", "missing"), 0.0)
            for q in questions
        )
        score = clamp(total_credit / len(questions))

        # Build status summary
        status_counts: dict[str, int] = {}
        for q in questions:
            status = q.get("status", "missing")
            status_counts[status] = status_counts.get(status, 0) + 1

        reasoning_parts: list[str] = [
            f"Identified {len(questions)} question(s)/request(s). "
            f"Status breakdown: {status_counts}. Score={score:.2f}.",
        ]
        for q in questions:
            if q.get("status") in ("missing", "partially_addressed", "deferred"):
                reasoning_parts.append(
                    f"  • {q.get('status', '?').upper()}: "
                    f"\"{q.get('question', '?')}\" — {q.get('explanation', '')}"
                )

        return MetricResult(
            name=self.name,
            score=round(score, 4),
            details={
                "total_questions": len(questions),
                "status_counts": status_counts,
                "questions": questions,
            },
            reasoning="\n".join(reasoning_parts),
        )
