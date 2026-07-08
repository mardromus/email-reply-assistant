"""
Intent Coverage Metric (Weight: 25%).

Uses an LLM to extract **intents** from the incoming email (questions,
requests, complaints, etc.) and then checks whether each intent is
adequately addressed in the generated reply.

Score = proportion of intents covered.

The LLM is prompted to return structured JSON so that the metric can
provide a per-intent breakdown with evidence for coverage or gaps.
"""

from __future__ import annotations

import logging
from typing import Any

from evaluation.metrics import BaseMetric, MetricResult, clamp, safe_json_parse

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Prompt templates
# ---------------------------------------------------------------------------

_EXTRACT_AND_CHECK_PROMPT = """\
You are an expert email analyst. Given an incoming email and a reply to that email, do the following:

1. Identify every distinct **intent** in the incoming email. An intent is a question, request, complaint, piece of feedback, or topic that the sender expects the replier to address.
2. For each intent, determine whether it is **covered** (fully addressed) in the reply. Provide a brief **evidence** quote or explanation.

Return your analysis as JSON with this exact schema (no markdown fences):
{{
  "intents": [
    {{
      "intent": "<short description of the intent>",
      "covered": true | false,
      "evidence": "<quote or explanation from the reply, or reason it is missing>"
    }}
  ]
}}

--- INCOMING EMAIL ---
{email}

--- GENERATED REPLY ---
{generated}

Return ONLY the JSON object. Do not include any other text.
"""


class IntentCoverageMetric(BaseMetric):
    """Evaluates how many of the original email's intents are addressed.

    Uses the project's ``CerebrasLLM`` for intent extraction and coverage
    checking in a single prompt call to minimise latency.

    Attributes:
        name: ``"intent_coverage"``
    """

    name: str = "intent_coverage"

    def __init__(self) -> None:
        from config import get_settings
        from generator.llm import CerebrasLLM

        settings = get_settings()
        self._llm = CerebrasLLM(
            model=settings.eval_llm_model,
            temperature=settings.eval_llm_temperature,
        )
        logger.info("IntentCoverageMetric initialised.")

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
        """Extract intents from *email* and check coverage in *generated*.

        Args:
            generated: AI-generated reply text.
            reference: Ground-truth reply (unused directly, but available for
                cross-validation in ``details``).
            email: Original incoming email whose intents to extract.
            context: Retrieved context snippets (unused).

        Returns:
            ``MetricResult`` with coverage ratio and per-intent breakdown.
        """
        if not email.strip():
            return MetricResult(
                name=self.name,
                score=1.0,
                reasoning="Empty email – no intents to cover.",
            )
        if not generated.strip():
            return MetricResult(
                name=self.name,
                score=0.0,
                reasoning="Empty generated reply – no intents can be covered.",
            )

        prompt = _EXTRACT_AND_CHECK_PROMPT.format(
            email=email,
            generated=generated,
        )

        messages = [{"role": "user", "content": prompt}]
        raw_response = self._llm.generate(messages)
        parsed = safe_json_parse(raw_response)

        if parsed is None or "intents" not in parsed:
            logger.warning(
                "IntentCoverage: LLM returned unparseable response. "
                "Returning neutral score."
            )
            return MetricResult(
                name=self.name,
                score=0.5,
                reasoning="Could not parse LLM intent analysis; returning neutral score.",
                details={"raw_llm_response": raw_response[:500]},
            )

        intents: list[dict[str, Any]] = parsed["intents"]
        if not intents:
            return MetricResult(
                name=self.name,
                score=1.0,
                reasoning="LLM found no intents in the email – trivially covered.",
                details={"intents": []},
            )

        covered = [i for i in intents if i.get("covered", False)]
        uncovered = [i for i in intents if not i.get("covered", False)]
        score = clamp(len(covered) / len(intents))

        reasoning_parts: list[str] = [
            f"Identified {len(intents)} intent(s); {len(covered)} covered, "
            f"{len(uncovered)} uncovered (score={score:.2f})."
        ]
        for item in uncovered:
            reasoning_parts.append(
                f"  • MISSED: \"{item.get('intent', '?')}\" — "
                f"{item.get('evidence', 'no evidence provided')}"
            )

        return MetricResult(
            name=self.name,
            score=round(score, 4),
            details={
                "total_intents": len(intents),
                "covered_count": len(covered),
                "uncovered_count": len(uncovered),
                "intents": intents,
            },
            reasoning="\n".join(reasoning_parts),
        )
