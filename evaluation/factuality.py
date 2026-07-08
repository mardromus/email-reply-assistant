"""
Factual Consistency Metric.

Checks whether the generated reply **contradicts** information in:
- The original incoming email
- The retrieved context examples

Uses an NLI-style (Natural Language Inference) contradiction detection
approach via LLM.  Each identified contradiction reduces the score.

Score:  1.0 = fully consistent
        0.0 = contradicts everything
"""

from __future__ import annotations

import logging
from typing import Any

from evaluation.metrics import BaseMetric, MetricResult, clamp, safe_json_parse

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Prompt template
# ---------------------------------------------------------------------------

_FACTUALITY_PROMPT = """\
You are a consistency checker. Compare the generated reply against the \
original email and retrieved context. Identify any **contradictions** — \
places where the generated reply states something that directly conflicts \
with information in the source materials.

For each contradiction found:
- Quote the conflicting statement from the generated reply
- Quote or describe the contradicted information from the sources
- Rate severity: "major" (core fact wrong) or "minor" (peripheral detail wrong)

Return ONLY a JSON object with this exact schema (no markdown fences):
{{
  "contradictions": [
    {{
      "generated_claim": "<statement from the generated reply>",
      "source_fact": "<the conflicting fact from email or context>",
      "severity": "major" | "minor",
      "explanation": "<why these contradict each other>"
    }}
  ],
  "consistency_assessment": "<brief summary>"
}}

If there are NO contradictions, return:
{{"contradictions": [], "consistency_assessment": "Fully consistent with sources."}}

--- ORIGINAL EMAIL ---
{email}

--- RETRIEVED CONTEXT ---
{context}

--- GENERATED REPLY ---
{generated}

Return ONLY the JSON object.
"""

_SEVERITY_PENALTY: dict[str, float] = {
    "major": 0.35,
    "minor": 0.15,
}


class FactualityMetric(BaseMetric):
    """Detects contradictions between the generated reply and source material.

    Attributes:
        name: ``"factuality"``
    """

    name: str = "factuality"

    def __init__(self) -> None:
        from config import get_settings
        from generator.llm import CerebrasLLM

        settings = get_settings()
        self._llm = CerebrasLLM(
            model=settings.eval_llm_model,
            temperature=settings.eval_llm_temperature,
        )
        logger.info("FactualityMetric initialised.")

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
        """Check *generated* for factual contradictions against sources.

        Args:
            generated: AI-generated reply text.
            reference: Ground-truth reply (available in details for analysis).
            email: Original incoming email.
            context: Retrieved context snippets.

        Returns:
            ``MetricResult`` with contradiction list and consistency assessment.
        """
        if not generated.strip():
            return MetricResult(
                name=self.name,
                score=1.0,
                reasoning="Empty generated text – no contradictions possible.",
            )

        context_text = (
            "\n---\n".join(context) if context else "(no context provided)"
        )
        prompt = _FACTUALITY_PROMPT.format(
            email=email,
            context=context_text,
            generated=generated,
        )

        messages = [{"role": "user", "content": prompt}]
        raw_response = self._llm.generate(messages)
        parsed = safe_json_parse(raw_response)

        if parsed is None or "contradictions" not in parsed:
            logger.warning("Factuality: LLM returned unparseable response.")
            return MetricResult(
                name=self.name,
                score=0.5,
                reasoning="Could not parse LLM factuality analysis; returning neutral score.",
                details={"raw_llm_response": raw_response[:500]},
            )

        contradictions: list[dict[str, Any]] = parsed["contradictions"]
        assessment: str = parsed.get(
            "consistency_assessment", "No assessment provided."
        )

        if not contradictions:
            return MetricResult(
                name=self.name,
                score=1.0,
                reasoning=f"No contradictions found. {assessment}",
                details={
                    "contradiction_count": 0,
                    "contradictions": [],
                    "consistency_assessment": assessment,
                },
            )

        # Compute penalty
        total_penalty = 0.0
        severity_counts: dict[str, int] = {"major": 0, "minor": 0}
        for c in contradictions:
            severity = c.get("severity", "minor")
            severity_counts[severity] = severity_counts.get(severity, 0) + 1
            total_penalty += _SEVERITY_PENALTY.get(severity, 0.15)

        score = clamp(1.0 - total_penalty)

        reasoning_parts: list[str] = [
            f"Found {len(contradictions)} contradiction(s). "
            f"Severity: {severity_counts}. Score={score:.2f}.",
        ]
        for c in contradictions:
            reasoning_parts.append(
                f"  • [{c.get('severity', '?').upper()}] "
                f"Reply says: \"{c.get('generated_claim', '?')}\" "
                f"but source says: \"{c.get('source_fact', '?')}\" — "
                f"{c.get('explanation', '')}"
            )
        reasoning_parts.append(f"Assessment: {assessment}")

        return MetricResult(
            name=self.name,
            score=round(score, 4),
            details={
                "contradiction_count": len(contradictions),
                "severity_counts": severity_counts,
                "contradictions": contradictions,
                "consistency_assessment": assessment,
            },
            reasoning="\n".join(reasoning_parts),
            penalized=score < 1.0,
            penalty_reason=(
                f"{len(contradictions)} contradiction(s) detected."
                if contradictions
                else ""
            ),
        )
