"""
Hallucination Detection Metric (Weight: 10%).

Uses an LLM judge to detect **fabricated claims** in the generated reply
that cannot be supported by the original email or the retrieved context.

Checks for invented:
- dates, deadlines, or time references
- prices, amounts, or financial figures
- policies, terms, or guarantees
- promises or commitments
- reference numbers, ticket IDs, or identifiers

Score:  1.0  = no hallucination detected
        0.0  = severe hallucination(s) found

Returns a detailed list of each hallucinated claim with explanations and
a severity rating.
"""

from __future__ import annotations

import logging
from typing import Any

from evaluation.metrics import BaseMetric, MetricResult, clamp, safe_json_parse

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Prompt template
# ---------------------------------------------------------------------------

_HALLUCINATION_PROMPT = """\
You are a fact-checking expert. Examine the generated email reply below and \
determine whether it contains any **hallucinated** (fabricated) claims — \
information that is NOT supported by the original email or the retrieved \
context examples.

Look specifically for:
- Invented dates, deadlines, or time references
- Fabricated prices, amounts, or financial figures
- Made-up policies, terms, or guarantees
- Unsupported promises or commitments
- Invented reference numbers, ticket IDs, or identifiers
- Any other factual claim that cannot be verified from the provided sources

For each hallucination found, rate its severity:
- "high": Could cause real harm (wrong policy, wrong price, false promise)
- "medium": Misleading but unlikely to cause direct harm
- "low": Minor embellishment or assumption

Return ONLY a JSON object with this exact schema (no markdown fences):
{{
  "hallucinations": [
    {{
      "claim": "<the fabricated claim>",
      "severity": "high" | "medium" | "low",
      "explanation": "<why this is a hallucination>"
    }}
  ],
  "overall_assessment": "<brief summary of hallucination status>"
}}

If there are NO hallucinations, return:
{{"hallucinations": [], "overall_assessment": "No hallucinations detected."}}

--- ORIGINAL EMAIL ---
{email}

--- RETRIEVED CONTEXT ---
{context}

--- GENERATED REPLY ---
{generated}

Return ONLY the JSON object.
"""

# Severity to score penalty mapping
_SEVERITY_PENALTY: dict[str, float] = {
    "high": 0.40,
    "medium": 0.20,
    "low": 0.10,
}


class HallucinationMetric(BaseMetric):
    """Detects fabricated claims in the generated reply.

    Attributes:
        name: ``"hallucination"``
    """

    name: str = "hallucination"

    def __init__(self) -> None:
        from config import get_settings
        from generator.llm import CerebrasLLM

        settings = get_settings()
        self._llm = CerebrasLLM(
            model=settings.eval_llm_model,
            temperature=settings.eval_llm_temperature,
        )
        logger.info("HallucinationMetric initialised.")

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
        """Check *generated* for hallucinated claims.

        Args:
            generated: AI-generated reply text.
            reference: Ground-truth reply (available for cross-ref in details).
            email: Original incoming email.
            context: Retrieved context snippets.

        Returns:
            ``MetricResult`` with hallucination list and severity breakdown.
        """
        if not generated.strip():
            return MetricResult(
                name=self.name,
                score=1.0,
                reasoning="Empty generated text – no hallucinations possible.",
            )

        context_text = "\n---\n".join(context) if context else "(no context provided)"

        prompt = _HALLUCINATION_PROMPT.format(
            email=email,
            context=context_text,
            generated=generated,
        )
        messages = [{"role": "user", "content": prompt}]
        raw_response = self._llm.generate(messages)
        parsed = safe_json_parse(raw_response)

        if parsed is None or "hallucinations" not in parsed:
            logger.warning(
                "Hallucination: LLM returned unparseable response."
            )
            return MetricResult(
                name=self.name,
                score=0.5,
                reasoning="Could not parse LLM hallucination analysis; returning neutral score.",
                details={"raw_llm_response": raw_response[:500]},
            )

        hallucinations: list[dict[str, Any]] = parsed["hallucinations"]
        overall_assessment: str = parsed.get(
            "overall_assessment", "No assessment provided."
        )

        if not hallucinations:
            return MetricResult(
                name=self.name,
                score=1.0,
                reasoning=f"No hallucinations detected. {overall_assessment}",
                details={
                    "hallucination_count": 0,
                    "hallucinations": [],
                    "overall_assessment": overall_assessment,
                },
            )

        # Accumulate penalties
        total_penalty = 0.0
        severity_counts: dict[str, int] = {"high": 0, "medium": 0, "low": 0}
        for h in hallucinations:
            severity = h.get("severity", "medium")
            severity_counts[severity] = severity_counts.get(severity, 0) + 1
            total_penalty += _SEVERITY_PENALTY.get(severity, 0.15)

        score = clamp(1.0 - total_penalty)

        reasoning_parts: list[str] = [
            f"Found {len(hallucinations)} hallucination(s). "
            f"Severity breakdown: {severity_counts}. Score={score:.2f}.",
        ]
        for h in hallucinations:
            reasoning_parts.append(
                f"  • [{h.get('severity', '?').upper()}] "
                f"\"{h.get('claim', '?')}\" — {h.get('explanation', '')}"
            )
        reasoning_parts.append(f"Overall: {overall_assessment}")

        return MetricResult(
            name=self.name,
            score=round(score, 4),
            details={
                "hallucination_count": len(hallucinations),
                "severity_counts": severity_counts,
                "hallucinations": hallucinations,
                "overall_assessment": overall_assessment,
            },
            reasoning="\n".join(reasoning_parts),
            penalized=score < 1.0,
            penalty_reason=(
                f"{len(hallucinations)} hallucination(s) detected."
                if hallucinations
                else ""
            ),
        )
