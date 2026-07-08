"""
Business Rule Checks.

Validates the generated email reply against a set of structural and
content-based business rules:

1. **Response length** – must be 50–500 words.
2. **Greeting** – must contain an appropriate greeting.
3. **Sign-off** – must contain a closing sign-off.
4. **Appropriate language** – no profanity or inappropriate terms.
5. **Category-specific rules** – e.g. refund emails must acknowledge the
   request; complaint emails must show empathy.

Returns a pass/fail status per rule with an overall compliance score
(proportion of rules passed).
"""

from __future__ import annotations

import logging
import re
from typing import Any

from evaluation.metrics import BaseMetric, MetricResult, clamp

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Rule definitions
# ---------------------------------------------------------------------------

# Common greetings (case-insensitive)
_GREETING_PATTERNS: list[str] = [
    r"\bhi\b",
    r"\bhello\b",
    r"\bhey\b",
    r"\bdear\b",
    r"\bgood\s+(morning|afternoon|evening|day)\b",
    r"\bgreetings\b",
    r"\bthank\s+you\s+for\s+(your|contacting|reaching|writing)\b",
]

# Common sign-offs (case-insensitive)
_SIGNOFF_PATTERNS: list[str] = [
    r"\bbest\s+regards?\b",
    r"\bkind\s+regards?\b",
    r"\bsincerely\b",
    r"\bthank\s*you\b",
    r"\bthanks\b",
    r"\bwarm\s+regards?\b",
    r"\byours\s+(truly|faithfully)\b",
    r"\bcheers\b",
    r"\bregards\b",
    r"\bbest\b",
    r"\btake\s+care\b",
    r"\blooking\s+forward\b",
]

# Inappropriate terms (non-exhaustive, for basic filtering)
_INAPPROPRIATE_PATTERNS: list[str] = [
    r"\bdamn\b",
    r"\bhell\b",
    r"\bstupid\b",
    r"\bidiot\b",
    r"\bshut\s+up\b",
    r"\bcrap\b",
]

# Category-specific keywords that signal the need for certain phrases
_CATEGORY_RULES: dict[str, dict[str, Any]] = {
    "refund": {
        "required_patterns": [
            r"\b(acknowledge|understand|noted|received)\b",
            r"\b(refund|return|reimburse|credit)\b",
        ],
        "description": "Refund emails must acknowledge the refund request.",
    },
    "complaint": {
        "required_patterns": [
            r"\b(sorry|apologize|apologise|regret|understand)\b",
        ],
        "description": "Complaint emails must express empathy or apology.",
    },
    "inquiry": {
        "required_patterns": [
            r"\b(help|assist|answer|information|detail)\b",
        ],
        "description": "Inquiry emails must offer help or provide information.",
    },
}


class BusinessRulesMetric(BaseMetric):
    """Validates structural and content business rules for email replies.

    Attributes:
        name: ``"business_rules"``
    """

    name: str = "business_rules"

    def __init__(self) -> None:
        logger.info("BusinessRulesMetric initialised.")

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
        """Run all business rules against *generated* reply.

        Args:
            generated: AI-generated reply text.
            reference: Ground-truth reply (unused).
            email: Original incoming email (used for category detection).
            context: Retrieved context (unused).

        Returns:
            ``MetricResult`` with per-rule pass/fail and compliance score.
        """
        if not generated.strip():
            return MetricResult(
                name=self.name,
                score=0.0,
                reasoning="Empty reply violates all business rules.",
            )

        rules_results: list[dict[str, Any]] = []

        # ── Rule 1: Word count ───────────────────────────────────────
        rules_results.append(self._check_word_count(generated))

        # ── Rule 2: Greeting ─────────────────────────────────────────
        rules_results.append(self._check_greeting(generated))

        # ── Rule 3: Sign-off ─────────────────────────────────────────
        rules_results.append(self._check_signoff(generated))

        # ── Rule 4: Appropriate language ─────────────────────────────
        rules_results.append(self._check_appropriate_language(generated))

        # ── Rule 5: Category-specific rules ──────────────────────────
        category = self._detect_category(email)
        if category:
            rules_results.append(
                self._check_category_rules(generated, category)
            )

        # ── Aggregate ────────────────────────────────────────────────
        passed = sum(1 for r in rules_results if r["passed"])
        total = len(rules_results)
        score = clamp(passed / total) if total > 0 else 1.0

        failed_rules = [r for r in rules_results if not r["passed"]]
        reasoning_parts: list[str] = [
            f"Business rule compliance: {passed}/{total} rules passed "
            f"(score={score:.2f})."
        ]
        for r in failed_rules:
            reasoning_parts.append(
                f"  ✗ {r['rule']}: {r['reason']}"
            )

        penalized = len(failed_rules) > 0

        return MetricResult(
            name=self.name,
            score=round(score, 4),
            details={
                "rules": rules_results,
                "passed_count": passed,
                "total_count": total,
                "failed_rules": [r["rule"] for r in failed_rules],
                "detected_category": category,
            },
            reasoning="\n".join(reasoning_parts),
            penalized=penalized,
            penalty_reason=(
                f"{len(failed_rules)} business rule(s) failed."
                if penalized
                else ""
            ),
        )

    # ------------------------------------------------------------------
    # Individual rule checks
    # ------------------------------------------------------------------

    @staticmethod
    def _check_word_count(text: str) -> dict[str, Any]:
        """Check word count is within 50-500 range."""
        word_count = len(text.split())
        passed = 50 <= word_count <= 500
        reason = (
            f"Word count is {word_count} (within range)."
            if passed
            else f"Word count is {word_count} (outside 50–500 range)."
        )
        return {
            "rule": "word_count",
            "passed": passed,
            "reason": reason,
            "value": word_count,
        }

    @staticmethod
    def _check_greeting(text: str) -> dict[str, Any]:
        """Check for presence of a greeting."""
        # Check the first ~100 characters for greeting patterns
        header = text[:200].lower()
        found = any(
            re.search(p, header, re.IGNORECASE) for p in _GREETING_PATTERNS
        )
        return {
            "rule": "greeting",
            "passed": found,
            "reason": (
                "Greeting detected." if found else "No greeting found in the opening."
            ),
        }

    @staticmethod
    def _check_signoff(text: str) -> dict[str, Any]:
        """Check for presence of a sign-off."""
        # Check the last ~200 characters for sign-off patterns
        footer = text[-300:].lower()
        found = any(
            re.search(p, footer, re.IGNORECASE) for p in _SIGNOFF_PATTERNS
        )
        return {
            "rule": "sign_off",
            "passed": found,
            "reason": (
                "Sign-off detected." if found else "No sign-off found at the end."
            ),
        }

    @staticmethod
    def _check_appropriate_language(text: str) -> dict[str, Any]:
        """Check for absence of inappropriate language."""
        found_terms: list[str] = []
        for pattern in _INAPPROPRIATE_PATTERNS:
            matches = re.findall(pattern, text, re.IGNORECASE)
            found_terms.extend(matches)
        passed = len(found_terms) == 0
        return {
            "rule": "appropriate_language",
            "passed": passed,
            "reason": (
                "No inappropriate language detected."
                if passed
                else f"Inappropriate term(s) found: {', '.join(found_terms[:5])}."
            ),
        }

    @staticmethod
    def _check_category_rules(
        text: str,
        category: str,
    ) -> dict[str, Any]:
        """Check category-specific content requirements."""
        rules = _CATEGORY_RULES.get(category)
        if not rules:
            return {
                "rule": f"category_{category}",
                "passed": True,
                "reason": f"No specific rules defined for category '{category}'.",
            }

        required = rules["required_patterns"]
        all_matched = all(
            re.search(p, text, re.IGNORECASE) for p in required
        )
        return {
            "rule": f"category_{category}",
            "passed": all_matched,
            "reason": (
                f"{rules['description']} — requirement {'met' if all_matched else 'NOT met'}."
            ),
        }

    @staticmethod
    def _detect_category(email: str) -> str | None:
        """Simple keyword-based category detection from the email text."""
        lower = email.lower()
        if any(w in lower for w in ("refund", "return", "money back", "reimburse")):
            return "refund"
        if any(
            w in lower
            for w in ("complaint", "dissatisfied", "unhappy", "terrible", "worst")
        ):
            return "complaint"
        if any(
            w in lower
            for w in ("question", "inquiry", "asking", "could you", "can you")
        ):
            return "inquiry"
        return None
