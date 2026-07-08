"""
Readability Metric (Bonus / Penalty Modifier).

Evaluates the readability and structural quality of the generated reply
using:

1. **Flesch Reading Ease** – via the ``textstat`` library.
2. **Word count & sentence length** analysis.
3. **Basic grammar checks** – capitalisation, punctuation, common errors.

The normalised score (0–1) is used as a **modifier** on the composite
score rather than as a primary evaluation dimension.  Extremely poor
readability penalises, while good readability gives a small bonus.
"""

from __future__ import annotations

import logging
import re
from typing import Any

from evaluation.metrics import BaseMetric, MetricResult, clamp

logger = logging.getLogger(__name__)

# Ideal ranges
_IDEAL_WORD_COUNT: tuple[int, int] = (50, 500)
_IDEAL_AVG_SENTENCE_LENGTH: tuple[int, int] = (10, 25)
_FLESCH_EXCELLENT_THRESHOLD: float = 60.0  # 60-100 is "easy to read"


class ReadabilityMetric(BaseMetric):
    """Assesses readability and surface-level grammar of the generated reply.

    Attributes:
        name: ``"readability"``
    """

    name: str = "readability"

    def __init__(self) -> None:
        logger.info("ReadabilityMetric initialised.")

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
        """Analyse readability of *generated* text.

        Args:
            generated: AI-generated reply text.
            reference: Ground-truth reply (unused).
            email: Original incoming email (unused).
            context: Retrieved context (unused).

        Returns:
            ``MetricResult`` with readability breakdown.
        """
        if not generated.strip():
            return MetricResult(
                name=self.name,
                score=0.0,
                reasoning="Empty text – readability cannot be assessed.",
            )

        # ── Flesch Reading Ease ──────────────────────────────────────
        flesch_score = self._flesch_reading_ease(generated)
        flesch_normalised = clamp(flesch_score / 100.0)

        # ── Word / sentence analysis ─────────────────────────────────
        word_count = len(generated.split())
        sentences = self._split_sentences(generated)
        sentence_count = len(sentences)
        avg_sentence_length = (
            word_count / sentence_count if sentence_count > 0 else word_count
        )

        # Word count score
        if _IDEAL_WORD_COUNT[0] <= word_count <= _IDEAL_WORD_COUNT[1]:
            word_count_score = 1.0
        elif word_count < _IDEAL_WORD_COUNT[0]:
            word_count_score = clamp(word_count / _IDEAL_WORD_COUNT[0])
        else:
            # Penalise overly long responses
            overshoot = word_count - _IDEAL_WORD_COUNT[1]
            word_count_score = clamp(1.0 - (overshoot / _IDEAL_WORD_COUNT[1]))

        # Sentence length score
        if (
            _IDEAL_AVG_SENTENCE_LENGTH[0]
            <= avg_sentence_length
            <= _IDEAL_AVG_SENTENCE_LENGTH[1]
        ):
            sentence_len_score = 1.0
        else:
            distance = min(
                abs(avg_sentence_length - _IDEAL_AVG_SENTENCE_LENGTH[0]),
                abs(avg_sentence_length - _IDEAL_AVG_SENTENCE_LENGTH[1]),
            )
            sentence_len_score = clamp(1.0 - (distance / 30.0))

        # ── Grammar checks ──────────────────────────────────────────
        grammar_issues = self._check_grammar(generated)
        grammar_penalty = min(len(grammar_issues) * 0.05, 0.3)
        grammar_score = clamp(1.0 - grammar_penalty)

        # ── Combined score ───────────────────────────────────────────
        combined = clamp(
            0.35 * flesch_normalised
            + 0.20 * word_count_score
            + 0.20 * sentence_len_score
            + 0.25 * grammar_score
        )

        # Build reasoning
        issues_str = (
            "; ".join(grammar_issues[:5])
            if grammar_issues
            else "None detected"
        )
        reasoning = (
            f"Flesch Reading Ease={flesch_score:.1f} "
            f"(normalised={flesch_normalised:.2f}). "
            f"Word count={word_count} (score={word_count_score:.2f}). "
            f"Avg sentence length={avg_sentence_length:.1f} words "
            f"(score={sentence_len_score:.2f}). "
            f"Grammar issues ({len(grammar_issues)}): {issues_str}. "
            f"Combined readability={combined:.2f}."
        )

        return MetricResult(
            name=self.name,
            score=round(combined, 4),
            details={
                "flesch_reading_ease": round(flesch_score, 2),
                "flesch_normalised": round(flesch_normalised, 4),
                "word_count": word_count,
                "sentence_count": sentence_count,
                "avg_sentence_length": round(avg_sentence_length, 2),
                "word_count_score": round(word_count_score, 4),
                "sentence_length_score": round(sentence_len_score, 4),
                "grammar_score": round(grammar_score, 4),
                "grammar_issues": grammar_issues,
                "grammar_issue_count": len(grammar_issues),
            },
            reasoning=reasoning,
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _flesch_reading_ease(text: str) -> float:
        """Compute Flesch Reading Ease using ``textstat``.

        Falls back to a heuristic if the library is unavailable.
        """
        try:
            import textstat

            return float(textstat.flesch_reading_ease(text))
        except ImportError:
            logger.warning(
                "textstat not installed – using rough heuristic for "
                "Flesch Reading Ease."
            )
            # Rough approximation
            words = text.split()
            sentences = max(text.count(".") + text.count("!") + text.count("?"), 1)
            syllables = sum(
                max(1, len(re.findall(r"[aeiouy]+", w, re.I))) for w in words
            )
            word_count = len(words)
            if word_count == 0:
                return 0.0
            return (
                206.835
                - 1.015 * (word_count / sentences)
                - 84.6 * (syllables / word_count)
            )
        except Exception as exc:
            logger.warning("Flesch computation failed: %s", exc)
            return 50.0  # neutral fallback

    @staticmethod
    def _split_sentences(text: str) -> list[str]:
        """Split text into sentences using basic regex."""
        sentences = re.split(r"(?<=[.!?])\s+", text.strip())
        return [s for s in sentences if s.strip()]

    @staticmethod
    def _check_grammar(text: str) -> list[str]:
        """Perform basic grammar checks and return a list of issue descriptions.

        This is intentionally lightweight — a full grammar checker
        (e.g. LanguageTool) could be plugged in as an enhancement.
        """
        issues: list[str] = []

        # 1. Sentences not starting with uppercase
        sentences = re.split(r"(?<=[.!?])\s+", text.strip())
        for i, sentence in enumerate(sentences):
            stripped = sentence.strip()
            if stripped and stripped[0].islower():
                issues.append(
                    f"Sentence {i + 1} does not start with a capital letter."
                )

        # 2. Double spaces
        if "  " in text:
            issues.append("Contains double spaces.")

        # 3. Missing final punctuation
        stripped_text = text.strip()
        if stripped_text and stripped_text[-1] not in ".!?":
            issues.append("Text does not end with punctuation (., !, or ?).")

        # 4. Repeated words (e.g., "the the")
        repeated = re.findall(r"\b(\w+)\s+\1\b", text, re.IGNORECASE)
        for word in repeated[:3]:
            issues.append(f'Repeated word: "{word}".')

        # 5. Common typos / errors
        _common_errors = {
            r"\bi\b": "Lowercase 'i' used instead of 'I'.",
            r"\bteh\b": "Common typo: 'teh' instead of 'the'.",
            r"\brecieve\b": "Misspelling: 'recieve' should be 'receive'.",
            r"\boccured\b": "Misspelling: 'occured' should be 'occurred'.",
            r"\bseperately\b": "Misspelling: 'seperately' should be 'separately'.",
        }
        for pattern, message in _common_errors.items():
            if re.search(pattern, text):
                issues.append(message)

        return issues
