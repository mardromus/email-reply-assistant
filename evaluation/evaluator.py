"""
Composite Evaluator — orchestrates ALL evaluation metrics.

This is the primary entry point for evaluating generated email responses.
It runs every metric, computes a weighted composite score (0–100), applies
readability and business-rule modifiers, and generates actionable feedback
(strengths, weaknesses, improvement suggestions).

Usage::

    from evaluation.evaluator import CompositeEvaluator

    evaluator = CompositeEvaluator()

    # Single evaluation
    result = evaluator.evaluate_single(
        email="I'd like a refund for order #123.",
        generated="Dear customer, we have processed your refund...",
        reference="Hi, your refund for order #123 has been initiated...",
        context=["Previous refund reply example..."],
    )
    print(result.composite_score)  # e.g. 78.5

    # Batch evaluation
    import pandas as pd
    df = pd.DataFrame({"email": [...], "generated": [...], "reference": [...]})
    results_df = evaluator.evaluate_batch(df)
"""

from __future__ import annotations

import logging
import time
from typing import Any

import pandas as pd
from pydantic import BaseModel, Field

from config import get_settings
from evaluation.metrics import BaseMetric, MetricResult, clamp

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Result model
# ---------------------------------------------------------------------------

class EvaluationResult(BaseModel):
    """Complete evaluation result for a single email–reply pair.

    Attributes:
        email: The original incoming email text.
        generated: The AI-generated reply.
        reference: The ground-truth / expected reply.
        metric_results: Per-metric results keyed by metric name.
        composite_score: Weighted composite score on a 0–100 scale.
        strengths: Identified strong points of the generated reply.
        weaknesses: Identified weak points of the generated reply.
        improvements: Actionable suggestions for improvement.
        readability_modifier: Readability-based modifier applied to composite.
        business_rule_compliance: Overall business-rule compliance (0–1).
        llm_judge_score: Independent LLM-as-Judge score (0–1).
        factuality_score: Factual consistency score (0–1).
        evaluation_time_seconds: Wall-clock time for the full evaluation.
    """

    email: str
    generated: str
    reference: str
    metric_results: dict[str, MetricResult] = Field(default_factory=dict)
    composite_score: float = Field(
        ge=0.0, le=100.0, default=0.0,
        description="Weighted composite score 0-100.",
    )
    strengths: list[str] = Field(default_factory=list)
    weaknesses: list[str] = Field(default_factory=list)
    improvements: list[str] = Field(default_factory=list)
    readability_modifier: float = Field(
        default=0.0, description="Readability modifier applied to composite."
    )
    business_rule_compliance: float = Field(
        default=1.0, description="Business-rule compliance 0-1."
    )
    llm_judge_score: float = Field(
        default=0.0, description="Independent LLM judge score 0-1."
    )
    factuality_score: float = Field(
        default=1.0, description="Factual consistency score 0-1."
    )
    evaluation_time_seconds: float = Field(
        default=0.0, description="Wall-clock seconds for evaluation."
    )


# ---------------------------------------------------------------------------
# Composite evaluator
# ---------------------------------------------------------------------------

class CompositeEvaluator:
    """Orchestrates all evaluation metrics and produces composite scores.

    Metrics are initialised lazily on first use to avoid import-time side
    effects.  Each metric uses ``safe_evaluate`` so that a failure in one
    metric does not prevent the others from running.

    Attributes:
        weights: Dimension weights loaded from ``config.evaluation_weights``.
    """

    def __init__(self) -> None:
        settings = get_settings()
        self.weights: dict[str, float] = settings.evaluation_weights

        # Lazy-initialised metric instances
        self._primary_metrics: dict[str, BaseMetric] | None = None
        self._auxiliary_metrics: dict[str, BaseMetric] | None = None

        logger.info(
            "CompositeEvaluator created with weights: %s",
            self.weights,
        )

    # ------------------------------------------------------------------
    # Lazy metric initialisation
    # ------------------------------------------------------------------

    def _init_metrics(self) -> None:
        """Initialise all metric instances (called once on first evaluate)."""
        if self._primary_metrics is not None:
            return  # already initialised

        logger.info("Initialising evaluation metrics …")

        self._primary_metrics = {}
        self._auxiliary_metrics = {}

        # ── Primary (weighted) metrics ───────────────────────────────
        try:
            from evaluation.semantic_similarity import SemanticSimilarityMetric
            self._primary_metrics["semantic_similarity"] = SemanticSimilarityMetric()
        except Exception as exc:
            logger.error("Failed to init SemanticSimilarityMetric: %s", exc)

        try:
            from evaluation.intent_coverage import IntentCoverageMetric
            self._primary_metrics["intent_coverage"] = IntentCoverageMetric()
        except Exception as exc:
            logger.error("Failed to init IntentCoverageMetric: %s", exc)

        try:
            from evaluation.completeness import CompletenessMetric
            self._primary_metrics["completeness"] = CompletenessMetric()
        except Exception as exc:
            logger.error("Failed to init CompletenessMetric: %s", exc)

        try:
            from evaluation.grounding import GroundingMetric
            self._primary_metrics["grounding"] = GroundingMetric()
        except Exception as exc:
            logger.error("Failed to init GroundingMetric: %s", exc)

        try:
            from evaluation.tone_consistency import ToneConsistencyMetric
            self._primary_metrics["tone"] = ToneConsistencyMetric()
        except Exception as exc:
            logger.error("Failed to init ToneConsistencyMetric: %s", exc)

        try:
            from evaluation.hallucination import HallucinationMetric
            self._primary_metrics["hallucination"] = HallucinationMetric()
        except Exception as exc:
            logger.error("Failed to init HallucinationMetric: %s", exc)

        # ── Auxiliary (modifier / cross-validation) metrics ──────────
        try:
            from evaluation.readability import ReadabilityMetric
            self._auxiliary_metrics["readability"] = ReadabilityMetric()
        except Exception as exc:
            logger.error("Failed to init ReadabilityMetric: %s", exc)

        try:
            from evaluation.llm_judge import LLMJudgeMetric
            self._auxiliary_metrics["llm_judge"] = LLMJudgeMetric()
        except Exception as exc:
            logger.error("Failed to init LLMJudgeMetric: %s", exc)

        try:
            from evaluation.business_rules import BusinessRulesMetric
            self._auxiliary_metrics["business_rules"] = BusinessRulesMetric()
        except Exception as exc:
            logger.error("Failed to init BusinessRulesMetric: %s", exc)

        try:
            from evaluation.factuality import FactualityMetric
            self._auxiliary_metrics["factuality"] = FactualityMetric()
        except Exception as exc:
            logger.error("Failed to init FactualityMetric: %s", exc)

        logger.info(
            "Metrics ready — %d primary, %d auxiliary.",
            len(self._primary_metrics),
            len(self._auxiliary_metrics),
        )

    # ------------------------------------------------------------------
    # Single evaluation
    # ------------------------------------------------------------------

    def evaluate_single(
        self,
        email: str,
        generated: str,
        reference: str,
        context: list[str] | None = None,
    ) -> EvaluationResult:
        """Evaluate a single generated reply against its reference.

        Runs all metrics, computes the weighted composite score, applies
        readability and business-rule modifiers, and produces actionable
        feedback.

        Args:
            email: Original incoming email.
            generated: AI-generated reply.
            reference: Expected / ground-truth reply.
            context: Retrieved context snippets used during generation.

        Returns:
            A fully populated ``EvaluationResult``.
        """
        self._init_metrics()
        assert self._primary_metrics is not None
        assert self._auxiliary_metrics is not None

        start_time = time.perf_counter()

        all_results: dict[str, MetricResult] = {}

        # ── Run primary metrics ──────────────────────────────────────
        for name, metric in self._primary_metrics.items():
            logger.debug("Running primary metric: %s", name)
            result = metric.safe_evaluate(generated, reference, email, context)
            all_results[name] = result
            logger.debug("  %s score=%.4f", name, result.score)

        # ── Run auxiliary metrics ────────────────────────────────────
        for name, metric in self._auxiliary_metrics.items():
            logger.debug("Running auxiliary metric: %s", name)
            result = metric.safe_evaluate(generated, reference, email, context)
            all_results[name] = result
            logger.debug("  %s score=%.4f", name, result.score)

        # ── Compute weighted composite (primary metrics only) ────────
        weighted_sum = 0.0
        total_weight = 0.0
        for dim_name, weight in self.weights.items():
            if dim_name in all_results:
                weighted_sum += weight * all_results[dim_name].score
                total_weight += weight

        raw_composite = (
            (weighted_sum / total_weight) if total_weight > 0 else 0.0
        )

        # ── Apply readability modifier ───────────────────────────────
        readability_result = all_results.get("readability")
        readability_modifier = 0.0
        if readability_result:
            # Modifier range: -5 to +5 on the 100-point scale
            readability_modifier = (readability_result.score - 0.5) * 10.0

        # ── Apply business-rule penalty ──────────────────────────────
        business_result = all_results.get("business_rules")
        business_compliance = business_result.score if business_result else 1.0
        # Penalty: up to -10 points for total non-compliance
        business_penalty = (1.0 - business_compliance) * 10.0

        composite_score = clamp(
            (raw_composite * 100.0) + readability_modifier - business_penalty,
            lo=0.0,
            hi=100.0,
        )

        # ── Extract auxiliary scores ─────────────────────────────────
        llm_judge_result = all_results.get("llm_judge")
        llm_judge_score = llm_judge_result.score if llm_judge_result else 0.0

        factuality_result = all_results.get("factuality")
        factuality_score = factuality_result.score if factuality_result else 1.0

        # ── Generate feedback ────────────────────────────────────────
        strengths = self._identify_strengths(all_results)
        weaknesses = self._identify_weaknesses(all_results)
        improvements = self._suggest_improvements(all_results, weaknesses)

        elapsed = time.perf_counter() - start_time

        logger.info(
            "Evaluation complete: composite=%.1f in %.2fs",
            composite_score,
            elapsed,
        )

        return EvaluationResult(
            email=email,
            generated=generated,
            reference=reference,
            metric_results=all_results,
            composite_score=round(composite_score, 2),
            strengths=strengths,
            weaknesses=weaknesses,
            improvements=improvements,
            readability_modifier=round(readability_modifier, 2),
            business_rule_compliance=round(business_compliance, 4),
            llm_judge_score=round(llm_judge_score, 4),
            factuality_score=round(factuality_score, 4),
            evaluation_time_seconds=round(elapsed, 3),
        )

    # ------------------------------------------------------------------
    # Batch evaluation
    # ------------------------------------------------------------------

    def evaluate_batch(
        self,
        data: pd.DataFrame,
        email_col: str = "email",
        generated_col: str = "generated",
        reference_col: str = "reference",
        context_col: str | None = None,
    ) -> pd.DataFrame:
        """Evaluate multiple email–reply pairs from a DataFrame.

        Args:
            data: Input DataFrame with at least ``email_col``,
                ``generated_col``, and ``reference_col`` columns.
            email_col: Column name for the original email text.
            generated_col: Column name for the generated reply.
            reference_col: Column name for the reference reply.
            context_col: Optional column name containing lists of context
                snippets.

        Returns:
            A copy of *data* augmented with evaluation result columns:
            ``composite_score``, ``strengths``, ``weaknesses``,
            ``improvements``, plus a column per primary metric score.
        """
        required_cols = {email_col, generated_col, reference_col}
        missing = required_cols - set(data.columns)
        if missing:
            raise ValueError(
                f"DataFrame missing required columns: {missing}"
            )

        results: list[EvaluationResult] = []
        total = len(data)

        logger.info("Starting batch evaluation of %d samples.", total)

        for idx, row in data.iterrows():
            logger.info("Evaluating sample %d / %d …", idx + 1, total)
            context = (
                row[context_col]
                if context_col and context_col in data.columns
                else None
            )
            result = self.evaluate_single(
                email=str(row[email_col]),
                generated=str(row[generated_col]),
                reference=str(row[reference_col]),
                context=context,
            )
            results.append(result)

        # Build output DataFrame
        out = data.copy()
        out["composite_score"] = [r.composite_score for r in results]
        out["readability_modifier"] = [r.readability_modifier for r in results]
        out["business_rule_compliance"] = [
            r.business_rule_compliance for r in results
        ]
        out["llm_judge_score"] = [r.llm_judge_score for r in results]
        out["factuality_score"] = [r.factuality_score for r in results]
        out["strengths"] = [r.strengths for r in results]
        out["weaknesses"] = [r.weaknesses for r in results]
        out["improvements"] = [r.improvements for r in results]
        out["evaluation_time_seconds"] = [
            r.evaluation_time_seconds for r in results
        ]

        # Add per-primary-metric scores
        for metric_name in self.weights:
            out[f"metric_{metric_name}"] = [
                r.metric_results[metric_name].score
                if metric_name in r.metric_results
                else None
                for r in results
            ]

        logger.info(
            "Batch evaluation complete. Mean composite=%.1f",
            out["composite_score"].mean(),
        )
        return out

    # ------------------------------------------------------------------
    # Feedback generation
    # ------------------------------------------------------------------

    @staticmethod
    def _identify_strengths(
        results: dict[str, MetricResult],
    ) -> list[str]:
        """Identify strong points based on high metric scores."""
        strengths: list[str] = []

        _strength_map: dict[str, str] = {
            "semantic_similarity": (
                "Strong semantic alignment with the reference reply."
            ),
            "intent_coverage": (
                "All or most intents from the email are addressed."
            ),
            "completeness": (
                "Thorough coverage of questions and requests."
            ),
            "grounding": (
                "Well-grounded in retrieved context examples."
            ),
            "tone": (
                "Tone is consistent with the expected reply style."
            ),
            "hallucination": (
                "No fabricated or hallucinated claims detected."
            ),
            "readability": (
                "Clear, readable, and well-structured text."
            ),
            "business_rules": (
                "Complies with all business rules (greeting, sign-off, length)."
            ),
            "llm_judge": (
                "High overall quality as assessed by the LLM judge."
            ),
            "factuality": (
                "Factually consistent with source materials."
            ),
        }

        for name, result in results.items():
            if result.score >= 0.80 and name in _strength_map:
                strengths.append(_strength_map[name])

        if not strengths:
            strengths.append(
                "No standout strengths identified; all metrics below 0.80."
            )

        return strengths

    @staticmethod
    def _identify_weaknesses(
        results: dict[str, MetricResult],
    ) -> list[str]:
        """Identify weak points based on low metric scores."""
        weaknesses: list[str] = []

        _weakness_map: dict[str, str] = {
            "semantic_similarity": (
                "Low semantic similarity to the reference reply."
            ),
            "intent_coverage": (
                "Some intents from the email are not addressed."
            ),
            "completeness": (
                "Not all questions or requests are fully answered."
            ),
            "grounding": (
                "Response is not well-grounded in retrieved context."
            ),
            "tone": (
                "Tone mismatch with the expected reply style."
            ),
            "hallucination": (
                "Potential hallucinated or fabricated claims detected."
            ),
            "readability": (
                "Readability issues (length, structure, or grammar)."
            ),
            "business_rules": (
                "Some business rules are not met."
            ),
            "llm_judge": (
                "Below-average quality as assessed by the LLM judge."
            ),
            "factuality": (
                "Factual inconsistencies with source materials."
            ),
        }

        for name, result in results.items():
            if result.score < 0.50 and name in _weakness_map:
                weaknesses.append(_weakness_map[name])
            # Also flag penalised metrics
            if result.penalized and name in _weakness_map:
                if _weakness_map[name] not in weaknesses:
                    weaknesses.append(
                        f"{_weakness_map[name]} (penalty: {result.penalty_reason})"
                    )

        return weaknesses

    @staticmethod
    def _suggest_improvements(
        results: dict[str, MetricResult],
        weaknesses: list[str],
    ) -> list[str]:
        """Generate actionable improvement suggestions based on weaknesses."""
        suggestions: list[str] = []

        _suggestion_map: dict[str, str] = {
            "semantic_similarity": (
                "Revise the reply to more closely match the meaning and "
                "key phrases of the expected response."
            ),
            "intent_coverage": (
                "Re-read the email and ensure every question, request, "
                "and topic is explicitly addressed."
            ),
            "completeness": (
                "Provide complete answers to all questions. Avoid "
                "deferring unless genuinely necessary."
            ),
            "grounding": (
                "Incorporate more information from the retrieved context "
                "examples to improve grounding."
            ),
            "tone": (
                "Adjust the tone to match the expected formality and "
                "empathy level."
            ),
            "hallucination": (
                "Remove any claims about dates, prices, policies, or "
                "reference numbers not supported by the source material."
            ),
            "readability": (
                "Improve readability: use shorter sentences, check grammar, "
                "and aim for 50–500 words."
            ),
            "business_rules": (
                "Ensure the reply includes a greeting, sign-off, and "
                "meets category-specific requirements."
            ),
            "factuality": (
                "Cross-check all factual claims against the original email "
                "and context to eliminate contradictions."
            ),
        }

        for name, result in results.items():
            if result.score < 0.60 and name in _suggestion_map:
                suggestions.append(_suggestion_map[name])

        if not suggestions and weaknesses:
            suggestions.append(
                "Review the identified weaknesses and refine the reply accordingly."
            )

        if not suggestions:
            suggestions.append(
                "The reply is strong overall. Consider minor polish for perfection."
            )

        return suggestions
