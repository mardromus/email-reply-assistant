"""
Evaluation framework for the AI Email Response System.

Provides a comprehensive suite of metrics for evaluating generated email
responses against reference replies, including semantic similarity, intent
coverage, completeness, tone analysis, grounding, hallucination detection,
factuality, readability, LLM-as-judge scoring, and business rule compliance.

Usage:
    from evaluation.evaluator import CompositeEvaluator

    evaluator = CompositeEvaluator()
    result = evaluator.evaluate_single(email, generated, reference, context)
"""

from evaluation.metrics import BaseMetric, MetricResult

__all__ = ["BaseMetric", "MetricResult"]
