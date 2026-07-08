"""
Metric validation experiments.

Validates that the evaluation metrics actually measure quality by:
1. Testing with deliberately degraded responses
2. Comparing lexical vs semantic metrics
3. Computing correlations between metrics
4. Showing that composite score aligns with human intuition

Usage:
    python -m experiments.metric_validation
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import get_settings
from evaluation.evaluator import CompositeEvaluator
from evaluation.semantic_similarity import SemanticSimilarityMetric
from evaluation.readability import ReadabilityMetric

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Test Cases — Deliberately Crafted for Validation
# ---------------------------------------------------------------------------

VALIDATION_CASES = [
    {
        "name": "Perfect Response",
        "email": "Hi, I'd like to request a refund for order #12345. The product arrived damaged.",
        "reference": "Dear Customer,\n\nThank you for reaching out. I'm sorry to hear that your product arrived damaged. I've initiated a full refund for order #12345, which should reflect in your account within 3-5 business days.\n\nIf you need any further assistance, please don't hesitate to contact us.\n\nBest regards,\nCustomer Support",
        "generated": "Dear Customer,\n\nThank you for reaching out. I'm sorry to hear that your product arrived damaged. I've initiated a full refund for order #12345, which should reflect in your account within 3-5 business days.\n\nIf you need any further assistance, please don't hesitate to contact us.\n\nBest regards,\nCustomer Support",
        "expected_range": (85, 100),
        "description": "Identical to reference — should score very high",
    },
    {
        "name": "Good Paraphrase",
        "email": "Hi, I'd like to request a refund for order #12345. The product arrived damaged.",
        "reference": "Dear Customer,\n\nThank you for reaching out. I'm sorry to hear that your product arrived damaged. I've initiated a full refund for order #12345, which should reflect in your account within 3-5 business days.\n\nBest regards,\nCustomer Support",
        "generated": "Hello,\n\nWe apologize for the inconvenience with your damaged order. A refund for order #12345 has been processed and will appear in your account in 3-5 business days.\n\nPlease let us know if there's anything else we can help with.\n\nRegards,\nSupport Team",
        "expected_range": (70, 95),
        "description": "Paraphrased but covers same content — should score well",
    },
    {
        "name": "Partial Response (Missing Intent)",
        "email": "Can I change my shipping address and also get a refund for order #12345?",
        "reference": "Hi,\n\nOf course! I've updated your shipping address and initiated a refund for order #12345.\n\nBest,\nSupport",
        "generated": "Hi,\n\nI've updated your shipping address as requested.\n\nBest,\nSupport",
        "expected_range": (30, 60),
        "description": "Only addresses one of two requests — should lose intent coverage points",
    },
    {
        "name": "Hallucinated Response",
        "email": "What's the status of my order #99999?",
        "reference": "Hi,\n\nLet me look into order #99999 for you. I'll check the status and get back to you within the hour.\n\nBest,\nSupport",
        "generated": "Hi,\n\nYour order #99999 was delivered yesterday at 3:42 PM and signed for by John Smith. The tracking number is 1Z999AA1012345678.\n\nBest,\nSupport",
        "expected_range": (10, 40),
        "description": "Fabricates specific details — should be heavily penalized for hallucination",
    },
    {
        "name": "Wrong Tone",
        "email": "I'm extremely frustrated! Your product broke after ONE day!",
        "reference": "Dear Customer,\n\nI sincerely apologize for this experience. I completely understand your frustration, and this is not the quality we stand for. Let me arrange an immediate replacement for you.\n\nBest regards,\nSupport",
        "generated": "lol yeah stuff breaks sometimes 🤷 just buy another one",
        "expected_range": (0, 20),
        "description": "Completely wrong tone — should score very low across all dimensions",
    },
    {
        "name": "Irrelevant Response",
        "email": "How do I reset my password?",
        "reference": "Hi,\n\nTo reset your password:\n1. Go to our login page\n2. Click 'Forgot Password'\n3. Enter your email address\n4. Check your inbox for a reset link\n\nBest,\nSupport",
        "generated": "Thank you for your interest in our premium subscription plan! It starts at $29.99/month and includes unlimited access to all features.",
        "expected_range": (0, 15),
        "description": "Completely off-topic — should score near zero on intent and completeness",
    },
    {
        "name": "Verbose but Complete",
        "email": "What are your business hours?",
        "reference": "Our business hours are Monday to Friday, 9 AM to 5 PM EST.",
        "generated": "Thank you for your inquiry about our business hours. We truly appreciate your interest in reaching out to us. Our team is available to assist you during the following hours: Monday through Friday, from 9:00 AM to 5:00 PM Eastern Standard Time. During these hours, our dedicated customer support representatives are ready and eager to help you with any questions, concerns, or issues you may have. We look forward to hearing from you during these times. Please don't hesitate to reach out. We value your business greatly.",
        "expected_range": (50, 75),
        "description": "Correct answer but excessively verbose — readability penalty",
    },
]


def run_validation_experiment() -> dict:
    """Run all validation test cases and analyze results."""
    logger.info("=" * 60)
    logger.info("METRIC VALIDATION EXPERIMENT")
    logger.info("=" * 60)
    
    evaluator = CompositeEvaluator()
    results = []
    
    for case in VALIDATION_CASES:
        logger.info(f"\nEvaluating: {case['name']}")
        logger.info(f"  Expected range: {case['expected_range']}")
        
        try:
            eval_result = evaluator.evaluate_single(
                email=case["email"],
                generated=case["generated"],
                reference=case["reference"],
            )
            
            score = eval_result.composite_score
            in_range = case["expected_range"][0] <= score <= case["expected_range"][1]
            
            result = {
                "name": case["name"],
                "description": case["description"],
                "expected_range": case["expected_range"],
                "actual_score": round(score, 2),
                "in_expected_range": in_range,
                "metric_scores": {
                    name: round(r.score, 3)
                    for name, r in eval_result.metric_results.items()
                },
                "strengths": eval_result.strengths,
                "weaknesses": eval_result.weaknesses,
            }
            
            results.append(result)
            
            status = "✅ PASS" if in_range else "❌ FAIL"
            logger.info(f"  Score: {score:.1f} {status}")
            
        except Exception as e:
            logger.error(f"  Error: {e}")
            results.append({
                "name": case["name"],
                "error": str(e),
                "in_expected_range": False,
            })
    
    # Summary
    passed = sum(1 for r in results if r.get("in_expected_range", False))
    total = len(results)
    
    logger.info("\n" + "=" * 60)
    logger.info(f"VALIDATION RESULTS: {passed}/{total} passed")
    logger.info("=" * 60)
    
    for r in results:
        status = "✅" if r.get("in_expected_range") else "❌"
        score = r.get("actual_score", "ERR")
        expected = r.get("expected_range", "N/A")
        logger.info(f"  {status} {r['name']:30s} Score: {score:>6} Expected: {expected}")
    
    return {"results": results, "passed": passed, "total": total}


def run_lexical_vs_semantic_comparison() -> dict:
    """Compare lexical (BLEU-like) vs semantic metrics to show why semantic is better."""
    logger.info("\n" + "=" * 60)
    logger.info("LEXICAL vs SEMANTIC COMPARISON")
    logger.info("=" * 60)
    
    # Cases where lexical and semantic metrics should diverge
    comparison_cases = [
        {
            "name": "Same meaning, different words",
            "reference": "Your refund has been processed and will arrive in 3-5 days.",
            "generated": "We've completed the reimbursement — expect it within three to five business days.",
            "expected": "Semantic HIGH, Lexical LOW",
        },
        {
            "name": "Same words, different meaning",
            "reference": "We cannot process your refund at this time.",
            "generated": "We can process your refund at this time.",
            "expected": "Semantic LOW, Lexical HIGH",
        },
        {
            "name": "Formal vs informal (same content)",
            "reference": "Dear Sir, I regret to inform you that your request has been declined.",
            "generated": "Hey, sorry but we had to turn down your request.",
            "expected": "Semantic MEDIUM-HIGH, Lexical LOW",
        },
    ]
    
    sem_metric = SemanticSimilarityMetric()
    results = []
    
    for case in comparison_cases:
        sem_result = sem_metric.evaluate(
            generated=case["generated"],
            reference=case["reference"],
            email="test email",
        )
        
        # Simple word overlap (pseudo-BLEU)
        ref_words = set(case["reference"].lower().split())
        gen_words = set(case["generated"].lower().split())
        lexical_overlap = len(ref_words & gen_words) / max(len(ref_words | gen_words), 1)
        
        result = {
            "name": case["name"],
            "semantic_score": round(sem_result.score, 3),
            "lexical_overlap": round(lexical_overlap, 3),
            "expected": case["expected"],
            "divergence": round(abs(sem_result.score - lexical_overlap), 3),
        }
        results.append(result)
        
        logger.info(f"\n  {case['name']}:")
        logger.info(f"    Semantic score:  {result['semantic_score']}")
        logger.info(f"    Lexical overlap: {result['lexical_overlap']}")
        logger.info(f"    Expected:        {case['expected']}")
    
    return {"comparisons": results}


def run_correlation_analysis(reports_path: Path | None = None) -> dict:
    """Analyze correlations between different metrics using evaluation results."""
    settings = get_settings()
    reports_file = reports_path or (settings.reports_abs_path / "evaluation_details.json")
    
    if not reports_file.exists():
        logger.warning(f"Reports file not found: {reports_file}. Run evaluation first.")
        return {"error": "No evaluation data available"}
    
    with open(reports_file) as f:
        reports_data = json.load(f)
    
    if not reports_data:
        return {"error": "Empty reports data"}
    
    # Build correlation matrix
    metric_cols = [
        "semantic_similarity_score", "intent_coverage_score",
        "completeness_score", "tone_score", "grounding_score",
        "hallucination_score", "readability_score", "llm_judge_score",
    ]
    
    rows = []
    for r in reports_data:
        row = {col: r.get(col, 0) for col in metric_cols}
        row["composite_score"] = r.get("composite_score", 0)
        rows.append(row)
    
    df = pd.DataFrame(rows)
    corr_matrix = df.corr()
    
    logger.info("\nMetric Correlation Matrix:")
    logger.info(corr_matrix.to_string())
    
    # Find strongest correlations with composite score
    composite_corrs = corr_matrix["composite_score"].drop("composite_score").sort_values(ascending=False)
    
    logger.info("\nCorrelation with Composite Score:")
    for metric, corr in composite_corrs.items():
        logger.info(f"  {metric:35s}: {corr:.3f}")
    
    return {
        "correlation_matrix": corr_matrix.to_dict(),
        "composite_correlations": composite_corrs.to_dict(),
    }


def main() -> None:
    """Run all validation experiments."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s │ %(levelname)-8s │ %(message)s",
        datefmt="%H:%M:%S",
    )
    
    settings = get_settings()
    output_dir = settings.reports_abs_path
    output_dir.mkdir(parents=True, exist_ok=True)
    
    all_results = {}
    
    # 1. Validation test cases
    all_results["validation"] = run_validation_experiment()
    
    # 2. Lexical vs Semantic
    all_results["lexical_vs_semantic"] = run_lexical_vs_semantic_comparison()
    
    # 3. Correlation analysis (if evaluation data exists)
    all_results["correlations"] = run_correlation_analysis()
    
    # Save results
    output_file = output_dir / "metric_validation_results.json"
    with open(output_file, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    
    logger.info(f"\nAll validation results saved to {output_file}")


if __name__ == "__main__":
    main()
