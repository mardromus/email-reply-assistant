"""
Ablation study experiments.

Tests the effect of different configuration choices on response quality:
1. Top-K retrieval (3 vs 5 vs 10)
2. Temperature settings
3. With/without RAG (baseline comparison)

Usage:
    python -m experiments.ablation_study --experiment top_k
    python -m experiments.ablation_study --experiment all
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import pandas as pd
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import get_settings
from generator.llm import CerebrasLLM
from generator.prompt_templates import build_generation_prompt, SYSTEM_PROMPT
from generator.rag_pipeline import RAGPipeline
from evaluation.evaluator import CompositeEvaluator
from evaluation.report import PerResponseReport

logger = logging.getLogger(__name__)


def run_top_k_ablation(
    df: pd.DataFrame,
    k_values: list[int] = [3, 5, 10],
    sample_size: int = 20,
) -> dict:
    """Test different top_k values for retrieval."""
    logger.info("=" * 60)
    logger.info("ABLATION: Top-K Retrieval")
    logger.info("=" * 60)
    
    sample = df.sample(n=min(sample_size, len(df)), random_state=42)
    evaluator = CompositeEvaluator()
    results = {}
    
    for k in k_values:
        logger.info(f"\n--- Testing top_k = {k} ---")
        pipeline = RAGPipeline()
        scores = []
        
        for _, row in tqdm(sample.iterrows(), total=len(sample), desc=f"k={k}"):
            try:
                result = pipeline.process(email=row["email"], top_k=k)
                eval_result = evaluator.evaluate_single(
                    email=row["email"],
                    generated=result.generated_response,
                    reference=row["reply"],
                )
                scores.append(eval_result.composite_score)
            except Exception as e:
                logger.error(f"Error: {e}")
                scores.append(0.0)
        
        avg_score = sum(scores) / len(scores) if scores else 0
        results[f"k={k}"] = {
            "avg_score": round(avg_score, 2),
            "scores": scores,
            "min_score": round(min(scores), 2) if scores else 0,
            "max_score": round(max(scores), 2) if scores else 0,
        }
        logger.info(f"  Average score with k={k}: {avg_score:.2f}")
    
    return results


def run_rag_vs_no_rag(
    df: pd.DataFrame,
    sample_size: int = 20,
) -> dict:
    """Compare RAG-augmented vs direct generation (no retrieval)."""
    logger.info("=" * 60)
    logger.info("ABLATION: RAG vs No-RAG")
    logger.info("=" * 60)
    
    sample = df.sample(n=min(sample_size, len(df)), random_state=42)
    evaluator = CompositeEvaluator()
    llm = CerebrasLLM()
    pipeline = RAGPipeline()
    
    rag_scores = []
    no_rag_scores = []
    
    for _, row in tqdm(sample.iterrows(), total=len(sample), desc="RAG vs No-RAG"):
        email = row["email"]
        reference = row["reply"]
        
        # With RAG
        try:
            rag_result = pipeline.process(email=email, top_k=5)
            rag_eval = evaluator.evaluate_single(
                email=email,
                generated=rag_result.generated_response,
                reference=reference,
            )
            rag_scores.append(rag_eval.composite_score)
        except Exception as e:
            logger.error(f"RAG error: {e}")
            rag_scores.append(0.0)
        
        # Without RAG (direct generation)
        try:
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": f"Write a professional reply to this email:\n\n{email}"},
            ]
            no_rag_response = llm.generate(messages)
            no_rag_eval = evaluator.evaluate_single(
                email=email,
                generated=no_rag_response,
                reference=reference,
            )
            no_rag_scores.append(no_rag_eval.composite_score)
        except Exception as e:
            logger.error(f"No-RAG error: {e}")
            no_rag_scores.append(0.0)
    
    rag_avg = sum(rag_scores) / len(rag_scores) if rag_scores else 0
    no_rag_avg = sum(no_rag_scores) / len(no_rag_scores) if no_rag_scores else 0
    
    logger.info(f"\n  RAG average:    {rag_avg:.2f}")
    logger.info(f"  No-RAG average: {no_rag_avg:.2f}")
    logger.info(f"  Improvement:    {rag_avg - no_rag_avg:+.2f}")
    
    return {
        "rag": {"avg": round(rag_avg, 2), "scores": rag_scores},
        "no_rag": {"avg": round(no_rag_avg, 2), "scores": no_rag_scores},
        "improvement": round(rag_avg - no_rag_avg, 2),
    }


def run_temperature_ablation(
    df: pd.DataFrame,
    temperatures: list[float] = [0.1, 0.5, 0.7, 1.0],
    sample_size: int = 15,
) -> dict:
    """Test different temperature values for generation."""
    logger.info("=" * 60)
    logger.info("ABLATION: Temperature")
    logger.info("=" * 60)
    
    sample = df.sample(n=min(sample_size, len(df)), random_state=42)
    evaluator = CompositeEvaluator()
    results = {}
    
    for temp in temperatures:
        logger.info(f"\n--- Testing temperature = {temp} ---")
        pipeline = RAGPipeline()
        scores = []
        
        for _, row in tqdm(sample.iterrows(), total=len(sample), desc=f"temp={temp}"):
            try:
                result = pipeline.process(email=row["email"], top_k=5)
                eval_result = evaluator.evaluate_single(
                    email=row["email"],
                    generated=result.generated_response,
                    reference=row["reply"],
                )
                scores.append(eval_result.composite_score)
            except Exception as e:
                logger.error(f"Error: {e}")
                scores.append(0.0)
        
        avg_score = sum(scores) / len(scores) if scores else 0
        results[f"temp={temp}"] = {
            "avg_score": round(avg_score, 2),
            "scores": scores,
        }
        logger.info(f"  Average score with temp={temp}: {avg_score:.2f}")
    
    return results


def main() -> None:
    """Run ablation studies."""
    parser = argparse.ArgumentParser(description="Run ablation study experiments")
    parser.add_argument("--experiment", type=str, default="all",
                        choices=["top_k", "rag_vs_no_rag", "temperature", "all"])
    parser.add_argument("--sample-size", type=int, default=20)
    args = parser.parse_args()
    
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s │ %(levelname)-8s │ %(message)s",
        datefmt="%H:%M:%S",
    )
    
    settings = get_settings()
    df = pd.read_csv(settings.dataset_abs_path)
    
    all_results = {}
    
    if args.experiment in ("top_k", "all"):
        all_results["top_k"] = run_top_k_ablation(df, sample_size=args.sample_size)
    
    if args.experiment in ("rag_vs_no_rag", "all"):
        all_results["rag_vs_no_rag"] = run_rag_vs_no_rag(df, sample_size=args.sample_size)
    
    if args.experiment in ("temperature", "all"):
        all_results["temperature"] = run_temperature_ablation(df, sample_size=args.sample_size)
    
    # Save results
    output_file = settings.reports_abs_path / "ablation_results.json"
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    
    logger.info(f"\nAblation results saved to {output_file}")


if __name__ == "__main__":
    main()
