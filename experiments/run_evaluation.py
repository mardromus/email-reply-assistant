"""
Main experiment runner.

Runs the full pipeline: load dataset → build index → generate responses → evaluate → report.
Can be run as a script or imported and called programmatically.

Usage:
    python -m experiments.run_evaluation --sample-size 50
    python -m experiments.run_evaluation --full
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

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import get_settings
from retrieval.embedding import get_embedding_model
from retrieval.vector_store import EmailVectorStore
from retrieval.search import retrieve_similar
from generator.rag_pipeline import RAGPipeline
from evaluation.evaluator import CompositeEvaluator
from evaluation.report import PerResponseReport, ReportGenerator
from evaluation.dashboard import DashboardData

logger = logging.getLogger(__name__)


def setup_logging(level: str = "INFO") -> None:
    """Configure logging for the experiment."""
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format="%(asctime)s │ %(name)-25s │ %(levelname)-8s │ %(message)s",
        datefmt="%H:%M:%S",
    )


def load_dataset(path: Path, sample_size: int | None = None) -> pd.DataFrame:
    """Load the email dataset and optionally sample it."""
    logger.info(f"Loading dataset from {path}")
    df = pd.read_csv(path)
    logger.info(f"Dataset loaded: {len(df)} rows, {list(df.columns)}")
    
    if sample_size and sample_size < len(df):
        df = df.sample(n=sample_size, random_state=42).reset_index(drop=True)
        logger.info(f"Sampled {sample_size} rows for evaluation")
    
    return df


def build_vector_index(df: pd.DataFrame) -> EmailVectorStore:
    """Build or load the vector store index."""
    settings = get_settings()
    store = EmailVectorStore()
    
    # Check if index already exists
    existing_count = store.get_collection_count()
    if existing_count > 0:
        logger.info(f"Vector store already has {existing_count} entries. Skipping rebuild.")
        return store
    
    logger.info("Building vector store index from dataset...")
    store.build_index(df)
    logger.info(f"Vector store built with {store.get_collection_count()} entries")
    return store


def run_generation(
    pipeline: RAGPipeline,
    df: pd.DataFrame,
    top_k: int = 5,
) -> pd.DataFrame:
    """Generate responses for all emails in the dataset."""
    logger.info(f"Generating responses for {len(df)} emails...")
    
    results = []
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Generating responses"):
        try:
            result = pipeline.process(
                email=row["email"],
                top_k=top_k,
                category=row.get("category"),
            )
            results.append({
                "id": row.get("id", str(idx)),
                "category": row.get("category", ""),
                "email": row["email"],
                "ground_truth": row["reply"],
                "generated": result.generated_response,
                "model_used": result.model_used,
                "latency_ms": result.latency_ms,
                "tokens_used": result.tokens_used,
                "num_retrieved": len(result.retrieved_examples),
            })
        except Exception as e:
            logger.error(f"Error generating response for row {idx}: {e}")
            results.append({
                "id": row.get("id", str(idx)),
                "category": row.get("category", ""),
                "email": row["email"],
                "ground_truth": row["reply"],
                "generated": f"[ERROR: {str(e)}]",
                "model_used": "error",
                "latency_ms": 0,
                "tokens_used": 0,
                "num_retrieved": 0,
            })
    
    return pd.DataFrame(results)


def run_evaluation(
    predictions_df: pd.DataFrame,
    evaluator: CompositeEvaluator,
) -> list[PerResponseReport]:
    """Run evaluation on all generated responses."""
    logger.info(f"Evaluating {len(predictions_df)} responses...")
    
    reports = []
    for idx, row in tqdm(predictions_df.iterrows(), total=len(predictions_df), desc="Evaluating"):
        try:
            eval_result = evaluator.evaluate_single(
                email=row["email"],
                generated=row["generated"],
                reference=row["ground_truth"],
                context=None,  # Could pass retrieved examples here
            )
            
            report = PerResponseReport.from_evaluation_result(eval_result, email_id=row["id"])
            report.category = row.get("category", "")
            reports.append(report)
            
        except Exception as e:
            logger.error(f"Error evaluating row {idx}: {e}")
            # Create a minimal report for failed evaluations
            reports.append(PerResponseReport(
                id=row.get("id", str(idx)),
                incoming_email=row["email"],
                generated_reply=row["generated"],
                ground_truth_reply=row["ground_truth"],
                category=row.get("category", ""),
                composite_score=0.0,
                weaknesses=[f"Evaluation failed: {str(e)}"],
            ))
    
    return reports


def save_predictions(predictions_df: pd.DataFrame, output_path: Path) -> None:
    """Save predictions to CSV."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    predictions_df.to_csv(output_path, index=False, encoding="utf-8")
    logger.info(f"Predictions saved to {output_path}")


def main() -> None:
    """Main experiment entry point."""
    parser = argparse.ArgumentParser(description="Run email response evaluation experiment")
    parser.add_argument("--sample-size", type=int, default=50,
                        help="Number of samples to evaluate (default: 50)")
    parser.add_argument("--full", action="store_true",
                        help="Run evaluation on the full dataset")
    parser.add_argument("--top-k", type=int, default=5,
                        help="Number of similar emails to retrieve (default: 5)")
    parser.add_argument("--skip-generation", action="store_true",
                        help="Skip generation, use existing predictions.csv")
    parser.add_argument("--log-level", type=str, default="INFO",
                        choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    args = parser.parse_args()
    
    setup_logging(args.log_level)
    settings = get_settings()
    
    logger.info("=" * 60)
    logger.info("Email Response Evaluation Experiment")
    logger.info("=" * 60)
    
    start_time = time.time()
    sample_size = None if args.full else args.sample_size
    
    # ── Step 1: Load Dataset ──────────────────────────────────────────
    df = load_dataset(settings.dataset_abs_path, sample_size)
    
    # ── Step 2: Build Vector Index ────────────────────────────────────
    # Use full dataset for the index, even if evaluating a subset
    full_df = pd.read_csv(settings.dataset_abs_path)
    store = build_vector_index(full_df)
    
    # ── Step 3: Generate Responses ────────────────────────────────────
    predictions_path = settings.outputs_abs_path / "predictions.csv"
    
    if args.skip_generation and predictions_path.exists():
        logger.info("Loading existing predictions...")
        predictions_df = pd.read_csv(predictions_path)
    else:
        pipeline = RAGPipeline()
        predictions_df = run_generation(pipeline, df, top_k=args.top_k)
        save_predictions(predictions_df, predictions_path)
    
    # ── Step 4: Evaluate ──────────────────────────────────────────────
    evaluator = CompositeEvaluator()
    reports = run_evaluation(predictions_df, evaluator)
    
    # ── Step 5: Generate Reports ──────────────────────────────────────
    report_gen = ReportGenerator()
    report_paths = report_gen.generate_full_report(reports)
    
    # ── Step 6: Generate Dashboard Data ───────────────────────────────
    dashboard = DashboardData(reports)
    dashboard.save_dashboard_data()
    
    # ── Summary ───────────────────────────────────────────────────────
    elapsed = time.time() - start_time
    summary = dashboard.get_summary_stats()
    
    logger.info("=" * 60)
    logger.info("EXPERIMENT COMPLETE")
    logger.info("=" * 60)
    logger.info(f"  Samples evaluated: {summary['total_samples']}")
    logger.info(f"  Average score:     {summary['avg_score']:.1f} / 100")
    logger.info(f"  Median score:      {summary['median_score']:.1f} / 100")
    logger.info(f"  Best score:        {summary['max_score']:.1f} / 100")
    logger.info(f"  Worst score:       {summary['min_score']:.1f} / 100")
    logger.info(f"  Failure rate:      {summary['failure_rate']:.1f}%")
    logger.info(f"  Total time:        {elapsed:.1f}s")
    logger.info(f"  Reports saved to:  {settings.reports_abs_path}")
    logger.info("=" * 60)
    
    # Print metric breakdown
    logger.info("Metric Breakdown (Averages):")
    for metric, score in summary["avg_metrics"].items():
        logger.info(f"  {metric:25s}: {score:.3f}")


if __name__ == "__main__":
    main()
