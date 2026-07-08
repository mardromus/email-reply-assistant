"""
Evaluation dashboard data generation module.

Generates chart data and aggregate visualizations for the evaluation dashboard.
Produces Plotly-ready data structures for the Streamlit UI.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from config import get_settings
from evaluation.report import PerResponseReport, AggregateReport, ReportGenerator

logger = logging.getLogger(__name__)


class DashboardData:
    """Generates chart-ready data for the evaluation dashboard."""
    
    def __init__(self, reports: list[PerResponseReport]):
        self.reports = reports
        self._generator = ReportGenerator()
        self.aggregate = self._generator.generate_aggregate_report(reports)
        logger.info(f"DashboardData initialized with {len(reports)} reports")
    
    # ── Score Distribution ────────────────────────────────────────────────
    
    def get_score_distribution(self, bins: int = 20) -> dict[str, Any]:
        """Get histogram data for composite score distribution."""
        scores = [r.composite_score for r in self.reports]
        hist, bin_edges = np.histogram(scores, bins=bins, range=(0, 100))
        return {
            "counts": hist.tolist(),
            "bin_edges": bin_edges.tolist(),
            "bin_labels": [
                f"{bin_edges[i]:.0f}-{bin_edges[i+1]:.0f}"
                for i in range(len(bin_edges) - 1)
            ],
            "mean": float(np.mean(scores)),
            "median": float(np.median(scores)),
            "std": float(np.std(scores)),
        }
    
    # ── Radar Chart Data ──────────────────────────────────────────────────
    
    def get_radar_chart_data(self, category: str | None = None) -> dict[str, Any]:
        """Get radar chart data for metric breakdown."""
        filtered = self.reports
        if category:
            filtered = [r for r in self.reports if r.category == category]
        
        if not filtered:
            return {"metrics": [], "values": []}
        
        metrics = [
            ("Semantic Similarity", "semantic_similarity_score"),
            ("Intent Coverage", "intent_coverage_score"),
            ("Completeness", "completeness_score"),
            ("Tone", "tone_score"),
            ("Grounding", "grounding_score"),
            ("Hallucination", "hallucination_score"),
            ("Readability", "readability_score"),
            ("LLM Judge", "llm_judge_score"),
        ]
        
        labels = [m[0] for m in metrics]
        values = [
            float(np.mean([getattr(r, m[1]) for r in filtered]))
            for m in metrics
        ]
        
        return {"metrics": labels, "values": values}
    
    def get_radar_chart_comparison(self) -> dict[str, Any]:
        """Get radar chart data comparing all categories."""
        categories = list(set(r.category for r in self.reports if r.category))
        
        result = {"metrics": [], "categories": {}}
        for cat in sorted(categories):
            data = self.get_radar_chart_data(cat)
            if not result["metrics"]:
                result["metrics"] = data["metrics"]
            result["categories"][cat] = data["values"]
        
        return result
    
    # ── Category Performance ──────────────────────────────────────────────
    
    def get_category_performance(self) -> pd.DataFrame:
        """Get category-wise performance as a DataFrame."""
        rows = []
        for cat, scores in self.aggregate.category_scores.items():
            rows.append({
                "Category": cat,
                "Count": scores["count"],
                "Avg Score": scores["avg_score"],
                "Min Score": scores["min_score"],
                "Max Score": scores["max_score"],
            })
        
        return pd.DataFrame(rows).sort_values("Avg Score", ascending=False)
    
    # ── Metric Correlation ────────────────────────────────────────────────
    
    def get_metric_correlation_matrix(self) -> pd.DataFrame:
        """Compute correlation matrix between all metrics."""
        metric_names = [
            "semantic_similarity_score", "intent_coverage_score",
            "completeness_score", "tone_score", "grounding_score",
            "hallucination_score", "readability_score", "llm_judge_score",
        ]
        
        data = {
            name.replace("_score", ""): [getattr(r, name) for r in self.reports]
            for name in metric_names
        }
        
        df = pd.DataFrame(data)
        return df.corr()
    
    # ── Per-Metric Distribution ───────────────────────────────────────────
    
    def get_metric_distributions(self) -> dict[str, list[float]]:
        """Get score distributions for each metric."""
        metric_names = [
            "semantic_similarity_score", "intent_coverage_score",
            "completeness_score", "tone_score", "grounding_score",
            "hallucination_score", "readability_score", "llm_judge_score",
        ]
        
        return {
            name.replace("_score", ""): [getattr(r, name) for r in self.reports]
            for name in metric_names
        }
    
    # ── Best / Worst Examples ─────────────────────────────────────────────
    
    def get_best_examples(self, n: int = 10) -> list[PerResponseReport]:
        """Get the top N highest-scoring responses."""
        return sorted(self.reports, key=lambda r: r.composite_score, reverse=True)[:n]
    
    def get_worst_examples(self, n: int = 10) -> list[PerResponseReport]:
        """Get the bottom N lowest-scoring responses."""
        return sorted(self.reports, key=lambda r: r.composite_score)[:n]
    
    # ── Failure Analysis ──────────────────────────────────────────────────
    
    def get_failure_analysis(self, threshold: float = 50.0) -> dict[str, Any]:
        """Analyze common failure patterns for low-scoring responses."""
        failures = [r for r in self.reports if r.composite_score < threshold]
        
        if not failures:
            return {
                "failure_count": 0,
                "failure_rate": 0.0,
                "common_issues": [],
                "worst_metrics": [],
            }
        
        # Find which metrics are consistently lowest in failures
        metric_names = [
            "semantic_similarity_score", "intent_coverage_score",
            "completeness_score", "tone_score", "grounding_score",
            "hallucination_score",
        ]
        
        metric_avgs = {}
        for name in metric_names:
            values = [getattr(r, name) for r in failures]
            metric_avgs[name.replace("_score", "")] = float(np.mean(values))
        
        worst_metrics = sorted(metric_avgs.items(), key=lambda x: x[1])
        
        # Collect common weaknesses
        weakness_counts: dict[str, int] = {}
        for r in failures:
            for w in r.weaknesses:
                key = w.strip().lower()[:80]
                weakness_counts[key] = weakness_counts.get(key, 0) + 1
        
        common_issues = [
            {"issue": k, "count": v}
            for k, v in sorted(weakness_counts.items(), key=lambda x: x[1], reverse=True)[:10]
        ]
        
        # Category breakdown for failures
        fail_categories: dict[str, int] = {}
        for r in failures:
            cat = r.category or "Unknown"
            fail_categories[cat] = fail_categories.get(cat, 0) + 1
        
        return {
            "failure_count": len(failures),
            "failure_rate": round(len(failures) / len(self.reports) * 100, 1),
            "common_issues": common_issues,
            "worst_metrics": [{"metric": m, "avg_score": s} for m, s in worst_metrics],
            "category_breakdown": fail_categories,
        }
    
    # ── Score Over Samples ────────────────────────────────────────────────
    
    def get_score_progression(self) -> list[float]:
        """Get composite scores in order (useful for seeing consistency)."""
        return [r.composite_score for r in self.reports]
    
    # ── Summary Statistics ────────────────────────────────────────────────
    
    def get_summary_stats(self) -> dict[str, Any]:
        """Get a compact summary of all dashboard metrics."""
        return {
            "total_samples": self.aggregate.total_samples,
            "avg_score": self.aggregate.avg_composite_score,
            "median_score": self.aggregate.median_composite_score,
            "std_score": self.aggregate.std_composite_score,
            "min_score": self.aggregate.min_composite_score,
            "max_score": self.aggregate.max_composite_score,
            "categories_count": len(self.aggregate.category_scores),
            "failure_rate": self.get_failure_analysis()["failure_rate"],
            "avg_metrics": self.aggregate.avg_scores,
        }
    
    def save_dashboard_data(self, output_dir: str | Path | None = None) -> Path:
        """Save all dashboard data as a JSON file for frontend consumption."""
        settings = get_settings()
        out_dir = Path(output_dir) if output_dir else settings.reports_abs_path
        out_dir.mkdir(parents=True, exist_ok=True)
        filepath = out_dir / "dashboard_data.json"
        
        data = {
            "summary": self.get_summary_stats(),
            "score_distribution": self.get_score_distribution(),
            "radar_overall": self.get_radar_chart_data(),
            "radar_comparison": self.get_radar_chart_comparison(),
            "category_performance": self.aggregate.category_scores,
            "metric_distributions": self.get_metric_distributions(),
            "failure_analysis": self.get_failure_analysis(),
            "best_examples": [r.model_dump() for r in self.get_best_examples(5)],
            "worst_examples": [r.model_dump() for r in self.get_worst_examples(5)],
        }
        
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, default=str)
        
        logger.info(f"Dashboard data saved to {filepath}")
        return filepath
