"""
Evaluation report generation module.

Produces per-response and aggregate reports in JSON, CSV, and HTML formats.
Uses Jinja2 templates for rich HTML report generation.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
from jinja2 import Template
from pydantic import BaseModel, Field

from config import get_settings

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Report Data Models
# ---------------------------------------------------------------------------

class PerResponseReport(BaseModel):
    """Detailed evaluation report for a single email-response pair."""
    
    id: str = ""
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat())
    
    # Input/Output
    incoming_email: str
    generated_reply: str
    ground_truth_reply: str
    category: str = ""
    
    # Metric Scores (0-1 scale)
    semantic_similarity_score: float = 0.0
    intent_coverage_score: float = 0.0
    completeness_score: float = 0.0
    tone_score: float = 0.0
    grounding_score: float = 0.0
    hallucination_score: float = 0.0
    readability_score: float = 0.0
    llm_judge_score: float = 0.0
    business_rules_score: float = 0.0
    
    # Composite
    composite_score: float = 0.0  # 0-100
    
    # Reasoning
    metric_details: dict[str, Any] = Field(default_factory=dict)
    strengths: list[str] = Field(default_factory=list)
    weaknesses: list[str] = Field(default_factory=list)
    improvements: list[str] = Field(default_factory=list)
    
    @classmethod
    def from_evaluation_result(cls, eval_result: Any, email_id: str = "") -> PerResponseReport:
        """Create a report from an EvaluationResult object."""
        metric_scores = {}
        metric_details = {}
        
        for name, result in eval_result.metric_results.items():
            metric_scores[name] = result.score
            metric_details[name] = {
                "score": result.score,
                "reasoning": result.reasoning,
                "details": result.details,
                "penalized": result.penalized,
                "penalty_reason": result.penalty_reason,
            }
        
        return cls(
            id=email_id,
            incoming_email=eval_result.email,
            generated_reply=eval_result.generated,
            ground_truth_reply=eval_result.reference,
            semantic_similarity_score=metric_scores.get("semantic_similarity", 0.0),
            intent_coverage_score=metric_scores.get("intent_coverage", 0.0),
            completeness_score=metric_scores.get("completeness", 0.0),
            tone_score=metric_scores.get("tone", 0.0),
            grounding_score=metric_scores.get("grounding", 0.0),
            hallucination_score=metric_scores.get("hallucination", 0.0),
            readability_score=metric_scores.get("readability", 0.0),
            llm_judge_score=metric_scores.get("llm_judge", 0.0),
            business_rules_score=metric_scores.get("business_rules", 0.0),
            composite_score=eval_result.composite_score,
            metric_details=metric_details,
            strengths=eval_result.strengths,
            weaknesses=eval_result.weaknesses,
            improvements=eval_result.improvements,
        )


class AggregateReport(BaseModel):
    """Aggregate evaluation report across all email-response pairs."""
    
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat())
    total_samples: int = 0
    
    # Aggregate scores
    avg_composite_score: float = 0.0
    median_composite_score: float = 0.0
    std_composite_score: float = 0.0
    min_composite_score: float = 0.0
    max_composite_score: float = 0.0
    
    # Per-metric averages
    avg_scores: dict[str, float] = Field(default_factory=dict)
    median_scores: dict[str, float] = Field(default_factory=dict)
    
    # Category-wise breakdown
    category_scores: dict[str, dict[str, float]] = Field(default_factory=dict)
    
    # Best and worst
    best_examples: list[dict[str, Any]] = Field(default_factory=list)
    worst_examples: list[dict[str, Any]] = Field(default_factory=list)
    
    # Distribution data
    score_distribution: list[float] = Field(default_factory=list)
    
    # Failure analysis
    common_weaknesses: list[dict[str, Any]] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Report Generator
# ---------------------------------------------------------------------------

class ReportGenerator:
    """Generates evaluation reports in multiple formats."""
    
    def __init__(self, output_dir: str | Path | None = None):
        settings = get_settings()
        self.output_dir = Path(output_dir) if output_dir else settings.reports_abs_path
        self.output_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"ReportGenerator initialized. Output directory: {self.output_dir}")
    
    # ── Per-Response Reports ──────────────────────────────────────────────
    
    def save_per_response_json(
        self, reports: list[PerResponseReport], filename: str = "evaluation_details.json"
    ) -> Path:
        """Save per-response reports as JSON."""
        filepath = self.output_dir / filename
        data = [r.model_dump() for r in reports]
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        logger.info(f"Saved {len(reports)} per-response reports to {filepath}")
        return filepath
    
    def save_per_response_csv(
        self, reports: list[PerResponseReport], filename: str = "evaluation.csv"
    ) -> Path:
        """Save per-response reports as CSV."""
        settings = get_settings()
        filepath = settings.outputs_abs_path / filename
        
        rows = []
        for r in reports:
            rows.append({
                "id": r.id,
                "category": r.category,
                "composite_score": round(r.composite_score, 2),
                "semantic_similarity": round(r.semantic_similarity_score, 3),
                "intent_coverage": round(r.intent_coverage_score, 3),
                "completeness": round(r.completeness_score, 3),
                "tone": round(r.tone_score, 3),
                "grounding": round(r.grounding_score, 3),
                "hallucination": round(r.hallucination_score, 3),
                "readability": round(r.readability_score, 3),
                "llm_judge": round(r.llm_judge_score, 3),
                "business_rules": round(r.business_rules_score, 3),
                "strengths": "; ".join(r.strengths),
                "weaknesses": "; ".join(r.weaknesses),
                "improvements": "; ".join(r.improvements),
                "incoming_email": r.incoming_email[:200],
                "generated_reply": r.generated_reply[:200],
                "ground_truth_reply": r.ground_truth_reply[:200],
            })
        
        df = pd.DataFrame(rows)
        df.to_csv(filepath, index=False, encoding="utf-8")
        logger.info(f"Saved {len(reports)} evaluation rows to {filepath}")
        return filepath
    
    def save_per_response_html(
        self, reports: list[PerResponseReport], filename: str = "evaluation_report.html"
    ) -> Path:
        """Save per-response reports as styled HTML."""
        filepath = self.output_dir / filename
        html_content = self._render_html_report(reports)
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(html_content)
        logger.info(f"Saved HTML report to {filepath}")
        return filepath
    
    # ── Aggregate Reports ─────────────────────────────────────────────────
    
    def generate_aggregate_report(
        self, reports: list[PerResponseReport]
    ) -> AggregateReport:
        """Generate aggregate statistics from per-response reports."""
        if not reports:
            logger.warning("No reports to aggregate")
            return AggregateReport()
        
        composite_scores = [r.composite_score for r in reports]
        
        # Per-metric averages
        metric_names = [
            "semantic_similarity_score", "intent_coverage_score",
            "completeness_score", "tone_score", "grounding_score",
            "hallucination_score", "readability_score", "llm_judge_score",
            "business_rules_score",
        ]
        
        avg_scores = {}
        median_scores = {}
        for metric in metric_names:
            values = [getattr(r, metric) for r in reports]
            clean_name = metric.replace("_score", "")
            avg_scores[clean_name] = round(sum(values) / len(values), 4)
            sorted_vals = sorted(values)
            mid = len(sorted_vals) // 2
            median_scores[clean_name] = round(
                (sorted_vals[mid] + sorted_vals[~mid]) / 2, 4
            )
        
        # Category-wise breakdown
        category_groups: dict[str, list[PerResponseReport]] = {}
        for r in reports:
            cat = r.category or "Unknown"
            category_groups.setdefault(cat, []).append(r)
        
        category_scores = {}
        for cat, cat_reports in category_groups.items():
            cat_composites = [r.composite_score for r in cat_reports]
            category_scores[cat] = {
                "count": len(cat_reports),
                "avg_score": round(sum(cat_composites) / len(cat_composites), 2),
                "min_score": round(min(cat_composites), 2),
                "max_score": round(max(cat_composites), 2),
            }
        
        # Best and worst examples
        sorted_reports = sorted(reports, key=lambda r: r.composite_score, reverse=True)
        best = [
            {"id": r.id, "category": r.category, "score": r.composite_score,
             "email_preview": r.incoming_email[:100]}
            for r in sorted_reports[:10]
        ]
        worst = [
            {"id": r.id, "category": r.category, "score": r.composite_score,
             "email_preview": r.incoming_email[:100],
             "weaknesses": r.weaknesses}
            for r in sorted_reports[-10:]
        ]
        
        # Common weaknesses analysis
        weakness_counts: dict[str, int] = {}
        for r in reports:
            for w in r.weaknesses:
                # Normalize weakness text for grouping
                key = w.strip().lower()[:80]
                weakness_counts[key] = weakness_counts.get(key, 0) + 1
        
        common_weaknesses = [
            {"weakness": k, "count": v, "percentage": round(v / len(reports) * 100, 1)}
            for k, v in sorted(weakness_counts.items(), key=lambda x: x[1], reverse=True)[:15]
        ]
        
        sorted_composites = sorted(composite_scores)
        mid = len(sorted_composites) // 2
        
        aggregate = AggregateReport(
            total_samples=len(reports),
            avg_composite_score=round(sum(composite_scores) / len(composite_scores), 2),
            median_composite_score=round(
                (sorted_composites[mid] + sorted_composites[~mid]) / 2, 2
            ),
            std_composite_score=round(
                (sum((x - sum(composite_scores)/len(composite_scores))**2 
                     for x in composite_scores) / len(composite_scores)) ** 0.5, 2
            ),
            min_composite_score=round(min(composite_scores), 2),
            max_composite_score=round(max(composite_scores), 2),
            avg_scores=avg_scores,
            median_scores=median_scores,
            category_scores=category_scores,
            best_examples=best,
            worst_examples=worst,
            score_distribution=composite_scores,
            common_weaknesses=common_weaknesses,
        )
        
        return aggregate
    
    def save_aggregate_report(
        self, aggregate: AggregateReport, filename: str = "aggregate_report.json"
    ) -> Path:
        """Save aggregate report as JSON."""
        filepath = self.output_dir / filename
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(aggregate.model_dump(), f, indent=2, ensure_ascii=False)
        logger.info(f"Saved aggregate report to {filepath}")
        return filepath
    
    # ── Full Report Pipeline ──────────────────────────────────────────────
    
    def generate_full_report(
        self, reports: list[PerResponseReport]
    ) -> dict[str, Path]:
        """Generate all report formats and return file paths."""
        paths = {}
        
        # Per-response reports
        paths["json"] = self.save_per_response_json(reports)
        paths["csv"] = self.save_per_response_csv(reports)
        paths["html"] = self.save_per_response_html(reports)
        
        # Aggregate report
        aggregate = self.generate_aggregate_report(reports)
        paths["aggregate_json"] = self.save_aggregate_report(aggregate)
        
        logger.info(f"Full report generation complete. Files: {list(paths.keys())}")
        return paths
    
    # ── HTML Template ─────────────────────────────────────────────────────
    
    def _render_html_report(self, reports: list[PerResponseReport]) -> str:
        """Render HTML report using embedded Jinja2 template."""
        aggregate = self.generate_aggregate_report(reports)
        template = Template(HTML_REPORT_TEMPLATE)
        return template.render(
            reports=reports,
            aggregate=aggregate,
            timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        )


# ---------------------------------------------------------------------------
# HTML Report Template
# ---------------------------------------------------------------------------

HTML_REPORT_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Email Response Evaluation Report</title>
    <style>
        :root {
            --bg-primary: #0f0f23;
            --bg-secondary: #1a1a3e;
            --bg-card: #1e1e4a;
            --text-primary: #e8e8f0;
            --text-secondary: #a0a0c0;
            --accent-blue: #6366f1;
            --accent-purple: #8b5cf6;
            --accent-green: #10b981;
            --accent-amber: #f59e0b;
            --accent-red: #ef4444;
            --border: #2a2a5a;
            --gradient-1: linear-gradient(135deg, #6366f1, #8b5cf6);
            --gradient-2: linear-gradient(135deg, #10b981, #06b6d4);
        }
        
        * { margin: 0; padding: 0; box-sizing: border-box; }
        
        body {
            font-family: 'Inter', 'Segoe UI', system-ui, -apple-system, sans-serif;
            background: var(--bg-primary);
            color: var(--text-primary);
            line-height: 1.6;
        }
        
        .container { max-width: 1400px; margin: 0 auto; padding: 2rem; }
        
        .header {
            text-align: center;
            padding: 3rem 2rem;
            background: linear-gradient(135deg, #1a1a3e 0%, #2d1b69 50%, #1a1a3e 100%);
            border-radius: 20px;
            margin-bottom: 2rem;
            border: 1px solid var(--border);
        }
        
        .header h1 {
            font-size: 2.5rem;
            background: var(--gradient-1);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin-bottom: 0.5rem;
        }
        
        .header .subtitle { color: var(--text-secondary); font-size: 1.1rem; }
        
        .stats-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 1.5rem;
            margin-bottom: 2rem;
        }
        
        .stat-card {
            background: var(--bg-card);
            border: 1px solid var(--border);
            border-radius: 16px;
            padding: 1.5rem;
            text-align: center;
            transition: transform 0.2s, box-shadow 0.2s;
        }
        
        .stat-card:hover {
            transform: translateY(-4px);
            box-shadow: 0 8px 30px rgba(99, 102, 241, 0.15);
        }
        
        .stat-card .value {
            font-size: 2.5rem;
            font-weight: 700;
            background: var(--gradient-1);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }
        
        .stat-card .label {
            color: var(--text-secondary);
            font-size: 0.9rem;
            margin-top: 0.5rem;
        }
        
        .section {
            background: var(--bg-card);
            border: 1px solid var(--border);
            border-radius: 16px;
            padding: 2rem;
            margin-bottom: 2rem;
        }
        
        .section h2 {
            font-size: 1.5rem;
            margin-bottom: 1.5rem;
            color: var(--accent-blue);
        }
        
        .score-badge {
            display: inline-block;
            padding: 0.25rem 0.75rem;
            border-radius: 20px;
            font-weight: 600;
            font-size: 0.85rem;
        }
        
        .score-excellent { background: rgba(16, 185, 129, 0.2); color: var(--accent-green); }
        .score-good { background: rgba(6, 182, 212, 0.2); color: #06b6d4; }
        .score-fair { background: rgba(245, 158, 11, 0.2); color: var(--accent-amber); }
        .score-poor { background: rgba(239, 68, 68, 0.2); color: var(--accent-red); }
        
        table {
            width: 100%;
            border-collapse: collapse;
            margin-top: 1rem;
        }
        
        th, td {
            padding: 0.75rem 1rem;
            text-align: left;
            border-bottom: 1px solid var(--border);
        }
        
        th {
            color: var(--text-secondary);
            font-weight: 600;
            font-size: 0.85rem;
            text-transform: uppercase;
            letter-spacing: 0.05em;
        }
        
        tr:hover { background: rgba(99, 102, 241, 0.05); }
        
        .email-card {
            background: var(--bg-secondary);
            border: 1px solid var(--border);
            border-radius: 12px;
            padding: 1.5rem;
            margin-bottom: 1.5rem;
        }
        
        .email-card .card-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 1rem;
            padding-bottom: 0.75rem;
            border-bottom: 1px solid var(--border);
        }
        
        .email-text {
            background: rgba(0, 0, 0, 0.2);
            padding: 1rem;
            border-radius: 8px;
            font-size: 0.9rem;
            margin: 0.5rem 0;
            white-space: pre-wrap;
            max-height: 200px;
            overflow-y: auto;
        }
        
        .metrics-bar {
            display: flex;
            flex-wrap: wrap;
            gap: 0.5rem;
            margin: 1rem 0;
        }
        
        .metric-pill {
            padding: 0.3rem 0.7rem;
            border-radius: 8px;
            font-size: 0.8rem;
            background: rgba(99, 102, 241, 0.1);
            border: 1px solid rgba(99, 102, 241, 0.3);
        }
        
        .feedback-list {
            list-style: none;
            padding: 0;
        }
        
        .feedback-list li {
            padding: 0.4rem 0;
            padding-left: 1.5rem;
            position: relative;
            font-size: 0.9rem;
        }
        
        .feedback-list li::before {
            position: absolute;
            left: 0;
            font-size: 0.8rem;
        }
        
        .feedback-list.strengths li::before { content: "✅"; }
        .feedback-list.weaknesses li::before { content: "⚠️"; }
        .feedback-list.improvements li::before { content: "💡"; }
        
        .footer {
            text-align: center;
            padding: 2rem;
            color: var(--text-secondary);
            font-size: 0.85rem;
        }
        
        details {
            cursor: pointer;
        }
        
        details summary {
            font-weight: 600;
            color: var(--accent-purple);
            margin-bottom: 0.5rem;
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>📧 Email Response Evaluation Report</h1>
            <p class="subtitle">Generated on {{ timestamp }} · {{ aggregate.total_samples }} responses evaluated</p>
        </div>
        
        <!-- Aggregate Stats -->
        <div class="stats-grid">
            <div class="stat-card">
                <div class="value">{{ "%.1f"|format(aggregate.avg_composite_score) }}</div>
                <div class="label">Average Score</div>
            </div>
            <div class="stat-card">
                <div class="value">{{ "%.1f"|format(aggregate.median_composite_score) }}</div>
                <div class="label">Median Score</div>
            </div>
            <div class="stat-card">
                <div class="value">{{ "%.1f"|format(aggregate.max_composite_score) }}</div>
                <div class="label">Best Score</div>
            </div>
            <div class="stat-card">
                <div class="value">{{ "%.1f"|format(aggregate.min_composite_score) }}</div>
                <div class="label">Worst Score</div>
            </div>
            <div class="stat-card">
                <div class="value">{{ aggregate.total_samples }}</div>
                <div class="label">Total Evaluated</div>
            </div>
        </div>
        
        <!-- Per-Metric Averages -->
        <div class="section">
            <h2>📊 Metric Breakdown (Averages)</h2>
            <table>
                <thead>
                    <tr>
                        <th>Metric</th>
                        <th>Average</th>
                        <th>Median</th>
                        <th>Rating</th>
                    </tr>
                </thead>
                <tbody>
                    {% for metric, avg in aggregate.avg_scores.items() %}
                    <tr>
                        <td>{{ metric | replace("_", " ") | title }}</td>
                        <td>{{ "%.3f"|format(avg) }}</td>
                        <td>{{ "%.3f"|format(aggregate.median_scores.get(metric, 0)) }}</td>
                        <td>
                            {% if avg >= 0.8 %}
                            <span class="score-badge score-excellent">Excellent</span>
                            {% elif avg >= 0.6 %}
                            <span class="score-badge score-good">Good</span>
                            {% elif avg >= 0.4 %}
                            <span class="score-badge score-fair">Fair</span>
                            {% else %}
                            <span class="score-badge score-poor">Poor</span>
                            {% endif %}
                        </td>
                    </tr>
                    {% endfor %}
                </tbody>
            </table>
        </div>
        
        <!-- Category Breakdown -->
        {% if aggregate.category_scores %}
        <div class="section">
            <h2>📂 Category Performance</h2>
            <table>
                <thead>
                    <tr>
                        <th>Category</th>
                        <th>Count</th>
                        <th>Avg Score</th>
                        <th>Min</th>
                        <th>Max</th>
                    </tr>
                </thead>
                <tbody>
                    {% for cat, scores in aggregate.category_scores.items() %}
                    <tr>
                        <td>{{ cat }}</td>
                        <td>{{ scores.count }}</td>
                        <td>
                            {% if scores.avg_score >= 80 %}
                            <span class="score-badge score-excellent">{{ "%.1f"|format(scores.avg_score) }}</span>
                            {% elif scores.avg_score >= 60 %}
                            <span class="score-badge score-good">{{ "%.1f"|format(scores.avg_score) }}</span>
                            {% elif scores.avg_score >= 40 %}
                            <span class="score-badge score-fair">{{ "%.1f"|format(scores.avg_score) }}</span>
                            {% else %}
                            <span class="score-badge score-poor">{{ "%.1f"|format(scores.avg_score) }}</span>
                            {% endif %}
                        </td>
                        <td>{{ "%.1f"|format(scores.min_score) }}</td>
                        <td>{{ "%.1f"|format(scores.max_score) }}</td>
                    </tr>
                    {% endfor %}
                </tbody>
            </table>
        </div>
        {% endif %}
        
        <!-- Common Weaknesses -->
        {% if aggregate.common_weaknesses %}
        <div class="section">
            <h2>🔍 Common Weakness Patterns</h2>
            <table>
                <thead>
                    <tr>
                        <th>Weakness</th>
                        <th>Occurrences</th>
                        <th>% of Responses</th>
                    </tr>
                </thead>
                <tbody>
                    {% for w in aggregate.common_weaknesses[:10] %}
                    <tr>
                        <td>{{ w.weakness }}</td>
                        <td>{{ w.count }}</td>
                        <td>{{ w.percentage }}%</td>
                    </tr>
                    {% endfor %}
                </tbody>
            </table>
        </div>
        {% endif %}
        
        <!-- Per-Response Details -->
        <div class="section">
            <h2>📝 Individual Response Evaluations</h2>
            {% for report in reports[:50] %}
            <div class="email-card">
                <div class="card-header">
                    <div>
                        <strong>#{{ report.id or loop.index }}</strong>
                        {% if report.category %}
                        <span class="metric-pill">{{ report.category }}</span>
                        {% endif %}
                    </div>
                    <div>
                        {% if report.composite_score >= 80 %}
                        <span class="score-badge score-excellent">{{ "%.1f"|format(report.composite_score) }} / 100</span>
                        {% elif report.composite_score >= 60 %}
                        <span class="score-badge score-good">{{ "%.1f"|format(report.composite_score) }} / 100</span>
                        {% elif report.composite_score >= 40 %}
                        <span class="score-badge score-fair">{{ "%.1f"|format(report.composite_score) }} / 100</span>
                        {% else %}
                        <span class="score-badge score-poor">{{ "%.1f"|format(report.composite_score) }} / 100</span>
                        {% endif %}
                    </div>
                </div>
                
                <div class="metrics-bar">
                    <span class="metric-pill">Semantic: {{ "%.2f"|format(report.semantic_similarity_score) }}</span>
                    <span class="metric-pill">Intent: {{ "%.2f"|format(report.intent_coverage_score) }}</span>
                    <span class="metric-pill">Complete: {{ "%.2f"|format(report.completeness_score) }}</span>
                    <span class="metric-pill">Tone: {{ "%.2f"|format(report.tone_score) }}</span>
                    <span class="metric-pill">Grounding: {{ "%.2f"|format(report.grounding_score) }}</span>
                    <span class="metric-pill">Hallucination: {{ "%.2f"|format(report.hallucination_score) }}</span>
                </div>
                
                <details>
                    <summary>View Email & Response</summary>
                    <p><strong>📥 Incoming Email:</strong></p>
                    <div class="email-text">{{ report.incoming_email }}</div>
                    <p><strong>🤖 Generated Reply:</strong></p>
                    <div class="email-text">{{ report.generated_reply }}</div>
                    <p><strong>✅ Ground Truth Reply:</strong></p>
                    <div class="email-text">{{ report.ground_truth_reply }}</div>
                </details>
                
                {% if report.strengths %}
                <details>
                    <summary>Strengths & Weaknesses</summary>
                    {% if report.strengths %}
                    <ul class="feedback-list strengths">
                        {% for s in report.strengths %}<li>{{ s }}</li>{% endfor %}
                    </ul>
                    {% endif %}
                    {% if report.weaknesses %}
                    <ul class="feedback-list weaknesses">
                        {% for w in report.weaknesses %}<li>{{ w }}</li>{% endfor %}
                    </ul>
                    {% endif %}
                    {% if report.improvements %}
                    <ul class="feedback-list improvements">
                        {% for i in report.improvements %}<li>{{ i }}</li>{% endfor %}
                    </ul>
                    {% endif %}
                </details>
                {% endif %}
            </div>
            {% endfor %}
            
            {% if reports | length > 50 %}
            <p style="text-align: center; color: var(--text-secondary); padding: 1rem;">
                Showing 50 of {{ reports | length }} responses. See JSON/CSV for full data.
            </p>
            {% endif %}
        </div>
        
        <div class="footer">
            <p>AI Email Response Evaluation System · Generated {{ timestamp }}</p>
        </div>
    </div>
</body>
</html>"""
