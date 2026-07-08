"""
Streamlit Web Interface for the AI Email Response System.

Features:
- Email Response Generator with RAG
- Evaluation Dashboard with interactive charts
- Dataset Explorer

Usage:
    streamlit run app/streamlit_app.py
"""

from __future__ import annotations

import json
import logging
import sys
import time
from pathlib import Path

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import get_settings

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Page Configuration
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Email Response AI",
    page_icon="📧",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Custom CSS — Dark theme with glassmorphism
# ---------------------------------------------------------------------------

st.markdown("""
<style>
    /* Import Google Font */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
    
    /* Global overrides */
    .stApp {
        font-family: 'Inter', sans-serif;
    }
    
    /* Main header styling */
    .main-header {
        text-align: center;
        padding: 2rem 1rem;
        background: linear-gradient(135deg, rgba(99, 102, 241, 0.1), rgba(139, 92, 246, 0.1));
        border-radius: 20px;
        border: 1px solid rgba(99, 102, 241, 0.2);
        margin-bottom: 2rem;
        backdrop-filter: blur(10px);
    }
    
    .main-header h1 {
        font-size: 2.5rem;
        font-weight: 800;
        background: linear-gradient(135deg, #6366f1, #8b5cf6, #a855f7);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.5rem;
    }
    
    .main-header p {
        color: #a0a0c0;
        font-size: 1.1rem;
    }
    
    /* Glassmorphism card */
    .glass-card {
        background: rgba(30, 30, 74, 0.6);
        border: 1px solid rgba(99, 102, 241, 0.2);
        border-radius: 16px;
        padding: 1.5rem;
        backdrop-filter: blur(10px);
        margin-bottom: 1rem;
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    
    .glass-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 30px rgba(99, 102, 241, 0.15);
    }
    
    /* Score display */
    .score-display {
        text-align: center;
        padding: 1.5rem;
    }
    
    .score-value {
        font-size: 3.5rem;
        font-weight: 800;
        line-height: 1;
    }
    
    .score-label {
        font-size: 0.9rem;
        color: #a0a0c0;
        margin-top: 0.5rem;
    }
    
    .score-excellent { color: #10b981; }
    .score-good { color: #06b6d4; }
    .score-fair { color: #f59e0b; }
    .score-poor { color: #ef4444; }
    
    /* Metric badge */
    .metric-badge {
        display: inline-block;
        padding: 0.4rem 1rem;
        border-radius: 20px;
        font-size: 0.85rem;
        font-weight: 600;
        margin: 0.2rem;
    }
    
    .badge-green { background: rgba(16, 185, 129, 0.15); color: #10b981; border: 1px solid rgba(16, 185, 129, 0.3); }
    .badge-blue { background: rgba(6, 182, 212, 0.15); color: #06b6d4; border: 1px solid rgba(6, 182, 212, 0.3); }
    .badge-amber { background: rgba(245, 158, 11, 0.15); color: #f59e0b; border: 1px solid rgba(245, 158, 11, 0.3); }
    .badge-red { background: rgba(239, 68, 68, 0.15); color: #ef4444; border: 1px solid rgba(239, 68, 68, 0.3); }
    .badge-purple { background: rgba(139, 92, 246, 0.15); color: #a78bfa; border: 1px solid rgba(139, 92, 246, 0.3); }
    
    /* Email display box */
    .email-box {
        background: rgba(0, 0, 0, 0.2);
        border: 1px solid rgba(255, 255, 255, 0.05);
        border-radius: 12px;
        padding: 1.2rem;
        font-size: 0.9rem;
        line-height: 1.6;
        white-space: pre-wrap;
        margin: 0.5rem 0;
    }
    
    /* Feedback items */
    .feedback-item {
        padding: 0.5rem 0;
        padding-left: 1.5rem;
        position: relative;
        font-size: 0.9rem;
    }
    
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    
    /* Improve sidebar */
    .css-1d391kg {
        background: rgba(15, 15, 35, 0.95);
    }
    
    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 2rem;
    }
    
    .stTabs [data-baseweb="tab"] {
        font-weight: 600;
        font-size: 1rem;
    }
    
    /* Divider */
    .custom-divider {
        height: 1px;
        background: linear-gradient(90deg, transparent, rgba(99, 102, 241, 0.3), transparent);
        margin: 1.5rem 0;
    }
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Helper Functions
# ---------------------------------------------------------------------------

def get_score_class(score: float) -> str:
    """Get CSS class for score coloring."""
    if score >= 80:
        return "score-excellent"
    elif score >= 60:
        return "score-good"
    elif score >= 40:
        return "score-fair"
    return "score-poor"


def get_badge_class(score: float) -> str:
    """Get badge CSS class based on score."""
    if score >= 0.8:
        return "badge-green"
    elif score >= 0.6:
        return "badge-blue"
    elif score >= 0.4:
        return "badge-amber"
    return "badge-red"


@st.cache_resource
def load_pipeline():
    """Load the RAG pipeline (cached)."""
    try:
        from generator.rag_pipeline import RAGPipeline
        return RAGPipeline()
    except Exception as e:
        st.error(f"Failed to load pipeline: {e}")
        return None


@st.cache_resource
def load_evaluator():
    """Load the composite evaluator (cached)."""
    try:
        from evaluation.evaluator import CompositeEvaluator
        return CompositeEvaluator()
    except Exception as e:
        st.error(f"Failed to load evaluator: {e}")
        return None


def load_dataset() -> pd.DataFrame | None:
    """Load the email dataset."""
    settings = get_settings()
    path = settings.dataset_abs_path
    if path.exists():
        return pd.read_csv(path)
    return None


def load_dashboard_data() -> dict | None:
    """Load pre-computed dashboard data."""
    settings = get_settings()
    path = settings.reports_abs_path / "dashboard_data.json"
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return None


def create_radar_chart(labels: list[str], values: list[float], title: str = "") -> go.Figure:
    """Create a beautiful radar chart."""
    fig = go.Figure()
    
    fig.add_trace(go.Scatterpolar(
        r=values + [values[0]],  # Close the polygon
        theta=labels + [labels[0]],
        fill='toself',
        fillcolor='rgba(99, 102, 241, 0.15)',
        line=dict(color='#6366f1', width=2),
        marker=dict(size=8, color='#8b5cf6'),
        name='Score',
    ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 1],
                tickfont=dict(size=10, color='#666'),
                gridcolor='rgba(255,255,255,0.1)',
            ),
            angularaxis=dict(
                tickfont=dict(size=12, color='#a0a0c0'),
                gridcolor='rgba(255,255,255,0.1)',
            ),
            bgcolor='rgba(0,0,0,0)',
        ),
        showlegend=False,
        title=dict(text=title, font=dict(size=16, color='#e8e8f0')),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        height=400,
        margin=dict(l=80, r=80, t=60, b=40),
    )
    
    return fig


def create_gauge_chart(score: float, title: str = "Composite Score") -> go.Figure:
    """Create a gauge chart for the composite score."""
    color = "#10b981" if score >= 80 else "#06b6d4" if score >= 60 else "#f59e0b" if score >= 40 else "#ef4444"
    
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=score,
        number=dict(font=dict(size=48, color=color)),
        gauge=dict(
            axis=dict(range=[0, 100], tickwidth=2, tickcolor='#444'),
            bar=dict(color=color, thickness=0.3),
            bgcolor='rgba(30,30,74,0.6)',
            borderwidth=0,
            steps=[
                dict(range=[0, 40], color='rgba(239,68,68,0.1)'),
                dict(range=[40, 60], color='rgba(245,158,11,0.1)'),
                dict(range=[60, 80], color='rgba(6,182,212,0.1)'),
                dict(range=[80, 100], color='rgba(16,185,129,0.1)'),
            ],
        ),
        title=dict(text=title, font=dict(size=16, color='#a0a0c0')),
    ))
    
    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        height=300,
        margin=dict(l=30, r=30, t=60, b=20),
    )
    
    return fig


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

with st.sidebar:
    st.markdown("""
    <div style="text-align: center; padding: 1rem 0;">
        <h2 style="background: linear-gradient(135deg, #6366f1, #8b5cf6);
                    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
                    font-size: 1.5rem; font-weight: 800;">📧 Email AI</h2>
        <p style="color: #888; font-size: 0.85rem;">Response Generation & Evaluation</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)
    
    page = st.radio(
        "Navigation",
        ["🚀 Generate Response", "📊 Evaluation Dashboard", "📂 Dataset Explorer"],
        label_visibility="collapsed",
    )
    
    st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)
    
    # Settings
    st.markdown("### ⚙️ Settings")
    top_k = st.slider("Retrieved Examples (top-k)", 1, 15, 5)
    
    settings = get_settings()
    st.markdown(f"**Model:** `{settings.llm_model}`")
    st.markdown(f"**Embeddings:** `{settings.embedding_model}`")
    
    st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)
    
    st.markdown("""
    <div style="text-align: center; color: #555; font-size: 0.75rem; padding: 1rem 0;">
        Built with Streamlit • Powered by Cerebras
    </div>
    """, unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Page 1: Generate Response
# ---------------------------------------------------------------------------

if page == "🚀 Generate Response":
    st.markdown("""
    <div class="main-header">
        <h1>Email Response Generator</h1>
        <p>Paste an email below to generate an AI-powered response grounded in historical data</p>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("### 📥 Incoming Email")
        email_input = st.text_area(
            "Paste the email here",
            height=250,
            placeholder="Dear Support,\n\nI ordered product XYZ last week and it arrived damaged. I'd like to request a refund and also change my shipping address for future orders.\n\nThanks,\nJohn",
            label_visibility="collapsed",
        )
        
        category_filter = st.selectbox(
            "Category Filter (optional)",
            ["Auto-detect"] + [
                "Customer Support", "HR", "Internal Team", "Sales", "Refund",
                "Technical Issue", "Scheduling", "Product Questions", "Complaint",
                "Thank You", "Partnership", "Billing", "Interview", "Logistics",
            ],
        )
        
        reference_input = st.text_area(
            "Reference Response (optional, for evaluation)",
            height=150,
            placeholder="Paste a ground truth response here if you want to evaluate the generated response...",
        )
        
        generate_btn = st.button("🚀 Generate Response", type="primary", use_container_width=True)
    
    with col2:
        if generate_btn and email_input.strip():
            with st.spinner("🔄 Generating response..."):
                pipeline = load_pipeline()
                if pipeline:
                    try:
                        start_time = time.time()
                        cat = None if category_filter == "Auto-detect" else category_filter
                        result = pipeline.process(
                            email=email_input, top_k=top_k, category=cat
                        )
                        elapsed = (time.time() - start_time) * 1000
                        
                        st.markdown("### 🤖 Generated Response")
                        st.markdown(f'<div class="email-box">{result.generated_response}</div>',
                                    unsafe_allow_html=True)
                        
                        # Metadata
                        meta_col1, meta_col2, meta_col3 = st.columns(3)
                        meta_col1.metric("⏱️ Latency", f"{elapsed:.0f}ms")
                        meta_col2.metric("📚 Retrieved", f"{len(result.retrieved_examples)}")
                        meta_col3.metric("🤖 Model", result.model_used)
                        
                        # Retrieved examples
                        with st.expander("📚 Retrieved Similar Emails", expanded=False):
                            for i, ex in enumerate(result.retrieved_examples):
                                st.markdown(f"**Example {i+1}** (Similarity: {ex.get('similarity_score', 'N/A')})")
                                st.markdown(f"*Email:* {ex.get('email_text', '')[:200]}...")
                                st.markdown(f"*Reply:* {ex.get('reply_text', '')[:200]}...")
                                st.markdown("---")
                        
                        # Evaluation (if reference provided)
                        if reference_input.strip():
                            st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)
                            st.markdown("### 📊 Evaluation")
                            
                            with st.spinner("Evaluating response quality..."):
                                evaluator = load_evaluator()
                                if evaluator:
                                    eval_result = evaluator.evaluate_single(
                                        email=email_input,
                                        generated=result.generated_response,
                                        reference=reference_input,
                                    )
                                    
                                    # Gauge chart
                                    fig = create_gauge_chart(eval_result.composite_score)
                                    st.plotly_chart(fig, use_container_width=True)
                                    
                                    # Metric badges
                                    st.markdown("#### Metric Breakdown")
                                    metrics_html = ""
                                    for name, mr in eval_result.metric_results.items():
                                        badge_cls = get_badge_class(mr.score)
                                        metrics_html += f'<span class="metric-badge {badge_cls}">{name}: {mr.score:.2f}</span> '
                                    st.markdown(metrics_html, unsafe_allow_html=True)
                                    
                                    # Radar chart
                                    metric_names = list(eval_result.metric_results.keys())
                                    metric_values = [r.score for r in eval_result.metric_results.values()]
                                    fig = create_radar_chart(metric_names, metric_values, "Metric Radar")
                                    st.plotly_chart(fig, use_container_width=True)
                                    
                                    # Strengths & Weaknesses
                                    s_col, w_col = st.columns(2)
                                    with s_col:
                                        st.markdown("#### ✅ Strengths")
                                        for s in eval_result.strengths:
                                            st.markdown(f"- {s}")
                                    with w_col:
                                        st.markdown("#### ⚠️ Weaknesses")
                                        for w in eval_result.weaknesses:
                                            st.markdown(f"- {w}")
                                    
                                    if eval_result.improvements:
                                        st.markdown("#### 💡 Improvement Suggestions")
                                        for imp in eval_result.improvements:
                                            st.markdown(f"- {imp}")
                                    
                                    # Detailed reasoning
                                    with st.expander("🔍 Detailed Metric Reasoning"):
                                        for name, mr in eval_result.metric_results.items():
                                            st.markdown(f"**{name}** ({mr.score:.3f})")
                                            st.markdown(f"> {mr.reasoning}")
                                            st.markdown("---")
                        
                    except Exception as e:
                        st.error(f"Error: {e}")
                        logger.error(f"Generation error: {e}", exc_info=True)
                else:
                    st.error("Pipeline not available. Check your configuration.")
        
        elif generate_btn:
            st.warning("Please enter an email to generate a response.")


# ---------------------------------------------------------------------------
# Page 2: Evaluation Dashboard
# ---------------------------------------------------------------------------

elif page == "📊 Evaluation Dashboard":
    st.markdown("""
    <div class="main-header">
        <h1>Evaluation Dashboard</h1>
        <p>Aggregate performance metrics and analysis across all evaluated responses</p>
    </div>
    """, unsafe_allow_html=True)
    
    dashboard_data = load_dashboard_data()
    
    if dashboard_data is None:
        st.info("🔧 No evaluation data available yet. Run the evaluation experiment first:")
        st.code("python -m experiments.run_evaluation --sample-size 50", language="bash")
        st.markdown("This will generate responses, evaluate them, and produce the dashboard data.")
    else:
        summary = dashboard_data.get("summary", {})
        
        # Top-level KPI cards
        kpi_cols = st.columns(6)
        kpis = [
            ("📊 Avg Score", f"{summary.get('avg_score', 0):.1f}", ""),
            ("📈 Median", f"{summary.get('median_score', 0):.1f}", ""),
            ("🏆 Best", f"{summary.get('max_score', 0):.1f}", ""),
            ("📉 Worst", f"{summary.get('min_score', 0):.1f}", ""),
            ("📋 Samples", f"{summary.get('total_samples', 0)}", ""),
            ("⚠️ Failure Rate", f"{summary.get('failure_rate', 0):.1f}%", ""),
        ]
        
        for col, (label, value, delta) in zip(kpi_cols, kpis):
            col.metric(label, value, delta or None)
        
        st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)
        
        # Charts row
        chart_col1, chart_col2 = st.columns(2)
        
        with chart_col1:
            # Score Distribution
            dist_data = dashboard_data.get("score_distribution", {})
            if dist_data.get("counts"):
                fig = go.Figure(go.Bar(
                    x=dist_data["bin_labels"],
                    y=dist_data["counts"],
                    marker_color='rgba(99, 102, 241, 0.7)',
                    marker_line=dict(color='#6366f1', width=1),
                ))
                fig.update_layout(
                    title="Score Distribution",
                    xaxis_title="Score Range",
                    yaxis_title="Count",
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)',
                    font=dict(color='#a0a0c0'),
                    height=400,
                )
                fig.add_vline(
                    x=dist_data.get("mean", 0),
                    line_dash="dash",
                    line_color="#f59e0b",
                    annotation_text=f"Mean: {dist_data.get('mean', 0):.1f}",
                )
                st.plotly_chart(fig, use_container_width=True)
        
        with chart_col2:
            # Radar chart
            radar_data = dashboard_data.get("radar_overall", {})
            if radar_data.get("metrics"):
                fig = create_radar_chart(
                    radar_data["metrics"],
                    radar_data["values"],
                    "Overall Metric Profile"
                )
                st.plotly_chart(fig, use_container_width=True)
        
        st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)
        
        # Category Performance
        cat_data = dashboard_data.get("category_performance", {})
        if cat_data:
            st.markdown("### 📂 Category Performance")
            
            cat_df = pd.DataFrame([
                {"Category": cat, **scores}
                for cat, scores in cat_data.items()
            ]).sort_values("avg_score", ascending=False)
            
            fig = go.Figure(go.Bar(
                x=cat_df["Category"],
                y=cat_df["avg_score"],
                marker_color=[
                    '#10b981' if s >= 80 else '#06b6d4' if s >= 60 
                    else '#f59e0b' if s >= 40 else '#ef4444'
                    for s in cat_df["avg_score"]
                ],
                text=cat_df["avg_score"].round(1),
                textposition='outside',
            ))
            fig.update_layout(
                xaxis_title="Category",
                yaxis_title="Average Score",
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font=dict(color='#a0a0c0'),
                height=400,
                yaxis=dict(range=[0, 110]),
            )
            st.plotly_chart(fig, use_container_width=True)
        
        st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)
        
        # Per-metric distributions
        metric_dists = dashboard_data.get("metric_distributions", {})
        if metric_dists:
            st.markdown("### 📊 Per-Metric Score Distributions")
            
            dist_cols = st.columns(4)
            colors = ['#6366f1', '#8b5cf6', '#06b6d4', '#10b981', '#f59e0b', '#ef4444', '#ec4899', '#f97316']
            
            for i, (metric, values) in enumerate(metric_dists.items()):
                with dist_cols[i % 4]:
                    fig = go.Figure(go.Histogram(
                        x=values,
                        nbinsx=15,
                        marker_color=colors[i % len(colors)],
                        opacity=0.7,
                    ))
                    fig.update_layout(
                        title=metric.replace("_", " ").title(),
                        paper_bgcolor='rgba(0,0,0,0)',
                        plot_bgcolor='rgba(0,0,0,0)',
                        font=dict(color='#a0a0c0', size=10),
                        height=250,
                        margin=dict(l=20, r=20, t=40, b=20),
                        showlegend=False,
                        xaxis=dict(range=[0, 1]),
                    )
                    st.plotly_chart(fig, use_container_width=True)
        
        # Failure Analysis
        failure_data = dashboard_data.get("failure_analysis", {})
        if failure_data.get("failure_count", 0) > 0:
            st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)
            st.markdown("### 🔍 Failure Analysis")
            
            f_col1, f_col2 = st.columns(2)
            with f_col1:
                st.metric("Failed Responses", failure_data["failure_count"])
                st.metric("Failure Rate", f"{failure_data['failure_rate']}%")
                
                if failure_data.get("worst_metrics"):
                    st.markdown("**Weakest Metrics in Failures:**")
                    for wm in failure_data["worst_metrics"][:5]:
                        st.markdown(f"- {wm['metric']}: {wm['avg_score']:.3f}")
            
            with f_col2:
                if failure_data.get("common_issues"):
                    st.markdown("**Most Common Issues:**")
                    for issue in failure_data["common_issues"][:7]:
                        st.markdown(f"- {issue['issue']} ({issue['count']}x)")
        
        # Best & Worst Examples
        st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)
        best_tab, worst_tab = st.tabs(["🏆 Best Examples", "📉 Worst Examples"])
        
        with best_tab:
            best_examples = dashboard_data.get("best_examples", [])
            for ex in best_examples[:5]:
                with st.expander(f"Score: {ex.get('composite_score', 0):.1f} — {ex.get('category', 'N/A')}"):
                    st.markdown(f"**Email:** {ex.get('incoming_email', '')[:300]}")
                    st.markdown(f"**Generated:** {ex.get('generated_reply', '')[:300]}")
        
        with worst_tab:
            worst_examples = dashboard_data.get("worst_examples", [])
            for ex in worst_examples[:5]:
                with st.expander(f"Score: {ex.get('composite_score', 0):.1f} — {ex.get('category', 'N/A')}"):
                    st.markdown(f"**Email:** {ex.get('incoming_email', '')[:300]}")
                    st.markdown(f"**Generated:** {ex.get('generated_reply', '')[:300]}")
                    if ex.get("weaknesses"):
                        st.markdown("**Weaknesses:**")
                        for w in ex["weaknesses"]:
                            st.markdown(f"- {w}")


# ---------------------------------------------------------------------------
# Page 3: Dataset Explorer
# ---------------------------------------------------------------------------

elif page == "📂 Dataset Explorer":
    st.markdown("""
    <div class="main-header">
        <h1>Dataset Explorer</h1>
        <p>Browse and filter the email-response training dataset</p>
    </div>
    """, unsafe_allow_html=True)
    
    df = load_dataset()
    
    if df is None:
        st.info("📂 No dataset found. Generate it first:")
        st.code("python -m dataset.synthetic_generation --count 2000", language="bash")
    else:
        # Filters
        filter_col1, filter_col2, filter_col3 = st.columns(3)
        
        with filter_col1:
            categories = ["All"] + sorted(df["category"].unique().tolist())
            selected_category = st.selectbox("Category", categories)
        
        with filter_col2:
            if "sentiment" in df.columns:
                sentiments = ["All"] + sorted(df["sentiment"].dropna().unique().tolist())
                selected_sentiment = st.selectbox("Sentiment", sentiments)
            else:
                selected_sentiment = "All"
        
        with filter_col3:
            if "urgency" in df.columns:
                urgencies = ["All"] + sorted(df["urgency"].dropna().unique().tolist())
                selected_urgency = st.selectbox("Urgency", urgencies)
            else:
                selected_urgency = "All"
        
        # Apply filters
        filtered_df = df.copy()
        if selected_category != "All":
            filtered_df = filtered_df[filtered_df["category"] == selected_category]
        if selected_sentiment != "All" and "sentiment" in filtered_df.columns:
            filtered_df = filtered_df[filtered_df["sentiment"] == selected_sentiment]
        if selected_urgency != "All" and "urgency" in filtered_df.columns:
            filtered_df = filtered_df[filtered_df["urgency"] == selected_urgency]
        
        st.markdown(f"**Showing {len(filtered_df)} of {len(df)} email pairs**")
        
        st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)
        
        # Stats row
        stat_cols = st.columns(4)
        stat_cols[0].metric("Total Pairs", len(df))
        stat_cols[1].metric("Categories", df["category"].nunique())
        if "word_count_email" in df.columns:
            stat_cols[2].metric("Avg Email Length", f"{df['word_count_email'].mean():.0f} words")
            stat_cols[3].metric("Avg Reply Length", f"{df['word_count_reply'].mean():.0f} words")
        
        st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)
        
        # Category distribution chart
        if len(df) > 0:
            cat_counts = df["category"].value_counts()
            fig = go.Figure(go.Bar(
                x=cat_counts.values,
                y=cat_counts.index,
                orientation='h',
                marker_color='rgba(99, 102, 241, 0.7)',
                marker_line=dict(color='#6366f1', width=1),
            ))
            fig.update_layout(
                title="Category Distribution",
                xaxis_title="Count",
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font=dict(color='#a0a0c0'),
                height=400,
            )
            st.plotly_chart(fig, use_container_width=True)
        
        st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)
        
        # Data table
        st.markdown("### 📋 Email-Response Pairs")
        
        # Paginate
        page_size = 10
        total_pages = max(1, len(filtered_df) // page_size + (1 if len(filtered_df) % page_size else 0))
        current_page = st.number_input("Page", min_value=1, max_value=total_pages, value=1)
        
        start_idx = (current_page - 1) * page_size
        end_idx = start_idx + page_size
        page_df = filtered_df.iloc[start_idx:end_idx]
        
        for _, row in page_df.iterrows():
            with st.expander(f"[{row.get('category', 'N/A')}] {str(row.get('email', ''))[:80]}..."):
                e_col, r_col = st.columns(2)
                with e_col:
                    st.markdown("**📥 Email:**")
                    st.markdown(f'<div class="email-box">{row.get("email", "")}</div>',
                                unsafe_allow_html=True)
                with r_col:
                    st.markdown("**📤 Reply:**")
                    st.markdown(f'<div class="email-box">{row.get("reply", "")}</div>',
                                unsafe_allow_html=True)
                
                meta_cols = st.columns(4)
                meta_cols[0].markdown(f"**Category:** {row.get('category', 'N/A')}")
                meta_cols[1].markdown(f"**Sentiment:** {row.get('sentiment', 'N/A')}")
                meta_cols[2].markdown(f"**Urgency:** {row.get('urgency', 'N/A')}")
                meta_cols[3].markdown(f"**Sender:** {row.get('sender_type', 'N/A')}")
