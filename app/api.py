"""
FastAPI backend for the AI Email Response System.

Provides REST API endpoints for email response generation and evaluation.

Usage:
    uvicorn app.api:app --reload --port 8000
"""

from __future__ import annotations

import logging
import sys
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import get_settings

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Lazy-loaded global instances (initialized at startup)
# ---------------------------------------------------------------------------
_pipeline = None
_evaluator = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize heavy resources on startup."""
    global _pipeline, _evaluator
    logger.info("Initializing API resources...")
    
    try:
        from generator.rag_pipeline import RAGPipeline
        from evaluation.evaluator import CompositeEvaluator
        
        _pipeline = RAGPipeline()
        _evaluator = CompositeEvaluator()
        logger.info("API resources initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize resources: {e}")
        logger.warning("API will start but some endpoints may not work")
    
    yield
    
    logger.info("Shutting down API resources")


# ---------------------------------------------------------------------------
# FastAPI App
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Email Response AI",
    description="AI-powered email response generation and evaluation system",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Request/Response Models
# ---------------------------------------------------------------------------

class GenerateRequest(BaseModel):
    email: str = Field(..., description="Incoming email text to respond to")
    top_k: int = Field(default=5, ge=1, le=20, description="Number of similar emails to retrieve")
    category: str | None = Field(default=None, description="Optional email category filter")


class GenerateResponse(BaseModel):
    generated_response: str
    retrieved_examples: list[dict[str, Any]]
    model_used: str
    latency_ms: float
    tokens_used: int


class EvaluateRequest(BaseModel):
    email: str = Field(..., description="Original incoming email")
    generated: str = Field(..., description="Generated response to evaluate")
    reference: str = Field(..., description="Ground truth / reference response")
    context: list[str] | None = Field(default=None, description="Retrieved context used")


class EvaluateResponse(BaseModel):
    composite_score: float
    metric_scores: dict[str, float]
    metric_reasoning: dict[str, str]
    strengths: list[str]
    weaknesses: list[str]
    improvements: list[str]


class FullPipelineRequest(BaseModel):
    email: str = Field(..., description="Incoming email to process")
    reference: str | None = Field(default=None, description="Optional reference response for evaluation")
    top_k: int = Field(default=5, ge=1, le=20)
    category: str | None = None


class FullPipelineResponse(BaseModel):
    generated_response: str
    retrieved_examples: list[dict[str, Any]]
    evaluation: EvaluateResponse | None = None
    model_used: str
    latency_ms: float


class HealthResponse(BaseModel):
    status: str
    pipeline_ready: bool
    evaluator_ready: bool
    version: str


class StatsResponse(BaseModel):
    vector_store_count: int
    model_name: str
    embedding_model: str
    available_categories: list[str]


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health", response_model=HealthResponse, tags=["System"])
async def health_check():
    """Check API health status."""
    return HealthResponse(
        status="healthy",
        pipeline_ready=_pipeline is not None,
        evaluator_ready=_evaluator is not None,
        version="1.0.0",
    )


@app.post("/generate", response_model=GenerateResponse, tags=["Generation"])
async def generate_response(request: GenerateRequest):
    """Generate an email response using the RAG pipeline."""
    if _pipeline is None:
        raise HTTPException(status_code=503, detail="Pipeline not initialized")
    
    try:
        start = time.time()
        result = _pipeline.process(
            email=request.email,
            top_k=request.top_k,
            category=request.category,
        )
        
        return GenerateResponse(
            generated_response=result.generated_response,
            retrieved_examples=result.retrieved_examples,
            model_used=result.model_used,
            latency_ms=result.latency_ms,
            tokens_used=result.tokens_used,
        )
    except Exception as e:
        logger.error(f"Generation error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/evaluate", response_model=EvaluateResponse, tags=["Evaluation"])
async def evaluate_response(request: EvaluateRequest):
    """Evaluate a generated email response against a reference."""
    if _evaluator is None:
        raise HTTPException(status_code=503, detail="Evaluator not initialized")
    
    try:
        eval_result = _evaluator.evaluate_single(
            email=request.email,
            generated=request.generated,
            reference=request.reference,
            context=request.context,
        )
        
        metric_scores = {
            name: round(r.score, 3)
            for name, r in eval_result.metric_results.items()
        }
        metric_reasoning = {
            name: r.reasoning
            for name, r in eval_result.metric_results.items()
        }
        
        return EvaluateResponse(
            composite_score=round(eval_result.composite_score, 2),
            metric_scores=metric_scores,
            metric_reasoning=metric_reasoning,
            strengths=eval_result.strengths,
            weaknesses=eval_result.weaknesses,
            improvements=eval_result.improvements,
        )
    except Exception as e:
        logger.error(f"Evaluation error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/pipeline", response_model=FullPipelineResponse, tags=["Pipeline"])
async def full_pipeline(request: FullPipelineRequest):
    """Run the full pipeline: retrieve → generate → evaluate."""
    if _pipeline is None:
        raise HTTPException(status_code=503, detail="Pipeline not initialized")
    
    try:
        # Generate
        result = _pipeline.process(
            email=request.email,
            top_k=request.top_k,
            category=request.category,
        )
        
        # Evaluate (if reference provided)
        evaluation = None
        if request.reference and _evaluator:
            eval_result = _evaluator.evaluate_single(
                email=request.email,
                generated=result.generated_response,
                reference=request.reference,
            )
            evaluation = EvaluateResponse(
                composite_score=round(eval_result.composite_score, 2),
                metric_scores={
                    name: round(r.score, 3)
                    for name, r in eval_result.metric_results.items()
                },
                metric_reasoning={
                    name: r.reasoning
                    for name, r in eval_result.metric_results.items()
                },
                strengths=eval_result.strengths,
                weaknesses=eval_result.weaknesses,
                improvements=eval_result.improvements,
            )
        
        return FullPipelineResponse(
            generated_response=result.generated_response,
            retrieved_examples=result.retrieved_examples,
            evaluation=evaluation,
            model_used=result.model_used,
            latency_ms=result.latency_ms,
        )
    except Exception as e:
        logger.error(f"Pipeline error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/stats", response_model=StatsResponse, tags=["System"])
async def get_stats():
    """Get system statistics."""
    settings = get_settings()
    
    vector_count = 0
    categories = []
    if _pipeline:
        try:
            vector_count = _pipeline.vector_store.get_collection_count()
        except Exception:
            pass
    
    return StatsResponse(
        vector_store_count=vector_count,
        model_name=settings.llm_model,
        embedding_model=settings.embedding_model,
        available_categories=[
            "Customer Support", "HR", "Internal Team", "Sales", "Refund",
            "Technical Issue", "Scheduling", "Product Questions", "Complaint",
            "Thank You", "Partnership", "Billing", "Interview", "Logistics",
        ],
    )
