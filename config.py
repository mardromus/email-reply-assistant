"""
Central configuration for the AI Email Response System.

Loads settings from environment variables / .env file using Pydantic Settings.
All modules should import configuration from here rather than reading env vars directly.
"""

from __future__ import annotations

import os
from pathlib import Path
from functools import lru_cache
from typing import Optional

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


# ---------------------------------------------------------------------------
# Project root — resolved relative to this file so imports work everywhere
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent


class Settings(BaseSettings):
    """Application-wide settings loaded from .env and environment variables."""

    model_config = SettingsConfigDict(
        env_file=str(PROJECT_ROOT / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # ── LLM Provider ──────────────────────────────────────────────────────
    cerebras_api_key: str = Field(default="", description="Cerebras API key")
    llm_provider: str = Field(default="cerebras", description="LLM provider name")
    llm_model: str = Field(default="gemma-4-31b", description="LLM model identifier")
    llm_temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    llm_max_tokens: int = Field(default=1024, ge=1)

    # ── Cerebras API ──────────────────────────────────────────────────────
    cerebras_base_url: str = Field(
        default="https://api.cerebras.ai/v1",
        description="Cerebras OpenAI-compatible base URL",
    )

    # ── Embedding ─────────────────────────────────────────────────────────
    embedding_model: str = Field(
        default="all-MiniLM-L6-v2",
        description="Sentence-transformers model for embeddings",
    )

    # ── Vector Store ──────────────────────────────────────────────────────
    chroma_persist_dir: str = Field(default="./data/chroma")
    chroma_collection_name: str = Field(default="email_responses")

    # ── Retrieval ─────────────────────────────────────────────────────────
    retrieval_top_k: int = Field(default=5, ge=1, le=20)

    # ── Evaluation ────────────────────────────────────────────────────────
    eval_llm_model: str = Field(default="gemma-4-31b")
    eval_llm_temperature: float = Field(default=0.1, ge=0.0, le=1.0)

    # ── Evaluation Weights ────────────────────────────────────────────────
    weight_semantic_similarity: float = Field(default=0.20)
    weight_intent_coverage: float = Field(default=0.25)
    weight_completeness: float = Field(default=0.20)
    weight_grounding: float = Field(default=0.15)
    weight_tone: float = Field(default=0.10)
    weight_hallucination: float = Field(default=0.10)

    # ── Paths ─────────────────────────────────────────────────────────────
    dataset_path: str = Field(default="./dataset/emails.csv")
    outputs_dir: str = Field(default="./outputs")
    reports_dir: str = Field(default="./outputs/reports")

    # ── Logging ───────────────────────────────────────────────────────────
    log_level: str = Field(default="INFO")

    # ── Computed properties ───────────────────────────────────────────────
    @property
    def dataset_abs_path(self) -> Path:
        return (PROJECT_ROOT / self.dataset_path).resolve()

    @property
    def outputs_abs_path(self) -> Path:
        return (PROJECT_ROOT / self.outputs_dir).resolve()

    @property
    def reports_abs_path(self) -> Path:
        return (PROJECT_ROOT / self.reports_dir).resolve()

    @property
    def chroma_abs_path(self) -> Path:
        return (PROJECT_ROOT / self.chroma_persist_dir).resolve()

    @property
    def evaluation_weights(self) -> dict[str, float]:
        """Return evaluation dimension weights as a dictionary."""
        return {
            "semantic_similarity": self.weight_semantic_similarity,
            "intent_coverage": self.weight_intent_coverage,
            "completeness": self.weight_completeness,
            "grounding": self.weight_grounding,
            "tone": self.weight_tone,
            "hallucination": self.weight_hallucination,
        }

    def ensure_directories(self) -> None:
        """Create output directories if they don't exist."""
        self.outputs_abs_path.mkdir(parents=True, exist_ok=True)
        self.reports_abs_path.mkdir(parents=True, exist_ok=True)
        self.chroma_abs_path.mkdir(parents=True, exist_ok=True)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Get cached application settings (singleton)."""
    settings = Settings()
    settings.ensure_directories()
    return settings
