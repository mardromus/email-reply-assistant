"""
End-to-end Retrieval-Augmented Generation (RAG) pipeline.

Orchestrates:
  1. Embedding-based retrieval of similar historical emails.
  2. Prompt construction with retrieved context.
  3. LLM-based response generation.
  4. Result packaging with full traceability.
"""

from __future__ import annotations

import logging
import time
from typing import Any

from pydantic import BaseModel, Field
from tqdm import tqdm

from config import get_settings
from generator.generate import generate_response, GeneratedResponse
from generator.llm import CerebrasLLM, LLMError

logger = logging.getLogger(__name__)


# ======================================================================
# Pipeline result model
# ======================================================================

class PipelineResult(BaseModel):
    """Complete trace of a single RAG pipeline invocation."""

    email: str = Field(
        ..., description="The incoming customer email."
    )
    generated_response: str = Field(
        ..., description="The generated reply text."
    )
    retrieved_examples: list[dict[str, Any]] = Field(
        default_factory=list,
        description="Retrieved historical examples used as context.",
    )
    model_used: str = Field(
        ..., description="LLM model identifier."
    )
    latency_ms: float = Field(
        default=0.0,
        description="Total wall-clock latency (retrieval + generation) in ms.",
    )
    tokens_used: int = Field(
        default=0, description="Estimated total tokens consumed."
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Arbitrary metadata (category, timings, errors, etc.).",
    )


# ======================================================================
# RAGPipeline
# ======================================================================

class RAGPipeline:
    """End-to-end RAG pipeline: retrieve → build prompt → generate.

    The pipeline lazily imports the retrieval layer so the module can be
    loaded even when the retrieval package hasn't been installed yet.
    When the retrieval layer is unavailable, the pipeline gracefully
    falls back to generation without retrieved context.

    Args:
        llm:    Pre-configured :class:`CerebrasLLM` (created with
                project defaults when ``None``).
        top_k:  Default number of examples to retrieve.
    """

    def __init__(
        self,
        llm: CerebrasLLM | None = None,
        top_k: int | None = None,
    ) -> None:
        settings = get_settings()
        self._top_k = top_k or settings.retrieval_top_k
        self._llm = llm or CerebrasLLM()

        # Lazy-load retrieval components so the generator module is
        # importable even before the retrieval package is fully built.
        self._embedding_model: Any | None = None
        self._vector_store: Any | None = None
        self._retrieval_available: bool = False

        self._init_retrieval()

        logger.info(
            "RAGPipeline initialised — model=%s, top_k=%d, retrieval=%s",
            self._llm.model,
            self._top_k,
            "available" if self._retrieval_available else "unavailable",
        )

    # ------------------------------------------------------------------
    # Retrieval bootstrap
    # ------------------------------------------------------------------
    def _init_retrieval(self) -> None:
        """Attempt to initialise embedding model and vector store."""
        try:
            from retrieval.embedding import get_embedding_model  # type: ignore[import-untyped]
            from retrieval.search import EmailVectorStore  # type: ignore[import-untyped]

            self._embedding_model = get_embedding_model()
            self._vector_store = EmailVectorStore(
                embedding_model=self._embedding_model,
            )
            self._retrieval_available = True
            logger.info("Retrieval components loaded successfully.")
        except ImportError as exc:
            logger.warning(
                "Retrieval modules not available (%s). "
                "Pipeline will generate without retrieved context.",
                exc,
            )
        except Exception as exc:
            logger.warning(
                "Failed to initialise retrieval (%s). "
                "Pipeline will generate without retrieved context.",
                exc,
            )

    # ------------------------------------------------------------------
    # Retrieval helper
    # ------------------------------------------------------------------
    def _retrieve(
        self,
        email: str,
        top_k: int | None = None,
        category: str | None = None,
    ) -> list[Any]:
        """Retrieve similar historical emails from the vector store.

        Returns an empty list if the retrieval layer is unavailable.
        """
        if not self._retrieval_available or self._vector_store is None:
            logger.debug("Retrieval unavailable — returning empty context.")
            return []

        k = top_k or self._top_k
        try:
            from retrieval.search import retrieve_similar  # type: ignore[import-untyped]

            results = retrieve_similar(
                query=email,
                vector_store=self._vector_store,
                top_k=k,
                category=category,
            )
            logger.info("Retrieved %d similar examples for query.", len(results))
            return results  # type: ignore[return-value]
        except Exception as exc:
            logger.error("Retrieval failed: %s", exc, exc_info=True)
            return []

    # ------------------------------------------------------------------
    # Single processing
    # ------------------------------------------------------------------
    def process(
        self,
        email: str,
        top_k: int | None = None,
        category: str | None = None,
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> PipelineResult:
        """Run the full RAG pipeline for a single email.

        Steps:
            1. Retrieve similar historical emails from the vector store.
            2. Build a RAG-augmented prompt with retrieved examples.
            3. Generate a response via the LLM.
            4. Package everything into a :class:`PipelineResult`.

        Args:
            email:       Incoming customer email text.
            top_k:       Number of examples to retrieve (overrides default).
            category:    Optional category label.
            temperature: Override sampling temperature.
            max_tokens:  Override max completion tokens.

        Returns:
            A :class:`PipelineResult` with the reply and full trace.

        Raises:
            LLMError: If generation fails after retries.
        """
        total_start = time.perf_counter()

        # Step 1 — Retrieve ─────────────────────────────────────────────
        retrieval_start = time.perf_counter()
        retrieved = self._retrieve(email, top_k=top_k, category=category)
        retrieval_ms = (time.perf_counter() - retrieval_start) * 1000

        # Step 2 + 3 — Build prompt & generate ──────────────────────────
        gen_result: GeneratedResponse = generate_response(
            email=email,
            retrieved_examples=retrieved,
            category=category,
            llm=self._llm,
            temperature=temperature,
            max_tokens=max_tokens,
        )

        total_ms = (time.perf_counter() - total_start) * 1000

        # Step 4 — Package ──────────────────────────────────────────────
        retrieved_dicts = self._examples_to_dicts(retrieved)

        result = PipelineResult(
            email=email,
            generated_response=gen_result.response_text,
            retrieved_examples=retrieved_dicts,
            model_used=gen_result.model_used,
            latency_ms=round(total_ms, 2),
            tokens_used=gen_result.tokens_used,
            metadata={
                "category": category,
                "top_k": top_k or self._top_k,
                "retrieval_latency_ms": round(retrieval_ms, 2),
                "generation_latency_ms": round(gen_result.latency_ms, 2),
                "num_examples_retrieved": len(retrieved),
            },
        )

        logger.info(
            "Pipeline complete — total=%.0f ms (retrieval=%.0f ms, gen=%.0f ms), "
            "tokens=%d, examples=%d",
            total_ms,
            retrieval_ms,
            gen_result.latency_ms,
            gen_result.tokens_used,
            len(retrieved),
        )
        return result

    # ------------------------------------------------------------------
    # Batch processing
    # ------------------------------------------------------------------
    def process_batch(
        self,
        emails: list[str],
        top_k: int | None = None,
        categories: list[str | None] | None = None,
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
        show_progress: bool = True,
    ) -> list[PipelineResult | None]:
        """Run the pipeline for a list of emails.

        Each item is processed independently — a failure on one email does
        not abort the rest of the batch.

        Args:
            emails:        List of customer email texts.
            top_k:         Number of examples to retrieve per email.
            categories:    Optional parallel list of category labels.
            temperature:   Override sampling temperature.
            max_tokens:    Override max completion tokens.
            show_progress: Display a ``tqdm`` progress bar.

        Returns:
            A list the same length as *emails*.  Each element is a
            :class:`PipelineResult` or ``None`` on failure.
        """
        if categories and len(categories) != len(emails):
            raise ValueError(
                f"Length mismatch: {len(emails)} emails vs "
                f"{len(categories)} categories."
            )

        results: list[PipelineResult | None] = []

        iterator: Any = enumerate(emails)
        if show_progress:
            iterator = tqdm(
                iterator,
                total=len(emails),
                desc="RAG pipeline",
                unit="email",
            )

        for idx, email in iterator:
            cat = categories[idx] if categories else None
            try:
                result = self.process(
                    email=email,
                    top_k=top_k,
                    category=cat,
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
                results.append(result)
            except LLMError as exc:
                logger.error("Pipeline failed for item %d: %s", idx, exc)
                results.append(None)
            except Exception as exc:
                logger.error(
                    "Unexpected pipeline error for item %d: %s",
                    idx,
                    exc,
                    exc_info=True,
                )
                results.append(None)

        succeeded = sum(1 for r in results if r is not None)
        logger.info(
            "Batch pipeline complete: %d/%d succeeded, %d failed",
            succeeded,
            len(emails),
            len(emails) - succeeded,
        )
        return results

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------
    def get_llm_usage(self) -> dict[str, int]:
        """Return cumulative LLM token usage across all pipeline calls."""
        return self._llm.get_cumulative_usage()

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _examples_to_dicts(examples: list[Any]) -> list[dict[str, Any]]:
        """Convert retrieved example objects to serialisable dicts."""
        dicts: list[dict[str, Any]] = []
        for ex in examples:
            if isinstance(ex, dict):
                dicts.append(ex)
            else:
                # Duck-type: pull known attributes.
                dicts.append({
                    "incoming_email": getattr(ex, "incoming_email", ""),
                    "response": getattr(ex, "response", ""),
                    "score": getattr(ex, "score", 0.0),
                    "metadata": getattr(ex, "metadata", {}),
                })
        return dicts

    def __repr__(self) -> str:
        return (
            f"RAGPipeline(model={self._llm.model!r}, top_k={self._top_k}, "
            f"retrieval={'yes' if self._retrieval_available else 'no'})"
        )
