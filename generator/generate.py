"""
Core response-generation logic for the AI Email Response System.

Provides the ``generate_response`` function that orchestrates prompt
building → LLM call → result packaging, as well as a ``generate_batch``
helper for processing many emails with per-item error handling.
"""

from __future__ import annotations

import logging
import time
from typing import Any

from pydantic import BaseModel, Field
from tqdm import tqdm

from generator.llm import CerebrasLLM, LLMError, estimate_tokens
from generator.prompt_templates import build_generation_prompt

logger = logging.getLogger(__name__)


# ======================================================================
# Result model
# ======================================================================

class GeneratedResponse(BaseModel):
    """Structured result from a single response-generation call."""

    response_text: str = Field(
        ..., description="The generated reply text."
    )
    model_used: str = Field(
        ..., description="LLM model identifier used for generation."
    )
    tokens_used: int = Field(
        default=0, description="Estimated total tokens consumed."
    )
    latency_ms: float = Field(
        default=0.0, description="Wall-clock latency in milliseconds."
    )
    retrieved_context: list[dict[str, Any]] = Field(
        default_factory=list,
        description="Summary of retrieved examples supplied as context.",
    )


# ======================================================================
# Single generation
# ======================================================================

def generate_response(
    email: str,
    retrieved_examples: list[Any],
    category: str | None = None,
    *,
    llm: CerebrasLLM | None = None,
    temperature: float | None = None,
    max_tokens: int | None = None,
) -> GeneratedResponse:
    """Generate a reply to *email* using retrieved context.

    Args:
        email:              Incoming customer email text.
        retrieved_examples: List of similar historical email objects
                            (e.g. ``RetrievedEmail`` or plain dicts).
        category:           Optional category label for additional context.
        llm:                Pre-initialised LLM client (created on-the-fly
                            if ``None``).
        temperature:        Override sampling temperature for this call.
        max_tokens:         Override max completion tokens for this call.

    Returns:
        A :class:`GeneratedResponse` with the reply and metadata.

    Raises:
        LLMError: Propagated if the LLM call fails after retries.
    """
    if llm is None:
        llm = CerebrasLLM()

    # 1. Build prompt ─────────────────────────────────────────────────
    messages = build_generation_prompt(email, retrieved_examples, category)
    logger.info("Generating response for email (len=%d) with %d examples", len(email), len(retrieved_examples))

    # 2. Call LLM ─────────────────────────────────────────────────────
    start = time.perf_counter()
    response_text = llm.generate(
        messages,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    latency_ms = (time.perf_counter() - start) * 1000

    # 3. Token estimation ─────────────────────────────────────────────
    prompt_tokens = sum(estimate_tokens(m["content"]) for m in messages)
    completion_tokens = estimate_tokens(response_text)
    total_tokens = prompt_tokens + completion_tokens

    # 4. Build context summary ────────────────────────────────────────
    context_summary = _summarise_examples(retrieved_examples)

    result = GeneratedResponse(
        response_text=response_text,
        model_used=llm.model,
        tokens_used=total_tokens,
        latency_ms=round(latency_ms, 2),
        retrieved_context=context_summary,
    )

    logger.info(
        "Response generated — tokens=%d, latency=%.0f ms, response_len=%d",
        total_tokens,
        latency_ms,
        len(response_text),
    )
    return result


# ======================================================================
# Batch generation
# ======================================================================

def generate_batch(
    emails: list[str],
    retrieved_examples_list: list[list[Any]],
    categories: list[str | None] | None = None,
    *,
    llm: CerebrasLLM | None = None,
    temperature: float | None = None,
    max_tokens: int | None = None,
    show_progress: bool = True,
) -> list[GeneratedResponse | None]:
    """Generate replies for a batch of emails.

    Processing is sequential.  Individual failures are caught so that one
    bad item does not abort the entire batch.

    Args:
        emails:                  List of incoming email texts.
        retrieved_examples_list: Parallel list of retrieved-example lists,
                                 one per email.
        categories:              Optional parallel list of category labels.
        llm:                     Shared LLM client (created once if ``None``).
        temperature:             Override sampling temperature.
        max_tokens:              Override max completion tokens.
        show_progress:           Display a ``tqdm`` progress bar.

    Returns:
        A list the same length as *emails*.  Each element is either a
        :class:`GeneratedResponse` or ``None`` if that item failed.
    """
    if len(emails) != len(retrieved_examples_list):
        raise ValueError(
            f"Length mismatch: {len(emails)} emails vs "
            f"{len(retrieved_examples_list)} example lists."
        )
    if categories and len(categories) != len(emails):
        raise ValueError(
            f"Length mismatch: {len(emails)} emails vs "
            f"{len(categories)} categories."
        )

    if llm is None:
        llm = CerebrasLLM()

    results: list[GeneratedResponse | None] = []
    iterator = enumerate(zip(emails, retrieved_examples_list))
    if show_progress:
        iterator = tqdm(  # type: ignore[assignment]
            iterator,
            total=len(emails),
            desc="Generating responses",
            unit="email",
        )

    for idx, (email, examples) in iterator:
        cat = categories[idx] if categories else None
        try:
            result = generate_response(
                email,
                examples,
                category=cat,
                llm=llm,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            results.append(result)
        except LLMError as exc:
            logger.error("Failed to generate response for item %d: %s", idx, exc)
            results.append(None)
        except Exception as exc:
            logger.error(
                "Unexpected error for item %d: %s", idx, exc, exc_info=True
            )
            results.append(None)

    succeeded = sum(1 for r in results if r is not None)
    logger.info(
        "Batch complete: %d/%d succeeded, %d failed",
        succeeded,
        len(emails),
        len(emails) - succeeded,
    )
    return results


# ======================================================================
# Private helpers
# ======================================================================

def _summarise_examples(examples: list[Any]) -> list[dict[str, Any]]:
    """Create a compact dict summary of each retrieved example."""
    summaries: list[dict[str, Any]] = []
    for ex in examples:
        if isinstance(ex, dict):
            summaries.append({
                "incoming_email": _truncate(ex.get("incoming_email", ""), 200),
                "response": _truncate(ex.get("response", ""), 200),
                "score": ex.get("score", 0.0),
            })
        else:
            summaries.append({
                "incoming_email": _truncate(getattr(ex, "incoming_email", ""), 200),
                "response": _truncate(getattr(ex, "response", ""), 200),
                "score": getattr(ex, "score", 0.0),
            })
    return summaries


def _truncate(text: str, max_len: int) -> str:
    """Truncate *text* to *max_len* chars, appending '…' if shortened."""
    if len(text) <= max_len:
        return text
    return text[: max_len - 1] + "…"
