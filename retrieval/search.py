"""
Search module for the AI Email Response System.

Provides a high-level ``retrieve_similar`` function that finds emails
similar to a query string, with optional category filtering.  Results
are returned as validated ``RetrievedEmail`` Pydantic models sorted
by descending similarity score.

Usage::

    from retrieval.search import retrieve_similar

    results = retrieve_similar("I need a refund for order #12345", top_k=5)
    for r in results:
        print(r.similarity_score, r.email_text[:80])
"""

from __future__ import annotations

import logging
from typing import Any

from pydantic import BaseModel, Field

from config import get_settings
from retrieval.embedding import get_embedding_model
from retrieval.vector_store import EmailVectorStore

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Pydantic result model
# ---------------------------------------------------------------------------

class RetrievedEmail(BaseModel):
    """A single search result from the vector store.

    Attributes:
        email_id: Unique identifier for the stored email.
        email_text: The original incoming email body.
        reply_text: The ideal reply associated with this email.
        similarity_score: Cosine similarity to the query (higher = more
            similar). Range roughly ``[0, 1]`` for normalised embeddings.
        metadata: Additional metadata (category, urgency, sentiment, …).
    """

    email_id: str = Field(..., description="Unique email identifier")
    email_text: str = Field(..., description="Original incoming email body")
    reply_text: str = Field(..., description="Associated ideal reply")
    similarity_score: float = Field(
        ...,
        ge=-1.0,
        le=1.0,
        description="Cosine similarity score",
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Additional metadata",
    )


# ---------------------------------------------------------------------------
# Module-level store instance (lazy)
# ---------------------------------------------------------------------------

_store_instance: EmailVectorStore | None = None


def _get_store() -> EmailVectorStore:
    """Return a lazily-initialised ``EmailVectorStore`` singleton.

    Returns:
        The module-level ``EmailVectorStore`` instance.
    """
    global _store_instance  # noqa: PLW0603
    if _store_instance is None:
        _store_instance = EmailVectorStore()
    return _store_instance


def set_store(store: EmailVectorStore) -> None:
    """Override the module-level store (useful for testing / DI).

    Args:
        store: An ``EmailVectorStore`` instance to use globally.
    """
    global _store_instance  # noqa: PLW0603
    _store_instance = store


# ---------------------------------------------------------------------------
# Public search API
# ---------------------------------------------------------------------------

def retrieve_similar(
    query: str,
    top_k: int | None = None,
    category_filter: str | None = None,
) -> list[RetrievedEmail]:
    """Retrieve the most similar stored emails for a given query.

    Args:
        query: The incoming email or text to search against.
        top_k: Number of results to return.  Defaults to the value
            configured in ``settings.retrieval_top_k``.
        category_filter: If provided, restrict results to emails whose
            ``category`` metadata matches this value (case-sensitive).

    Returns:
        List of ``RetrievedEmail`` objects sorted by similarity score
        in descending order (most similar first).

    Raises:
        ValueError: If *query* is empty.
    """
    if not query or not query.strip():
        raise ValueError("Query string must be non-empty.")

    settings = get_settings()
    k = top_k if top_k is not None else settings.retrieval_top_k

    store = _get_store()

    # Build optional metadata filter
    where: dict[str, Any] | None = None
    if category_filter:
        where = {"category": {"$eq": category_filter}}

    logger.info(
        "Searching — top_k=%d  category_filter=%s  query_preview='%s'",
        k,
        category_filter,
        query[:80],
    )

    raw = store.query(
        query_texts=[query],
        n_results=k,
        where=where,
    )

    # Unpack first (and only) query's results
    ids: list[str] = raw["ids"][0] if raw["ids"] else []
    documents: list[str] = raw["documents"][0] if raw["documents"] else []
    metadatas: list[dict[str, Any]] = (
        raw["metadatas"][0] if raw["metadatas"] else []
    )
    distances: list[float] = raw["distances"][0] if raw["distances"] else []

    results: list[RetrievedEmail] = []

    for idx in range(len(ids)):
        meta = dict(metadatas[idx]) if idx < len(metadatas) else {}
        reply_text = meta.pop("reply_text", "")

        # ChromaDB with cosine space returns *distances* (lower = closer).
        # Convert to a similarity score: similarity = 1 - distance.
        distance = distances[idx] if idx < len(distances) else 0.0
        similarity = 1.0 - distance

        results.append(
            RetrievedEmail(
                email_id=ids[idx],
                email_text=documents[idx] if idx < len(documents) else "",
                reply_text=reply_text,
                similarity_score=round(similarity, 6),
                metadata=meta,
            )
        )

    # Sort descending by similarity (should already be, but ensure)
    results.sort(key=lambda r: r.similarity_score, reverse=True)

    logger.info("Retrieved %d results.", len(results))
    return results
