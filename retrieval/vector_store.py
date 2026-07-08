"""
ChromaDB-backed vector store for the AI Email Response System.

Stores email embeddings and metadata for fast similarity retrieval.
Uses the ``EmbeddingModel`` from ``retrieval.embedding`` to generate
embeddings and a custom ChromaDB ``EmbeddingFunction`` wrapper so that
the same model is used for both indexing and querying.

Usage::

    from retrieval.vector_store import EmailVectorStore
    import pandas as pd

    store = EmailVectorStore()
    df = pd.read_csv("dataset/emails.csv")
    store.build_index(df)
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Sequence

import chromadb
import numpy as np
import pandas as pd
from chromadb.api.types import (
    Documents,
    EmbeddingFunction,
    Embeddings,
)
from tqdm import tqdm

from config import get_settings
from retrieval.embedding import EmbeddingModel, get_embedding_model

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Custom ChromaDB embedding function backed by our EmbeddingModel
# ---------------------------------------------------------------------------

class _SentenceTransformerEmbeddingFunction(EmbeddingFunction[Documents]):
    """ChromaDB-compatible wrapper around ``EmbeddingModel``.

    ChromaDB requires an object that implements ``__call__`` with the
    ``EmbeddingFunction`` protocol.  This adapter delegates to our
    singleton ``EmbeddingModel``.
    """

    def __init__(self, embedding_model: EmbeddingModel) -> None:
        self._model = embedding_model

    def __call__(self, input: Documents) -> Embeddings:  # noqa: A002
        """Embed a list of documents.

        Args:
            input: List of text strings.

        Returns:
            List of embedding vectors (each as a list of floats).
        """
        vectors = self._model.encode(list(input), show_progress=False)
        return vectors.tolist()


# ---------------------------------------------------------------------------
# Vector store
# ---------------------------------------------------------------------------

class EmailVectorStore:
    """ChromaDB-backed vector store for email-response pairs.

    Manages a single ChromaDB collection that stores email text as
    documents, reply text + metadata as metadata fields, and
    sentence-transformer embeddings for similarity search.

    Args:
        persist_dir: Directory for ChromaDB persistence.  Defaults to
            the value from application settings.
        collection_name: Name of the ChromaDB collection.
        embedding_model: Optional ``EmbeddingModel`` instance.  If not
            provided, the module-level singleton is used.
    """

    def __init__(
        self,
        persist_dir: str | Path | None = None,
        collection_name: str | None = None,
        embedding_model: EmbeddingModel | None = None,
    ) -> None:
        settings = get_settings()

        self._persist_dir = Path(
            persist_dir or settings.chroma_abs_path,
        )
        self._persist_dir.mkdir(parents=True, exist_ok=True)

        self._collection_name: str = (
            collection_name or settings.chroma_collection_name
        )

        self._embedding_model: EmbeddingModel = (
            embedding_model or get_embedding_model()
        )

        self._chroma_ef = _SentenceTransformerEmbeddingFunction(
            self._embedding_model,
        )

        logger.info(
            "Initialising ChromaDB — persist_dir=%s  collection=%s",
            self._persist_dir,
            self._collection_name,
        )

        self._client = chromadb.PersistentClient(
            path=str(self._persist_dir),
        )
        self._collection = self._client.get_or_create_collection(
            name=self._collection_name,
            embedding_function=self._chroma_ef,
            metadata={"hnsw:space": "cosine"},
        )

        logger.info(
            "Collection '%s' ready — %d documents.",
            self._collection_name,
            self._collection.count(),
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def collection(self) -> chromadb.Collection:
        """Return the underlying ChromaDB collection."""
        return self._collection

    def get_collection_count(self) -> int:
        """Return the number of documents in the collection.

        Returns:
            Document count.
        """
        return self._collection.count()

    def add_email(
        self,
        email_id: str,
        email_text: str,
        reply_text: str,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Add a single email-response pair to the store.

        If an entry with the same *email_id* already exists it will be
        upserted (updated in place).

        Args:
            email_id: Unique identifier.
            email_text: The incoming email text (used as the document).
            reply_text: The ideal reply (stored in metadata).
            metadata: Additional metadata (category, urgency, …).
        """
        meta: dict[str, Any] = dict(metadata) if metadata else {}
        meta["reply_text"] = reply_text

        self._collection.upsert(
            ids=[email_id],
            documents=[email_text],
            metadatas=[meta],
        )
        logger.debug("Upserted email %s", email_id)

    def build_index(
        self,
        emails_df: pd.DataFrame,
        batch_size: int = 128,
    ) -> None:
        """Embed and store all rows from *emails_df*.

        Expected DataFrame columns: ``id``, ``email``, ``reply``, and
        optionally ``category``, ``sender_type``, ``urgency``,
        ``sentiment``, ``style``.

        Args:
            emails_df: DataFrame of email-response pairs.
            batch_size: Number of rows to upsert per ChromaDB call.

        Raises:
            ValueError: If required columns are missing.
        """
        required_cols = {"id", "email", "reply"}
        missing = required_cols - set(emails_df.columns)
        if missing:
            raise ValueError(f"DataFrame is missing required columns: {missing}")

        metadata_cols = [
            "category", "sender_type", "urgency", "sentiment", "style",
        ]

        total = len(emails_df)
        logger.info("Building index — %d emails, batch_size=%d", total, batch_size)

        for start in tqdm(
            range(0, total, batch_size),
            desc="Indexing emails",
            unit="batch",
        ):
            chunk = emails_df.iloc[start : start + batch_size]

            ids: list[str] = chunk["id"].astype(str).tolist()
            documents: list[str] = chunk["email"].astype(str).tolist()

            metadatas: list[dict[str, Any]] = []
            for _, row in chunk.iterrows():
                meta: dict[str, Any] = {
                    "reply_text": str(row["reply"]),
                }
                for col in metadata_cols:
                    if col in row.index and pd.notna(row[col]):
                        meta[col] = str(row[col])
                metadatas.append(meta)

            self._collection.upsert(
                ids=ids,
                documents=documents,
                metadatas=metadatas,
            )

        logger.info(
            "Index build complete — collection now has %d documents.",
            self._collection.count(),
        )

    def query(
        self,
        query_texts: list[str],
        n_results: int = 5,
        where: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Raw query against the ChromaDB collection.

        This is a thin wrapper around ``collection.query`` that adds
        logging and error handling.

        Args:
            query_texts: List of query strings.
            n_results: Number of results per query.
            where: Optional ChromaDB metadata filter.

        Returns:
            Raw ChromaDB query result dict with keys
            ``ids``, ``documents``, ``metadatas``, ``distances``.
        """
        kwargs: dict[str, Any] = {
            "query_texts": query_texts,
            "n_results": n_results,
            "include": ["documents", "metadatas", "distances"],
        }
        if where:
            kwargs["where"] = where

        logger.debug(
            "Querying collection — n_results=%d  where=%s",
            n_results,
            where,
        )

        results = self._collection.query(**kwargs)
        return results  # type: ignore[return-value]

    def delete_collection(self) -> None:
        """Delete the entire collection (irreversible).

        Useful for testing or forced re-indexing.
        """
        logger.warning("Deleting collection '%s'", self._collection_name)
        self._client.delete_collection(self._collection_name)
        # Re-create so the store remains usable
        self._collection = self._client.get_or_create_collection(
            name=self._collection_name,
            embedding_function=self._chroma_ef,
            metadata={"hnsw:space": "cosine"},
        )
