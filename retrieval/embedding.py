"""
Embedding model wrapper for the AI Email Response System.

Provides a singleton ``EmbeddingModel`` backed by *sentence-transformers*
that encodes text into dense vectors for similarity search.

Usage::

    from retrieval.embedding import get_embedding_model

    model = get_embedding_model()
    vectors = model.encode(["Hello world", "Goodbye world"])
"""

from __future__ import annotations

import logging
from functools import lru_cache
from typing import Sequence

import numpy as np
from numpy.typing import NDArray
from sentence_transformers import SentenceTransformer

from config import get_settings

logger = logging.getLogger(__name__)


class EmbeddingModel:
    """Thin wrapper around a ``SentenceTransformer`` model.

    Provides convenience methods for encoding text, computing similarity,
    and batch encoding with a progress bar.

    Args:
        model_name: HuggingFace model identifier. Defaults to the value
            configured in :pydata:`config.Settings.embedding_model`.
    """

    def __init__(self, model_name: str | None = None) -> None:
        settings = get_settings()
        self._model_name: str = model_name or settings.embedding_model

        logger.info("Loading embedding model: %s", self._model_name)
        self._model = SentenceTransformer(self._model_name)
        self._dimension: int = self._model.get_sentence_embedding_dimension()  # type: ignore[assignment]
        logger.info(
            "Embedding model loaded — dimension=%d", self._dimension,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def model_name(self) -> str:
        """Return the model identifier."""
        return self._model_name

    @property
    def dimension(self) -> int:
        """Return the embedding dimensionality."""
        return self._dimension

    def encode(
        self,
        texts: list[str] | Sequence[str],
        batch_size: int = 64,
        show_progress: bool | None = None,
    ) -> NDArray[np.float32]:
        """Encode a list of texts into dense vectors.

        For large inputs (> 256 texts) a tqdm progress bar is shown
        automatically unless *show_progress* is explicitly set.

        Args:
            texts: Texts to encode.
            batch_size: Internal encoding batch size.
            show_progress: Whether to show a progress bar.  ``None`` means
                auto-detect based on input size.

        Returns:
            2-D ``np.ndarray`` of shape ``(len(texts), dimension)``.
        """
        if show_progress is None:
            show_progress = len(texts) > 256

        logger.debug("Encoding %d texts (batch_size=%d)", len(texts), batch_size)
        embeddings: NDArray[np.float32] = self._model.encode(
            list(texts),
            batch_size=batch_size,
            show_progress_bar=show_progress,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        return embeddings

    def encode_single(self, text: str) -> NDArray[np.float32]:
        """Encode a single text string.

        Args:
            text: Text to encode.

        Returns:
            1-D ``np.ndarray`` of shape ``(dimension,)``.
        """
        return self.encode([text], show_progress=False)[0]

    def similarity(
        self,
        emb1: NDArray[np.float32],
        emb2: NDArray[np.float32],
    ) -> float:
        """Compute cosine similarity between two embedding vectors.

        Both vectors are assumed to be L2-normalised (which
        ``SentenceTransformer.encode`` does when ``normalize_embeddings=True``),
        so the cosine similarity simplifies to a dot product.

        Args:
            emb1: First embedding vector.
            emb2: Second embedding vector.

        Returns:
            Cosine similarity in the range ``[-1, 1]``.
        """
        # Flatten in case caller passes 2-D single-row arrays
        v1 = emb1.flatten()
        v2 = emb2.flatten()

        dot = float(np.dot(v1, v2))
        # Clamp to valid range to avoid floating-point artefacts
        return max(-1.0, min(1.0, dot))


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------

@lru_cache(maxsize=1)
def get_embedding_model(model_name: str | None = None) -> EmbeddingModel:
    """Return a cached singleton ``EmbeddingModel``.

    The first call loads the model; subsequent calls return the same
    instance.

    Args:
        model_name: Optional override for the model identifier.

    Returns:
        Singleton ``EmbeddingModel`` instance.
    """
    return EmbeddingModel(model_name=model_name)
