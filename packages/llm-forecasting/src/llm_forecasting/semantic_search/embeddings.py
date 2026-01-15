"""Embedding computation and caching using sentence-transformers."""

import json
from pathlib import Path
from typing import Optional

import numpy as np
from sentence_transformers import SentenceTransformer

# Default model: small, fast, good quality
DEFAULT_MODEL = "all-MiniLM-L6-v2"

# Lazy-loaded model singleton
_model: Optional[SentenceTransformer] = None


def get_model(model_name: str = DEFAULT_MODEL) -> SentenceTransformer:
    """Get or create the sentence transformer model (singleton)."""
    global _model
    if _model is None:
        _model = SentenceTransformer(model_name)
    return _model


def embed_texts(texts: list[str], model_name: str = DEFAULT_MODEL) -> np.ndarray:
    """
    Embed a list of texts into vectors.

    Args:
        texts: List of text strings to embed
        model_name: Name of sentence-transformer model

    Returns:
        numpy array of shape (len(texts), embedding_dim)
    """
    model = get_model(model_name)
    embeddings = model.encode(texts, convert_to_numpy=True, show_progress_bar=False)
    return embeddings


def embed_single(text: str, model_name: str = DEFAULT_MODEL) -> np.ndarray:
    """
    Embed a single text into a vector.

    Args:
        text: Text string to embed
        model_name: Name of sentence-transformer model

    Returns:
        numpy array of shape (embedding_dim,)
    """
    return embed_texts([text], model_name)[0]


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Compute cosine similarity between two vectors."""
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


class EmbeddingCache:
    """
    Cache embeddings to disk for fast retrieval.

    Stores:
    - embeddings.npy: (N, dim) float32 array
    - index.json: {"signal_key": row_index, ...}
    """

    def __init__(self, cache_dir: Path):
        self.cache_dir = Path(cache_dir)
        self.embeddings_path = self.cache_dir / "embeddings.npy"
        self.index_path = self.cache_dir / "index.json"
        self._embeddings: Optional[np.ndarray] = None
        self._index: Optional[dict[str, int]] = None

    @property
    def exists(self) -> bool:
        """Check if cache exists on disk."""
        return self.embeddings_path.exists() and self.index_path.exists()

    def load(self) -> tuple[np.ndarray, dict[str, int]]:
        """Load embeddings and index from disk."""
        if self._embeddings is None:
            self._embeddings = np.load(self.embeddings_path)
            with open(self.index_path) as f:
                self._index = json.load(f)
        return self._embeddings, self._index

    def save(self, embeddings: np.ndarray, index: dict[str, int]) -> None:
        """Save embeddings and index to disk."""
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        np.save(self.embeddings_path, embeddings.astype(np.float32))
        with open(self.index_path, "w") as f:
            json.dump(index, f)
        self._embeddings = embeddings
        self._index = index

    def get_embedding(self, key: str) -> Optional[np.ndarray]:
        """Get embedding for a specific key, or None if not found."""
        embeddings, index = self.load()
        if key not in index:
            return None
        return embeddings[index[key]]
