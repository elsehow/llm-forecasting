"""Semantic search for signals using sentence-transformers."""

from .embeddings import embed_texts, embed_single, EmbeddingCache
from .searcher import SemanticSignalSearcher, SearchResult, OBSERVED_SOURCES

__all__ = [
    "embed_texts",
    "embed_single",
    "EmbeddingCache",
    "SemanticSignalSearcher",
    "SearchResult",
    "OBSERVED_SOURCES",
]
