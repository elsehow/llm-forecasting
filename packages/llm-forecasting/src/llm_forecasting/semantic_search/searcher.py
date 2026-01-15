"""Semantic signal searcher with deduplication."""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

from .embeddings import embed_texts, embed_single, EmbeddingCache, cosine_similarity


# Sources considered "observed" (from prediction markets)
# vs "synthesized" (LLM-generated)
# Note: FRED/YahooFinance removed - they're resolution sources, not question sources
# See: Obsidian/projects/Scenario Generation.md#Source Architecture
OBSERVED_SOURCES = {"polymarket", "metaculus", "kalshi", "manifold", "infer"}


@dataclass
class SearchResult:
    """A single search result."""
    text: str
    source: str
    similarity: float
    id: Optional[str] = None


class SemanticSignalSearcher:
    """
    Semantic search over signals using sentence-transformer embeddings.

    Features:
    - Build embedding cache from database
    - Search by query similarity
    - Deduplicate signals with preference for observed sources
    """

    def __init__(self, db_path: Optional[str] = None, cache_dir: Optional[Path] = None):
        """
        Initialize searcher.

        Args:
            db_path: Path to forecastbench.db (optional, for building cache)
            cache_dir: Directory for embedding cache (default: data/embeddings/)
        """
        self.db_path = db_path
        if cache_dir is None:
            # Default to data/embeddings relative to db
            if db_path:
                cache_dir = Path(db_path).parent / "embeddings"
            else:
                cache_dir = Path("data/embeddings")
        self.cache = EmbeddingCache(cache_dir)
        self._signals: Optional[list[dict]] = None

    def build_cache(self, batch_size: int = 100) -> None:
        """
        Build embedding cache from database.

        Embeds all texts and saves to disk.
        """
        if not self.db_path:
            raise ValueError("db_path required to build cache")

        import sqlite3

        conn = sqlite3.connect(self.db_path)
        cursor = conn.execute("""
            SELECT id, source, text
            FROM questions
            WHERE text IS NOT NULL AND text != ''
        """)

        signals = []
        texts = []
        index = {}

        for row in cursor:
            qid, source, text = row
            key = f"{qid}|{source}"
            index[key] = len(signals)
            signals.append({"id": qid, "source": source, "text": text})
            texts.append(text)

        conn.close()

        print(f"Embedding {len(texts)} signals...")
        embeddings = embed_texts(texts)

        self.cache.save(embeddings, index)
        self._signals = signals
        print(f"Cache saved to {self.cache.cache_dir}")

    def _load_signals_from_db(self) -> list[dict]:
        """Load signal metadata from database."""
        if self._signals is not None:
            return self._signals

        if not self.db_path:
            raise ValueError("db_path required to load signals")

        import sqlite3

        conn = sqlite3.connect(self.db_path)
        cursor = conn.execute("""
            SELECT id, source, text
            FROM questions
            WHERE text IS NOT NULL AND text != ''
        """)

        self._signals = [
            {"id": row[0], "source": row[1], "text": row[2]}
            for row in cursor
        ]
        conn.close()
        return self._signals

    def search(self, query: str, top_k: int = 20) -> list[SearchResult]:
        """
        Find signals most similar to query.

        Args:
            query: Search query text
            top_k: Number of results to return

        Returns:
            List of SearchResult sorted by similarity (descending)
        """
        if not self.cache.exists:
            raise ValueError("Cache not built. Call build_cache() first.")

        embeddings, index = self.cache.load()
        signals = self._load_signals_from_db()

        query_emb = embed_single(query)

        # Compute similarities
        similarities = np.dot(embeddings, query_emb)

        # Get top-k indices
        top_indices = np.argsort(similarities)[::-1][:top_k]

        results = []
        for idx in top_indices:
            sig = signals[idx]
            results.append(SearchResult(
                text=sig["text"],
                source=sig["source"],
                similarity=float(similarities[idx]),
                id=sig.get("id"),
            ))

        return results

    def find_duplicates(
        self,
        signals: list[dict],
        threshold: float = 0.85
    ) -> list[tuple[int, int, float]]:
        """
        Find near-duplicate pairs in a list of signals.

        Args:
            signals: List of dicts with "text" key
            threshold: Similarity threshold for duplicates

        Returns:
            List of (idx1, idx2, similarity) tuples for pairs above threshold
        """
        texts = [s["text"] for s in signals]
        embeddings = embed_texts(texts)

        duplicates = []
        n = len(signals)

        for i in range(n):
            for j in range(i + 1, n):
                sim = cosine_similarity(embeddings[i], embeddings[j])
                if sim >= threshold:
                    duplicates.append((i, j, sim))

        return sorted(duplicates, key=lambda x: -x[2])

    def deduplicate(
        self,
        signals: list[dict],
        threshold: float = 0.85,
        prefer_observed: bool = True,
    ) -> list[dict]:
        """
        Remove near-duplicates, preferring observed signals over LLM-generated.

        Algorithm:
        1. Embed all signal texts
        2. Sort signals: observed sources first, then LLM
        3. Greedy clustering: for each signal, if similar to existing → skip
        4. Result: observed signals always kept over LLM duplicates

        Args:
            signals: List of dicts with "text" and "source" keys
            threshold: Similarity threshold (0-1). Higher = stricter matching.
            prefer_observed: If True, always keep observed over LLM duplicates.

        Returns:
            Deduplicated list of signals
        """
        if not signals:
            return []

        texts = [s["text"] for s in signals]
        embeddings = embed_texts(texts)

        # Create list of (signal, embedding) pairs
        pairs = list(zip(signals, embeddings))

        # Sort to process observed signals first if preferred
        # Within each category, sort by VOI descending so we keep highest-VOI variant
        if prefer_observed:
            pairs = sorted(
                pairs,
                key=lambda x: (
                    0 if x[0].get("source", "").lower() in OBSERVED_SOURCES else 1,
                    -x[0].get("voi", 0),  # Higher VOI first within category
                )
            )

        kept = []
        kept_embeddings = []

        for signal, emb in pairs:
            # Check if too similar to any kept signal
            if kept_embeddings:
                # Compute similarities to all kept embeddings
                kept_matrix = np.array(kept_embeddings)
                sims = np.dot(kept_matrix, emb)

                if np.max(sims) >= threshold:
                    # Too similar to an existing signal - skip
                    continue

            kept.append(signal)
            kept_embeddings.append(emb)

        return kept
