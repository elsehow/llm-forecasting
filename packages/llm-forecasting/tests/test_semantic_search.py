"""Tests for semantic search and deduplication."""

import pytest
import numpy as np

from llm_forecasting.semantic_search import (
    embed_texts,
    embed_single,
    SemanticSignalSearcher,
    OBSERVED_SOURCES,
)
from llm_forecasting.semantic_search.embeddings import cosine_similarity


# Test pairs with known semantic relationships
HIGH_SIMILARITY_PAIRS = [
    ("Fed raises rates", "FOMC increases federal funds target"),
    ("AI takes over jobs", "Automation causes unemployment"),
    ("GDP growth accelerates", "Economic expansion speeds up"),
    ("China invades Taiwan", "Military conflict in Taiwan strait"),
]

LOW_SIMILARITY_PAIRS = [
    ("Fed raises rates", "Olympics swimming results"),
    ("AI takes over jobs", "Best pizza in New York"),
    ("GDP growth accelerates", "Chess championship winner"),
]


class TestEmbeddings:
    """Test embedding computation."""

    def test_embed_single_returns_vector(self):
        """Single embedding should return 1D array."""
        vec = embed_single("test sentence")
        assert isinstance(vec, np.ndarray)
        assert vec.ndim == 1
        assert len(vec) == 384  # MiniLM dimension

    def test_embed_texts_returns_matrix(self):
        """Batch embedding should return 2D array."""
        texts = ["first sentence", "second sentence", "third one"]
        vecs = embed_texts(texts)
        assert isinstance(vecs, np.ndarray)
        assert vecs.shape == (3, 384)

    def test_cosine_similarity_range(self):
        """Cosine similarity should be between -1 and 1."""
        a = embed_single("hello world")
        b = embed_single("goodbye universe")
        sim = cosine_similarity(a, b)
        assert -1.0 <= sim <= 1.0

    def test_identical_texts_high_similarity(self):
        """Identical texts should have similarity ~1.0."""
        text = "The Federal Reserve announced rate changes"
        a = embed_single(text)
        b = embed_single(text)
        sim = cosine_similarity(a, b)
        assert sim > 0.99


class TestSimilarityRanking:
    """Test that semantic similarity captures meaning."""

    @pytest.mark.parametrize("text1,text2", HIGH_SIMILARITY_PAIRS)
    def test_similar_texts_high_similarity(self, text1, text2):
        """Semantically similar texts should have high similarity."""
        a = embed_single(text1)
        b = embed_single(text2)
        sim = cosine_similarity(a, b)
        # MiniLM produces ~0.4-0.5 for semantically related but differently phrased texts
        assert sim > 0.4, f"Expected high similarity for '{text1}' and '{text2}', got {sim:.3f}"

    @pytest.mark.parametrize("text1,text2", LOW_SIMILARITY_PAIRS)
    def test_unrelated_texts_low_similarity(self, text1, text2):
        """Unrelated texts should have low similarity."""
        a = embed_single(text1)
        b = embed_single(text2)
        sim = cosine_similarity(a, b)
        assert sim < 0.4, f"Expected low similarity for '{text1}' and '{text2}', got {sim:.3f}"

    def test_ranking_order(self):
        """Related signals should rank higher than unrelated."""
        query = "Federal Reserve rate hike"
        related = "FOMC raises interest rates"
        unrelated = "Chess world championship"

        query_vec = embed_single(query)
        related_sim = cosine_similarity(query_vec, embed_single(related))
        unrelated_sim = cosine_similarity(query_vec, embed_single(unrelated))

        assert related_sim > unrelated_sim


class TestDeduplication:
    """Test deduplication logic."""

    def test_deduplication_removes_duplicates(self):
        """Near-duplicates should be removed."""
        searcher = SemanticSignalSearcher()

        signals = [
            {"text": "Fed raises rates", "source": "polymarket"},
            {"text": "FOMC increases federal funds target", "source": "llm"},  # duplicate
            {"text": "Olympics swimming results", "source": "metaculus"},      # unique
        ]
        # MiniLM similarity for "Fed raises rates" vs "FOMC increases..." is ~0.47
        unique = searcher.deduplicate(signals, threshold=0.4)

        # Should keep 2: Fed signal (observed, first) + Olympics (unique)
        assert len(unique) == 2
        texts = [s["text"] for s in unique]
        assert "Fed raises rates" in texts
        assert "Olympics swimming results" in texts

    def test_deduplication_prefers_observed_over_llm(self):
        """When LLM signal duplicates observed signal, ALWAYS keep observed."""
        searcher = SemanticSignalSearcher()

        # LLM signal comes first in the list, but observed should be preferred
        signals = [
            {"text": "FOMC increases federal funds target", "source": "llm"},       # LLM first
            {"text": "Fed raises rates", "source": "polymarket"},                    # Observed second
            {"text": "Federal Reserve tightens monetary policy", "source": "llm"},  # Another LLM dup
        ]
        # MiniLM similarity for these is ~0.4-0.5
        unique = searcher.deduplicate(signals, threshold=0.4, prefer_observed=True)

        # Should keep 1: the polymarket signal (observed) wins over all LLM duplicates
        assert len(unique) == 1
        assert unique[0]["source"] == "polymarket"

    def test_deduplication_without_preference(self):
        """Without prefer_observed, first signal wins."""
        searcher = SemanticSignalSearcher()

        signals = [
            {"text": "FOMC increases federal funds target", "source": "llm"},
            {"text": "Fed raises rates", "source": "polymarket"},
        ]
        unique = searcher.deduplicate(signals, threshold=0.4, prefer_observed=False)

        # Without preference, first in list wins
        assert len(unique) == 1
        assert unique[0]["source"] == "llm"

    def test_deduplication_keeps_all_unique(self):
        """Signals below threshold should all be kept."""
        searcher = SemanticSignalSearcher()

        signals = [
            {"text": "Fed raises rates", "source": "polymarket"},
            {"text": "Olympics swimming results", "source": "metaculus"},
            {"text": "Chess championship winner", "source": "manifold"},
        ]
        unique = searcher.deduplicate(signals, threshold=0.8)

        # All unique, should keep all
        assert len(unique) == 3

    def test_deduplication_empty_list(self):
        """Empty input should return empty output."""
        searcher = SemanticSignalSearcher()
        assert searcher.deduplicate([]) == []

    def test_deduplication_single_signal(self):
        """Single signal should be returned as-is."""
        searcher = SemanticSignalSearcher()
        signals = [{"text": "Fed raises rates", "source": "llm"}]
        unique = searcher.deduplicate(signals)
        assert unique == signals


class TestObservedSources:
    """Test observed source classification."""

    def test_observed_sources_defined(self):
        """Verify expected observed sources are defined.

        Note: FRED/YahooFinance removed (2026-01-15) - they're resolution sources,
        not question sources. See Scenario Generation.md#Source Architecture
        """
        expected = {"polymarket", "metaculus", "kalshi", "manifold", "infer"}
        assert OBSERVED_SOURCES == expected

    def test_all_observed_sources_preferred(self):
        """All observed sources should be preferred over LLM."""
        searcher = SemanticSignalSearcher()

        for source in OBSERVED_SOURCES:
            signals = [
                {"text": "Some signal about economy", "source": "llm"},
                {"text": "Some similar signal about economy", "source": source},
            ]
            # These texts are very similar (~0.95+ similarity)
            unique = searcher.deduplicate(signals, threshold=0.8, prefer_observed=True)

            assert len(unique) == 1, f"Failed for source: {source}"
            assert unique[0]["source"] == source, f"Expected {source} to be preferred over llm"


class TestFindDuplicates:
    """Test duplicate finding (before removal)."""

    def test_find_duplicates_returns_pairs(self):
        """Should return pairs of duplicate indices with similarity."""
        searcher = SemanticSignalSearcher()

        signals = [
            {"text": "Fed raises rates"},
            {"text": "FOMC increases rates"},  # similar to 0
            {"text": "Olympics swimming"},      # unique
        ]
        # MiniLM similarity for Fed/FOMC pair is ~0.45
        dups = searcher.find_duplicates(signals, threshold=0.4)

        # Should find one pair: (0, 1)
        assert len(dups) >= 1
        indices = [(d[0], d[1]) for d in dups]
        assert (0, 1) in indices

    def test_find_duplicates_sorted_by_similarity(self):
        """Duplicates should be sorted by similarity descending."""
        searcher = SemanticSignalSearcher()

        signals = [
            {"text": "Fed raises rates"},
            {"text": "FOMC increases rates"},
            {"text": "Federal Reserve hikes"},
        ]
        dups = searcher.find_duplicates(signals, threshold=0.5)

        # Should be sorted by similarity
        sims = [d[2] for d in dups]
        assert sims == sorted(sims, reverse=True)
