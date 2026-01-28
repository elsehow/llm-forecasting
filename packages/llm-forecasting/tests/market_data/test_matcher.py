"""Unit tests for market matcher."""

from datetime import datetime, timezone
from pathlib import Path

import pytest

from llm_forecasting.market_data.models import Market, MarketStatus
from llm_forecasting.market_data.storage import MarketDataStorage
from llm_forecasting.market_data.matcher import (
    extract_keywords,
    find_matching_market,
    compute_validation_status,
)


class TestExtractKeywords:
    """Tests for keyword extraction."""

    def test_extracts_years(self):
        """Extracts year numbers from questions."""
        keywords = extract_keywords("Will the 2026 election happen?")
        assert "2026" in keywords

    def test_extracts_proper_nouns(self):
        """Extracts capitalized multi-word phrases."""
        keywords = extract_keywords("Will One Battle After Another win?")
        # Should extract "One Battle After Another" or similar
        assert any("One" in kw or "Battle" in kw for kw in keywords)

    def test_extracts_quoted_phrases(self):
        """Extracts phrases in quotes."""
        keywords = extract_keywords('Will "Best Picture" be awarded?')
        assert "Best Picture" in keywords

    def test_filters_stopwords(self):
        """Filters out common stopwords."""
        keywords = extract_keywords("Will the rain fall on the ground?")
        assert "the" not in [k.lower() for k in keywords]
        assert "on" not in [k.lower() for k in keywords]

    def test_empty_question(self):
        """Returns empty list for empty question."""
        keywords = extract_keywords("")
        assert keywords == []

    def test_caps_at_10_keywords(self):
        """Returns at most 10 keywords."""
        long_q = "Will Apple Microsoft Google Amazon Meta Tesla Nvidia Intel AMD Qualcomm Oracle SAP Adobe Salesforce win?"
        keywords = extract_keywords(long_q)
        assert len(keywords) <= 10

    def test_oscars_question(self):
        """Real example: Oscar prediction question."""
        q = "Will One Battle After Another win Best Picture at the 2026 Oscars?"
        keywords = extract_keywords(q)
        # Should get key terms
        assert any("2026" in kw for kw in keywords)
        assert any("Oscar" in kw or "Picture" in kw for kw in keywords)


class TestComputeValidationStatus:
    """Tests for validation status computation."""

    def test_ok_status(self):
        """Small gaps get OK status."""
        assert compute_validation_status(0) == "OK"
        assert compute_validation_status(3) == "OK"
        assert compute_validation_status(-5) == "OK"
        assert compute_validation_status(5) == "OK"

    def test_warning_status(self):
        """Medium gaps get WARNING status."""
        status = compute_validation_status(10)
        assert "WARNING" in status
        status = compute_validation_status(-10)
        assert "WARNING" in status
        status = compute_validation_status(15)
        assert "WARNING" in status

    def test_review_status(self):
        """Large gaps get REVIEW status."""
        status = compute_validation_status(20)
        assert "REVIEW" in status
        status = compute_validation_status(-25)
        assert "REVIEW" in status


class TestSearchByTitle:
    """Tests for storage.search_by_title()."""

    @pytest.fixture
    async def storage_with_markets(self, tmp_db_path: Path):
        """Create storage with sample markets."""
        storage = MarketDataStorage(tmp_db_path)

        markets = [
            Market(
                id="poly-1",
                platform="polymarket",
                title="Will One Battle After Another win Best Picture?",
                current_probability=0.81,
                liquidity=50000,
                status=MarketStatus.OPEN,
            ),
            Market(
                id="poly-2",
                platform="polymarket",
                title="Who will win Best Picture 2026?",
                current_probability=None,
                liquidity=30000,
                status=MarketStatus.OPEN,
            ),
            Market(
                id="poly-3",
                platform="polymarket",
                title="Will there be a recession in 2026?",
                current_probability=0.25,
                liquidity=100000,
                status=MarketStatus.OPEN,
            ),
            Market(
                id="meta-1",
                platform="metaculus",
                title="2026 Academy Awards Best Picture Winner",
                current_probability=0.65,
                status=MarketStatus.OPEN,
            ),
        ]

        await storage.save_markets(markets)
        yield storage
        await storage.close()

    async def test_search_single_keyword(self, storage_with_markets: MarketDataStorage):
        """Search with single keyword."""
        results = await storage_with_markets.search_by_title(
            keywords=["Picture"],
        )
        assert len(results) >= 2  # Should find multiple Picture markets

    async def test_search_multiple_keywords(self, storage_with_markets: MarketDataStorage):
        """Search with multiple keywords (OR logic)."""
        results = await storage_with_markets.search_by_title(
            keywords=["recession", "Picture"],
        )
        assert len(results) >= 3  # Should find recession + picture markets

    async def test_search_filter_platform(self, storage_with_markets: MarketDataStorage):
        """Filter by platform."""
        results = await storage_with_markets.search_by_title(
            keywords=["Picture"],
            platform="polymarket",
        )
        assert all(m.platform == "polymarket" for m in results)

    async def test_search_filter_liquidity(self, storage_with_markets: MarketDataStorage):
        """Filter by minimum liquidity."""
        results = await storage_with_markets.search_by_title(
            keywords=["recession", "Picture"],
            min_liquidity=40000,
        )
        assert all(m.liquidity is None or m.liquidity >= 40000 for m in results)

    async def test_search_no_results(self, storage_with_markets: MarketDataStorage):
        """Search with no matches returns empty."""
        results = await storage_with_markets.search_by_title(
            keywords=["xyznonexistent"],
        )
        assert results == []

    async def test_search_empty_keywords(self, storage_with_markets: MarketDataStorage):
        """Search with empty keywords returns empty."""
        results = await storage_with_markets.search_by_title(keywords=[])
        assert results == []

    async def test_search_respects_limit(self, storage_with_markets: MarketDataStorage):
        """Search respects limit parameter."""
        results = await storage_with_markets.search_by_title(
            keywords=["Picture", "2026"],
            limit=2,
        )
        assert len(results) <= 2


class TestFindMatchingMarket:
    """Tests for find_matching_market()."""

    @pytest.fixture
    async def storage_with_markets(self, tmp_db_path: Path):
        """Create storage with sample markets."""
        storage = MarketDataStorage(tmp_db_path)

        markets = [
            Market(
                id="poly-oscars",
                platform="polymarket",
                title="Will One Battle After Another win Best Picture?",
                url="https://polymarket.com/event/oscars-2026",
                current_probability=0.81,
                liquidity=50000,
                status=MarketStatus.OPEN,
            ),
            Market(
                id="poly-recession",
                platform="polymarket",
                title="Will there be a recession in 2026?",
                url="https://polymarket.com/event/recession-2026",
                current_probability=0.25,
                liquidity=100000,
                status=MarketStatus.OPEN,
            ),
        ]

        await storage.save_markets(markets)
        yield storage
        await storage.close()

    async def test_find_match_no_llm(self, storage_with_markets: MarketDataStorage):
        """Find match without LLM reranking."""
        match = await find_matching_market(
            question="Will One Battle win Best Picture?",
            storage=storage_with_markets,
            platforms=["polymarket"],
            use_llm_rerank=False,
        )

        assert match is not None
        assert match.market.id == "poly-oscars"
        assert match.match_method == "keyword"

    async def test_no_match_found(self, storage_with_markets: MarketDataStorage):
        """Returns None when no match found."""
        match = await find_matching_market(
            question="Will aliens land on Earth?",
            storage=storage_with_markets,
            platforms=["polymarket"],
            use_llm_rerank=False,
        )

        assert match is None

    async def test_match_includes_probability(self, storage_with_markets: MarketDataStorage):
        """Match includes market probability."""
        match = await find_matching_market(
            question="Will there be a recession?",
            storage=storage_with_markets,
            platforms=["polymarket"],
            use_llm_rerank=False,
        )

        assert match is not None
        assert match.market.current_probability == 0.25


@pytest.mark.integration
class TestFindMatchingMarketWithLLM:
    """Integration tests that require LLM API access."""

    @pytest.fixture
    async def storage_with_markets(self, tmp_db_path: Path):
        """Create storage with sample markets."""
        storage = MarketDataStorage(tmp_db_path)

        markets = [
            Market(
                id="poly-oscars",
                platform="polymarket",
                title="Will One Battle After Another win Best Picture at 2026 Oscars?",
                url="https://polymarket.com/event/oscars-2026",
                current_probability=0.81,
                liquidity=50000,
                status=MarketStatus.OPEN,
            ),
            Market(
                id="poly-sinners",
                platform="polymarket",
                title="Will Sinners win Best Picture at 2026 Oscars?",
                url="https://polymarket.com/event/sinners-2026",
                current_probability=0.12,
                liquidity=30000,
                status=MarketStatus.OPEN,
            ),
        ]

        await storage.save_markets(markets)
        yield storage
        await storage.close()

    async def test_llm_rerank_selects_correct_match(
        self, storage_with_markets: MarketDataStorage
    ):
        """LLM correctly selects best match from candidates."""
        match = await find_matching_market(
            question="What are the odds One Battle After Another wins Best Picture?",
            storage=storage_with_markets,
            platforms=["polymarket"],
            use_llm_rerank=True,
        )

        assert match is not None
        assert match.match_method == "llm_rerank"
        assert "One Battle" in match.market.title
        assert match.match_confidence > 0.5
