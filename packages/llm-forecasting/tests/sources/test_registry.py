"""Tests for the source registry."""

import pytest

from llm_forecasting.sources import registry


class TestSourceRegistry:
    """Tests for the source registry."""

    def test_list_sources(self):
        """List all registered sources."""
        sources = registry.list()
        assert "infer" in sources
        assert "manifold" in sources
        assert "metaculus" in sources
        assert "polymarket" in sources
        assert "fred" in sources
        assert "yfinance" in sources

    def test_get_unknown_source_raises(self):
        """Getting unknown source raises KeyError."""
        with pytest.raises(KeyError):
            registry.get("nonexistent_source")

    def test_all_sources_have_required_interface(self):
        """All registered sources implement the required interface."""
        for source_name in registry.list():
            source_cls = registry.get(source_name)
            assert hasattr(source_cls, "name")
            assert hasattr(source_cls, "fetch_questions")
            assert hasattr(source_cls, "fetch_resolution")
