"""Pytest configuration and fixtures."""

import asyncio
from pathlib import Path

import pytest
from dotenv import load_dotenv

# Load .env file so API keys are available to tests
# Use override=True to replace empty values that may be set in shell
load_dotenv(override=True)


def pytest_addoption(parser):
    """Add custom command line options."""
    parser.addoption(
        "--integration",
        action="store_true",
        default=False,
        help="Run integration tests (requires network access)",
    )


def pytest_configure(config):
    """Register custom markers."""
    config.addinivalue_line(
        "markers", "integration: mark test as integration test (requires --integration to run)"
    )


def pytest_collection_modifyitems(config, items):
    """Skip integration tests unless --integration flag is passed."""
    if config.getoption("--integration"):
        # --integration given: run all tests
        return

    skip_integration = pytest.mark.skip(reason="use --integration to run")
    for item in items:
        if "integration" in item.keywords:
            item.add_marker(skip_integration)


@pytest.fixture(scope="session")
def event_loop():
    """Create an event loop for the test session."""
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
def tmp_db_path(tmp_path: Path) -> Path:
    """Provide a temporary database path."""
    return tmp_path / "test.db"
