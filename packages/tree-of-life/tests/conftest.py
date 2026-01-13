"""Pytest fixtures for conditional forecasting tests."""

import json
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest

from tree_of_life.models import (
    Question,
    QuestionType,
    Scenario,
    GlobalScenario,
    Relationship,
    ContinuousForecast,
    CategoricalForecast,
    Signal,
)

FIXTURES_DIR = Path(__file__).parent / "fixtures"


def load_fixture(name: str) -> dict | list:
    """Load a JSON fixture file."""
    path = FIXTURES_DIR / f"{name}.json"
    with open(path) as f:
        return json.load(f)


@pytest.fixture
def questions() -> list[Question]:
    """Load test questions, transforming tree format to core Question format."""
    data = load_fixture("questions")
    questions = []
    for q in data:
        # Transform tree format to core format
        q_data = {
            "id": q["id"],
            "text": q["text"],
            "source": "tree",  # Tree questions don't come from a market source
            "question_type": QuestionType(q["type"]),  # Map 'type' string to enum
            "options": q.get("options"),
            "resolution_source": q.get("resolution_source"),
            "domain": q.get("domain"),
        }
        questions.append(Question(**q_data))
    return questions


@pytest.fixture
def raw_scenarios() -> list[Scenario]:
    """Load Phase 1 output: raw scenarios."""
    data = load_fixture("phase1_raw_scenarios")
    return [Scenario(**s) for s in data]


@pytest.fixture
def global_scenarios() -> list[GlobalScenario]:
    """Load Phase 2 output: global scenarios (without probabilities)."""
    data = load_fixture("phase2_global_scenarios")
    return [GlobalScenario(**s) for s in data]


@pytest.fixture
def relationships() -> list[Relationship]:
    """Load Phase 3 output: relationships."""
    data = load_fixture("phase3_relationships")
    return [Relationship(**r) for r in data]


@pytest.fixture
def quantified_scenarios() -> list[GlobalScenario]:
    """Load Phase 4 output: scenarios with probabilities."""
    data = load_fixture("phase4_quantified_scenarios")
    return [GlobalScenario(**s) for s in data]


@pytest.fixture
def conditionals() -> list:
    """Load Phase 5 output: conditional forecasts."""
    data = load_fixture("phase5_conditionals")
    forecasts = []
    for c in data:
        if "median" in c:
            forecasts.append(ContinuousForecast(**c))
        elif "probabilities" in c:
            forecasts.append(CategoricalForecast(**c))
    return forecasts


@pytest.fixture
def signals() -> list[Signal]:
    """Load Phase 6 output: signals."""
    data = load_fixture("phase6_signals")
    return [Signal(**s) for s in data]


@pytest.fixture
def mock_llm_responses():
    """Factory fixture to create mock LLM responses."""

    def _mock(phase: str):
        """Return mock responses for a specific phase."""
        if phase == "diverge":
            return load_fixture("phase1_raw_scenarios")
        elif phase == "converge":
            data = load_fixture("phase2_global_scenarios")
            return {"global_scenarios": data}
        elif phase == "structure":
            data = load_fixture("phase3_relationships")
            return {"relationships": data}
        elif phase == "quantify":
            data = load_fixture("phase4_quantified_scenarios")
            return {
                "probabilities": [
                    {"scenario_id": s["id"], "probability": s["probability"]}
                    for s in data
                ]
            }
        elif phase == "condition":
            return load_fixture("phase5_conditionals")
        elif phase == "signals":
            data = load_fixture("phase6_signals")
            # Group by scenario_id
            by_scenario = {}
            for s in data:
                sid = s["scenario_id"]
                if sid not in by_scenario:
                    by_scenario[sid] = []
                by_scenario[sid].append(s)
            return by_scenario

    return _mock
