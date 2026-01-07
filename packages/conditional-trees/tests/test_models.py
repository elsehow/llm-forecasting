"""Tests for Pydantic models."""

import pytest
from conditional_trees.models import (
    Question,
    Scenario,
    GlobalScenario,
    Relationship,
    ContinuousForecast,
    CategoricalForecast,
    BinaryForecast,
    Signal,
    ForecastTree,
)


class TestQuestion:
    def test_continuous_question(self):
        q = Question(
            id="test",
            text="US GDP in 2040",
            type="continuous",
            resolution_source="BEA",
            domain="Macro",
        )
        assert q.type == "continuous"
        assert q.options is None

    def test_categorical_question(self):
        q = Question(
            id="test",
            text="Taiwan status",
            type="categorical",
            options=["Self-governing", "PRC-administered"],
        )
        assert q.type == "categorical"
        assert len(q.options) == 2

    def test_binary_question(self):
        q = Question(id="test", text="Will X happen?", type="binary")
        assert q.type == "binary"


class TestScenario:
    def test_scenario_creation(self):
        s = Scenario(
            name="Test Scenario",
            description="A test scenario",
            key_assumptions=["assumption 1", "assumption 2"],
            source_question_id="q1",
        )
        assert s.name == "Test Scenario"
        assert len(s.key_assumptions) == 2


class TestGlobalScenario:
    def test_global_scenario_defaults(self):
        gs = GlobalScenario(
            id="test",
            name="Test Global",
            description="A global scenario",
        )
        assert gs.probability == 0.0
        assert gs.key_drivers == []
        assert gs.member_scenarios == []


class TestRelationship:
    def test_relationship_types(self):
        r = Relationship(
            scenario_a="a",
            scenario_b="b",
            type="correlated",
            strength=0.5,
        )
        assert r.type == "correlated"
        assert r.strength == 0.5

    def test_mutually_exclusive(self):
        r = Relationship(
            scenario_a="a",
            scenario_b="b",
            type="mutually_exclusive",
        )
        assert r.strength is None


class TestForecasts:
    def test_continuous_forecast(self):
        f = ContinuousForecast(
            question_id="q1",
            scenario_id="s1",
            median=100.0,
            ci_80_low=80.0,
            ci_80_high=120.0,
        )
        assert f.ci_80_low < f.median < f.ci_80_high

    def test_categorical_forecast(self):
        f = CategoricalForecast(
            question_id="q1",
            scenario_id="s1",
            probabilities={"A": 0.6, "B": 0.4},
        )
        assert sum(f.probabilities.values()) == pytest.approx(1.0)

    def test_binary_forecast(self):
        f = BinaryForecast(
            question_id="q1",
            scenario_id="s1",
            probability=0.7,
        )
        assert 0 <= f.probability <= 1


class TestSignal:
    def test_signal_creation(self):
        from datetime import date

        s = Signal(
            id="sig1",
            text="Observable event",
            resolves_by=date(2026, 6, 1),
            scenario_id="s1",
            direction="increases",
            magnitude="medium",
        )
        assert s.direction == "increases"
        assert s.magnitude == "medium"
