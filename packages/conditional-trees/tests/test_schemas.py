"""Tests to validate fixtures against Pydantic response schemas.

These tests ensure that:
1. Fixture data matches the expected LLM response format
2. The Pydantic schemas correctly validate the fixture data
3. Schema changes don't silently break fixtures
"""

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from conditional_trees.schemas import (
    BaseRateItem,
    BaseRatesResponse,
    ConditionBatchBinaryResponse,
    ConditionBatchCategoricalResponse,
    ConditionBatchContinuousResponse,
    ConvergeResponse,
    DivergeResponse,
    GlobalScenarioItem,
    ProbabilityItem,
    QuantifyResponse,
    RelationshipItem,
    ScenarioItem,
    SignalItem,
    SignalsResponse,
    StructureResponse,
    make_strict_schema,
)

FIXTURES_DIR = Path(__file__).parent / "fixtures"


def load_fixture(name: str) -> list | dict:
    """Load a JSON fixture file."""
    path = FIXTURES_DIR / f"{name}.json"
    with open(path) as f:
        return json.load(f)


class TestMakeStrictSchema:
    """Tests for the make_strict_schema helper."""

    def test_adds_additional_properties_false(self):
        """Verify additionalProperties: false is added to all objects."""
        schema = make_strict_schema(DivergeResponse)

        # Check root object
        assert schema.get("additionalProperties") is False

        # Check nested objects in $defs
        if "$defs" in schema:
            for name, defn in schema["$defs"].items():
                if defn.get("type") == "object":
                    assert defn.get("additionalProperties") is False, (
                        f"Missing additionalProperties: false in {name}"
                    )

    def test_all_response_schemas_are_strict(self):
        """Verify all response schemas get additionalProperties: false."""
        response_models = [
            BaseRatesResponse,
            DivergeResponse,
            ConvergeResponse,
            StructureResponse,
            QuantifyResponse,
            ConditionBatchContinuousResponse,
            ConditionBatchCategoricalResponse,
            ConditionBatchBinaryResponse,
            SignalsResponse,
        ]

        for model in response_models:
            schema = make_strict_schema(model)
            assert schema.get("additionalProperties") is False, (
                f"{model.__name__} missing additionalProperties: false"
            )


class TestPhase1DivergeSchema:
    """Tests for Phase 1 (Diverge) schema validation."""

    def test_scenario_item_validates(self):
        """Individual scenario items validate correctly."""
        data = load_fixture("phase1_raw_scenarios")
        for item in data[:5]:  # Test first 5
            # ScenarioItem expects name, description, key_assumptions
            scenario = ScenarioItem(
                name=item["name"],
                description=item["description"],
                key_assumptions=item["key_assumptions"],
            )
            assert scenario.name == item["name"]

    def test_diverge_response_validates(self):
        """DivergeResponse schema validates wrapped fixture data."""
        data = load_fixture("phase1_raw_scenarios")

        # Wrap in LLM response format
        response_data = {
            "scenarios": [
                {
                    "name": s["name"],
                    "description": s["description"],
                    "key_assumptions": s["key_assumptions"],
                }
                for s in data[:5]
            ]
        }

        response = DivergeResponse(**response_data)
        assert len(response.scenarios) == 5

    def test_diverge_response_rejects_invalid(self):
        """DivergeResponse rejects invalid data."""
        with pytest.raises(ValidationError):
            DivergeResponse(scenarios=[{"invalid": "data"}])


class TestPhase2ConvergeSchema:
    """Tests for Phase 2 (Converge) schema validation."""

    def test_global_scenario_item_validates(self):
        """Individual global scenario items validate correctly."""
        data = load_fixture("phase2_global_scenarios")
        for item in data[:5]:
            scenario = GlobalScenarioItem(
                id=item["id"],
                name=item["name"],
                description=item["description"],
                key_drivers=item["key_drivers"],
                member_scenarios=item["member_scenarios"],
            )
            assert scenario.id == item["id"]

    def test_converge_response_validates(self):
        """ConvergeResponse schema validates wrapped fixture data."""
        data = load_fixture("phase2_global_scenarios")

        response_data = {"global_scenarios": data}
        response = ConvergeResponse(**response_data)
        assert len(response.global_scenarios) == len(data)

    def test_converge_response_rejects_missing_id(self):
        """ConvergeResponse rejects scenarios without id."""
        with pytest.raises(ValidationError):
            ConvergeResponse(
                global_scenarios=[
                    {
                        "name": "Test",
                        "description": "Test desc",
                        "key_drivers": [],
                        "member_scenarios": [],
                        # Missing "id"
                    }
                ]
            )


class TestPhase3StructureSchema:
    """Tests for Phase 3 (Structure) schema validation."""

    def test_relationship_item_validates(self):
        """Individual relationship items validate correctly."""
        data = load_fixture("phase3_relationships")
        for item in data[:5]:
            rel = RelationshipItem(
                scenario_a=item["scenario_a"],
                scenario_b=item["scenario_b"],
                type=item["type"],
                strength=item.get("strength"),
                notes=item.get("notes"),
            )
            assert rel.type in ["orthogonal", "correlated", "hierarchical", "mutually_exclusive"]

    def test_structure_response_validates(self):
        """StructureResponse schema validates wrapped fixture data."""
        data = load_fixture("phase3_relationships")

        response_data = {"relationships": data}
        response = StructureResponse(**response_data)
        assert len(response.relationships) == len(data)

    def test_structure_response_rejects_invalid_type(self):
        """StructureResponse rejects invalid relationship types."""
        with pytest.raises(ValidationError):
            StructureResponse(
                relationships=[
                    {
                        "scenario_a": "a",
                        "scenario_b": "b",
                        "type": "invalid_type",  # Not a valid type
                    }
                ]
            )


class TestPhase4QuantifySchema:
    """Tests for Phase 4 (Quantify) schema validation."""

    def test_probability_item_validates(self):
        """Individual probability items validate correctly."""
        data = load_fixture("phase4_quantified_scenarios")
        for item in data:
            prob = ProbabilityItem(
                scenario_id=item["id"],
                probability=item["probability"],
                reasoning="Test reasoning",
            )
            assert 0 <= prob.probability <= 1

    def test_quantify_response_validates(self):
        """QuantifyResponse schema validates fixture-derived data."""
        data = load_fixture("phase4_quantified_scenarios")

        # Convert to LLM response format
        response_data = {
            "probabilities": [
                {
                    "scenario_id": s["id"],
                    "probability": s["probability"],
                    "reasoning": "Test reasoning",
                }
                for s in data
            ]
        }

        response = QuantifyResponse(**response_data)
        assert len(response.probabilities) == len(data)

    def test_probability_bounds(self):
        """Probabilities must be between 0 and 1."""
        with pytest.raises(ValidationError):
            ProbabilityItem(scenario_id="test", probability=1.5, reasoning="Test")

        with pytest.raises(ValidationError):
            ProbabilityItem(scenario_id="test", probability=-0.1, reasoning="Test")


class TestPhase5ConditionSchema:
    """Tests for Phase 5 (Condition) schema validation."""

    def test_continuous_forecasts_validate(self):
        """Continuous forecasts validate correctly."""
        data = load_fixture("phase5_conditionals")
        continuous = [c for c in data if "median" in c]

        # Build batched response format (grouped by question)
        by_question: dict[str, dict] = {}
        directions_by_question: dict[str, dict] = {}
        for c in continuous[:10]:
            qid = c["question_id"]
            if qid not in by_question:
                by_question[qid] = {}
                directions_by_question[qid] = {}
            by_question[qid][c["scenario_id"]] = {
                "median": c["median"],
                "ci_80_low": c["ci_80_low"],
                "ci_80_high": c["ci_80_high"],
                "reasoning": c.get("reasoning", "Test"),
            }
            # Add direction (use "neutral" as default for test data)
            directions_by_question[qid][c["scenario_id"]] = "neutral"

        # Validate each question's response
        for qid, forecasts in by_question.items():
            response = ConditionBatchContinuousResponse(
                directions=directions_by_question[qid],
                forecasts=forecasts
            )
            assert len(response.forecasts) > 0
            assert len(response.directions) > 0

    def test_categorical_forecasts_validate(self):
        """Categorical forecasts validate correctly."""
        data = load_fixture("phase5_conditionals")
        categorical = [c for c in data if "probabilities" in c and "median" not in c]

        if categorical:
            # Build batched response
            by_question: dict[str, dict] = {}
            directions_by_question: dict[str, dict] = {}
            for c in categorical[:10]:
                qid = c["question_id"]
                if qid not in by_question:
                    by_question[qid] = {}
                    directions_by_question[qid] = {}
                by_question[qid][c["scenario_id"]] = {
                    "probabilities": c["probabilities"],
                    "reasoning": c.get("reasoning", "Test"),
                }
                # Add direction (use "neutral" as default for test data)
                directions_by_question[qid][c["scenario_id"]] = "neutral"

            for qid, forecasts in by_question.items():
                response = ConditionBatchCategoricalResponse(
                    directions=directions_by_question[qid],
                    forecasts=forecasts
                )
                assert len(response.forecasts) > 0
                assert len(response.directions) > 0

    def test_binary_forecasts_validate(self):
        """Binary forecasts validate correctly."""
        data = load_fixture("phase5_conditionals")
        binary = [c for c in data if "probability" in c and "median" not in c]

        if binary:
            by_question: dict[str, dict] = {}
            directions_by_question: dict[str, dict] = {}
            for c in binary[:10]:
                qid = c["question_id"]
                if qid not in by_question:
                    by_question[qid] = {}
                    directions_by_question[qid] = {}
                by_question[qid][c["scenario_id"]] = {
                    "probability": c["probability"],
                    "reasoning": c.get("reasoning", "Test"),
                }
                # Add direction (use "neutral" as default for test data)
                directions_by_question[qid][c["scenario_id"]] = "neutral"

            for qid, forecasts in by_question.items():
                response = ConditionBatchBinaryResponse(
                    directions=directions_by_question[qid],
                    forecasts=forecasts
                )
                assert len(response.forecasts) > 0
                assert len(response.directions) > 0


class TestPhase6SignalsSchema:
    """Tests for Phase 6 (Signals) schema validation."""

    def test_signal_item_validates(self):
        """Individual signal items validate correctly."""
        data = load_fixture("phase6_signals")
        for item in data[:5]:
            signal = SignalItem(
                text=item["text"],
                resolves_by=item["resolves_by"],
                direction=item["direction"],
                magnitude=item["magnitude"],
                current_probability=item.get("current_probability"),
                update_cadence=item.get("update_cadence", "event"),
                causal_priority=item.get("causal_priority", 50),
            )
            assert signal.direction in ["increases", "decreases"]
            assert signal.magnitude in ["small", "medium", "large"]

    def test_signals_response_validates(self):
        """SignalsResponse schema validates wrapped fixture data."""
        data = load_fixture("phase6_signals")

        # Group by scenario and take first 5 signals
        first_scenario = data[0]["scenario_id"]
        scenario_signals = [s for s in data if s["scenario_id"] == first_scenario][:5]

        response_data = {
            "signals": [
                {
                    "text": s["text"],
                    "resolves_by": s["resolves_by"],
                    "direction": s["direction"],
                    "magnitude": s["magnitude"],
                    "current_probability": s.get("current_probability"),
                    "update_cadence": s.get("update_cadence", "event"),
                    "causal_priority": s.get("causal_priority", 50),
                }
                for s in scenario_signals
            ]
        }

        response = SignalsResponse(**response_data)
        assert len(response.signals) == len(scenario_signals)

    def test_signal_direction_enum(self):
        """Signal direction must be increases or decreases."""
        with pytest.raises(ValidationError):
            SignalItem(
                text="Test",
                resolves_by="2026-01-01",
                direction="invalid",  # Not valid
                magnitude="medium",
            )

    def test_signal_magnitude_enum(self):
        """Signal magnitude must be small, medium, or large."""
        with pytest.raises(ValidationError):
            SignalItem(
                text="Test",
                resolves_by="2026-01-01",
                direction="increases",
                magnitude="huge",  # Not valid
            )


class TestBaseRatesSchema:
    """Tests for Phase 0 (Base Rates) schema validation."""

    def test_base_rate_item_validates(self):
        """BaseRateItem validates correctly."""
        item = BaseRateItem(
            name="us_gdp",
            description="US Real GDP",
            value=29.0,
            unit="trillion_usd",
            as_of="2025-09-30",
            source="BEA",
        )
        assert item.value == 29.0

    def test_base_rates_response_validates(self):
        """BaseRatesResponse validates correctly."""
        response = BaseRatesResponse(
            base_rates=[
                BaseRateItem(
                    name="us_gdp",
                    description="US Real GDP",
                    value=29.0,
                    unit="trillion_usd",
                    as_of="2025-09-30",
                    source="BEA",
                ),
                BaseRateItem(
                    name="us_10y_yield",
                    description="10-Year Treasury Yield",
                    value=4.17,
                    unit="percent",
                    as_of="2026-01-05",
                    source="Treasury",
                ),
            ]
        )
        assert len(response.base_rates) == 2


class TestSchemaConsistency:
    """Tests for overall schema consistency."""

    def test_all_fixtures_loadable(self):
        """All fixture files load without error."""
        fixtures = [
            "questions",
            "phase1_raw_scenarios",
            "phase2_global_scenarios",
            "phase3_relationships",
            "phase4_quantified_scenarios",
            "phase5_conditionals",
            "phase6_signals",
        ]
        for name in fixtures:
            data = load_fixture(name)
            assert data is not None
            assert len(data) > 0

    def test_fixture_counts_reasonable(self):
        """Fixture sizes match expected pipeline outputs."""
        questions = load_fixture("questions")
        raw_scenarios = load_fixture("phase1_raw_scenarios")
        global_scenarios = load_fixture("phase2_global_scenarios")
        conditionals = load_fixture("phase5_conditionals")

        # 10 questions * 5 scenarios = 50 raw scenarios
        assert len(raw_scenarios) == len(questions) * 5

        # 10 questions * 10 global scenarios = 100 conditionals
        assert len(conditionals) == len(questions) * len(global_scenarios)
