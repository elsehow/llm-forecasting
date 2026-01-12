"""Integration tests for pipeline phases using fixtures."""

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from conditional_trees.models import Question, Scenario, GlobalScenario, Relationship
from conditional_trees.phases.base_rates import fetch_base_rates, format_base_rates_context
from conditional_trees.phases.diverge import diverge
from conditional_trees.phases.converge import converge
from conditional_trees.phases.structure import structure
from conditional_trees.phases.quantify import quantify
from conditional_trees.phases.condition import condition
from conditional_trees.phases.signals import signals
from conditional_trees.phases.condition import _validate_direction_consistency
from conditional_trees.models import QuestionType


class TestPhase0BaseRates:
    """Tests for Phase 0: Base Rates."""

    def test_format_base_rates_context_empty(self):
        """Test formatting empty base rates."""
        assert format_base_rates_context({}) == ""
        assert format_base_rates_context(None) == ""

    def test_format_base_rates_context(self):
        """Test formatting base rates for prompts."""
        base_rates = {
            "us_gdp": {
                "value": 29.0,
                "unit": "trillion_usd",
                "as_of": "2025-09-30",
                "source": "BEA",
            }
        }
        result = format_base_rates_context(base_rates)
        assert "Us Gdp" in result
        assert "29.0" in result
        assert "trillion usd" in result

    @pytest.mark.asyncio
    async def test_fetch_base_rates_with_mock(self, questions):
        """Test fetching base rates with mocked Anthropic client."""
        # Mock response from Anthropic with web search results
        mock_response = MagicMock()
        mock_text_block = MagicMock()
        mock_text_block.text = json.dumps({
            "base_rates": [
                {
                    "name": "us_gdp",
                    "description": "US Real GDP",
                    "value": 29.0,
                    "unit": "trillion_usd",
                    "as_of": "2025-09-30",
                    "source": "BEA",
                }
            ]
        })
        mock_response.content = [mock_text_block]

        with patch("conditional_trees.phases.base_rates.anthropic.Anthropic") as mock_client_class:
            mock_client = MagicMock()
            mock_client.messages.create.return_value = mock_response
            mock_client_class.return_value = mock_client

            result = await fetch_base_rates(questions[:1], verbose=False)

            assert "us_gdp" in result
            assert result["us_gdp"]["value"] == 29.0
            # Verify web search tool was configured
            call_args = mock_client.messages.create.call_args
            assert any(
                t.get("type") == "web_search_20250305"
                for t in call_args.kwargs.get("tools", [])
            )


class TestPhase1Diverge:
    """Tests for Phase 1: Diverge."""

    def test_raw_scenarios_structure(self, raw_scenarios):
        """Verify raw scenarios have correct structure."""
        assert len(raw_scenarios) == 50  # 10 questions * 5 scenarios
        for s in raw_scenarios:
            assert s.name
            assert s.description
            assert len(s.key_assumptions) >= 1
            assert s.source_question_id

    def test_scenarios_per_question(self, raw_scenarios, questions):
        """Verify each question got scenarios."""
        question_ids = {q.id for q in questions}
        scenario_sources = {s.source_question_id for s in raw_scenarios}
        assert scenario_sources == question_ids

    @pytest.mark.asyncio
    async def test_diverge_with_mock(self, questions):
        """Test diverge phase with mocked LLM."""
        mock_response = {
            "scenarios": [
                {
                    "name": "Test Scenario",
                    "description": "A test description",
                    "key_assumptions": ["assumption 1"],
                }
            ]
        }

        with patch("conditional_trees.phases.diverge.llm_call_many") as mock_llm:
            # Return mock response for each question
            mock_llm.return_value = {q.id: mock_response for q in questions}

            result = await diverge(questions, verbose=False)

            assert len(result) == len(questions)  # 1 scenario per question
            assert all(isinstance(s, Scenario) for s in result)


class TestPhase2Converge:
    """Tests for Phase 2: Converge."""

    def test_global_scenarios_structure(self, global_scenarios):
        """Verify global scenarios have correct structure."""
        assert 5 <= len(global_scenarios) <= 12
        for gs in global_scenarios:
            assert gs.id
            assert gs.name
            assert gs.description
            assert isinstance(gs.key_drivers, list)

    def test_global_scenarios_have_ids(self, global_scenarios):
        """Verify all global scenarios have unique IDs."""
        ids = [gs.id for gs in global_scenarios]
        assert len(ids) == len(set(ids))  # All unique

    @pytest.mark.asyncio
    async def test_converge_with_mock(self, raw_scenarios):
        """Test converge phase with mocked LLM."""
        mock_response = {
            "global_scenarios": [
                {
                    "id": "test_scenario",
                    "name": "Test Global Scenario",
                    "description": "A consolidated scenario",
                    "key_drivers": ["driver 1"],
                    "member_scenarios": ["scenario 1"],
                }
            ]
        }

        with patch("conditional_trees.phases.converge.llm_call") as mock_llm:
            mock_llm.return_value = mock_response

            result = await converge(raw_scenarios)

            assert len(result) == 1
            assert result[0].id == "test_scenario"


class TestPhase3Structure:
    """Tests for Phase 3: Structure."""

    def test_relationships_structure(self, relationships):
        """Verify relationships have correct structure."""
        assert len(relationships) > 0
        for r in relationships:
            assert r.scenario_a
            assert r.scenario_b
            assert r.type in ["orthogonal", "correlated", "hierarchical", "mutually_exclusive"]

    def test_relationship_coverage(self, relationships, global_scenarios):
        """Verify relationships cover scenario pairs."""
        scenario_ids = {gs.id for gs in global_scenarios}
        referenced_ids = set()
        for r in relationships:
            referenced_ids.add(r.scenario_a)
            referenced_ids.add(r.scenario_b)

        # All scenarios should be referenced
        assert referenced_ids == scenario_ids

    @pytest.mark.asyncio
    async def test_structure_with_mock(self, global_scenarios):
        """Test structure phase with mocked LLM."""
        mock_response = {
            "relationships": [
                {
                    "scenario_a": global_scenarios[0].id,
                    "scenario_b": global_scenarios[1].id,
                    "type": "orthogonal",
                }
            ]
        }

        with patch("conditional_trees.phases.structure.llm_call") as mock_llm:
            mock_llm.return_value = mock_response

            result = await structure(global_scenarios)

            assert len(result) == 1
            assert result[0].type == "orthogonal"


class TestPhase4Quantify:
    """Tests for Phase 4: Quantify."""

    def test_probabilities_assigned(self, quantified_scenarios):
        """Verify all scenarios have probabilities."""
        for gs in quantified_scenarios:
            assert gs.probability > 0

    def test_probabilities_reasonable(self, quantified_scenarios):
        """Verify probabilities are in valid range."""
        for gs in quantified_scenarios:
            assert 0 < gs.probability <= 1

    @pytest.mark.asyncio
    async def test_quantify_with_mock(self, global_scenarios, relationships):
        """Test quantify phase with mocked LLM."""
        mock_response = {
            "probabilities": [
                {"scenario_id": gs.id, "probability": 0.1}
                for gs in global_scenarios
            ]
        }

        with patch("conditional_trees.phases.quantify.llm_call") as mock_llm:
            mock_llm.return_value = mock_response

            result, prob_result = await quantify(global_scenarios, relationships)

            assert len(result) == len(global_scenarios)
            # Probabilities are normalized, so 0.1 * 10 = 1.0, each stays 0.1
            assert all(gs.probability == 0.1 for gs in result)
            assert prob_result.status == "ok"


class TestPhase5Condition:
    """Tests for Phase 5: Condition."""

    def test_conditionals_structure(self, conditionals):
        """Verify conditionals have correct structure."""
        assert len(conditionals) == 100  # 10 questions * 10 scenarios

    def test_conditionals_coverage(self, conditionals, questions, quantified_scenarios):
        """Verify all question-scenario pairs have forecasts."""
        pairs = {(c.question_id, c.scenario_id) for c in conditionals}
        expected_pairs = {
            (q.id, s.id) for q in questions for s in quantified_scenarios
        }
        assert pairs == expected_pairs

    @pytest.mark.asyncio
    async def test_condition_with_mock(self, questions, quantified_scenarios):
        """Test condition phase with mocked LLM."""
        # Use just first question and scenario for simplicity
        q = questions[0]
        s = quantified_scenarios[0]

        # New batched format: one response per question with all scenarios
        mock_response = {
            q.id: {
                "forecasts": {
                    s.id: {
                        "median": 1000,
                        "ci_80_low": 800,
                        "ci_80_high": 1200,
                        "reasoning": "test",
                    }
                }
            }
        }

        with patch("conditional_trees.phases.condition.llm_call_many") as mock_llm:
            mock_llm.return_value = mock_response

            result = await condition([q], [s], verbose=False)

            assert len(result) == 1
            assert result[0].median == 1000


class TestPhase6Signals:
    """Tests for Phase 6: Signals."""

    def test_signals_structure(self, signals):
        """Verify signals have correct structure."""
        assert len(signals) > 0
        for s in signals:
            assert s.id
            assert s.text
            assert s.resolves_by
            assert s.scenario_id
            assert s.direction in ["increases", "decreases"]
            assert s.magnitude in ["small", "medium", "large"]

    def test_signals_per_scenario(self, signals, quantified_scenarios):
        """Verify each scenario has signals."""
        scenario_ids = {gs.id for gs in quantified_scenarios}
        signal_scenarios = {s.scenario_id for s in signals}
        assert signal_scenarios == scenario_ids

    @pytest.mark.asyncio
    async def test_signals_with_mock(self, quantified_scenarios):
        """Test signals phase with mocked LLM."""
        s = quantified_scenarios[0]
        mock_response = {
            s.id: {
                "signals": [
                    {
                        "text": "Test signal",
                        "resolves_by": "2026-06-01",
                        "direction": "increases",
                        "magnitude": "medium",
                        "current_probability": 0.3,
                    }
                ]
            }
        }

        with patch("conditional_trees.phases.signals.llm_call_many") as mock_llm:
            mock_llm.return_value = mock_response

            result = await signals([s], verbose=False)

            assert len(result) == 1
            assert result[0].direction == "increases"


class TestDirectionConsistencyValidation:
    """Tests for Bracket-style direction consistency validation."""

    @pytest.mark.parametrize("q_type,key,dirs,vals,expected_violations,check_strs", [
        # Valid orderings (no violations)
        (QuestionType.BINARY, "probability",
         {"a": "increases", "b": "neutral", "c": "decreases"}, [0.7, 0.5, 0.3], 0, []),
        (QuestionType.CONTINUOUS, "median",
         {"a": "increases", "b": "neutral", "c": "decreases"}, [100, 50, 25], 0, []),
        (QuestionType.BINARY, "probability",
         {"a": "neutral", "b": "neutral", "c": "neutral"}, [0.5, 0.6, 0.4], 0, []),
        # Increases below neutral
        (QuestionType.BINARY, "probability",
         {"a": "increases", "b": "neutral"}, [0.4, 0.5], 1, ["a", "increases"]),
        # Neutral below decreases
        (QuestionType.BINARY, "probability",
         {"a": "increases", "b": "neutral", "c": "decreases"}, [0.7, 0.3, 0.4], 1, ["b", "c"]),
        # Increases below decreases
        (QuestionType.BINARY, "probability",
         {"a": "increases", "b": "decreases"}, [0.2, 0.6], 1, ["a", "b"]),
        # Continuous invalid
        (QuestionType.CONTINUOUS, "median",
         {"a": "increases", "b": "neutral"}, [30, 50], 1, ["a"]),
    ])
    def test_binary_continuous_validation(self, q_type, key, dirs, vals, expected_violations, check_strs):
        """Test binary/continuous direction validation with parameterized cases."""
        forecasts = {sid: {key: val} for sid, val in zip(dirs.keys(), vals)}
        violations = _validate_direction_consistency(dirs, forecasts, q_type)
        assert len(violations) == expected_violations
        for s in check_strs:
            assert any(s in v for v in violations)

    @pytest.mark.parametrize("dirs,probs_a,expected_violations,check_strs", [
        # Valid: ai_scenario has higher P(Option A)
        ({"ai": "Option A", "base": "neutral"}, [0.7, 0.5], 0, []),
        # Invalid: ai_scenario has lower P(Option A)
        ({"ai": "Option A", "base": "neutral"}, [0.4, 0.5], 1, ["ai", "Option A"]),
        # Invalid option name
        ({"ai": "Option C", "base": "neutral"}, [0.5, 0.5], 1, ["Option C", "not in options"]),
    ])
    def test_categorical_validation(self, dirs, probs_a, expected_violations, check_strs):
        """Test categorical direction validation."""
        forecasts = {
            list(dirs.keys())[0]: {"probabilities": {"Option A": probs_a[0], "Option B": 1 - probs_a[0]}},
            list(dirs.keys())[1]: {"probabilities": {"Option A": probs_a[1], "Option B": 1 - probs_a[1]}},
        }
        violations = _validate_direction_consistency(
            dirs, forecasts, QuestionType.CATEGORICAL, options=["Option A", "Option B"]
        )
        assert len(violations) == expected_violations
        for s in check_strs:
            assert any(s in v for v in violations)

    def test_missing_forecast_is_skipped(self):
        """Test that missing forecasts don't cause errors."""
        violations = _validate_direction_consistency(
            {"exists": "increases", "missing": "neutral"},
            {"exists": {"probability": 0.7}},
            QuestionType.BINARY
        )
        assert violations == []
