"""End-to-end tests for the full pipeline."""

import json
from pathlib import Path
from unittest.mock import patch, AsyncMock

import pytest

from tree_of_life.pipeline import build_forecast_tree, load_questions, save_tree
from tree_of_life.models import ForecastTree, Question

# Path to package root (parent of tests/)
PACKAGE_ROOT = Path(__file__).parent.parent


class TestPipelineHelpers:
    """Tests for pipeline helper functions."""

    def test_load_questions(self):
        """Test loading questions from JSON."""
        questions_path = PACKAGE_ROOT / "examples" / "fri_questions.json"
        questions = load_questions(str(questions_path))
        assert len(questions) == 20
        assert all(isinstance(q, Question) for q in questions)
        # Verify units are loaded
        assert all(q.unit is not None for q in questions)

    def test_save_tree(self, tmp_path, questions, quantified_scenarios, relationships, conditionals, signals, raw_scenarios):
        """Test saving forecast tree to JSON."""
        tree = ForecastTree(
            questions=questions,
            raw_scenarios=raw_scenarios,
            global_scenarios=quantified_scenarios,
            relationships=relationships,
            conditionals=conditionals,
            signals=signals,
        )

        output_path = tmp_path / "test_tree.json"
        save_tree(tree, output_path)

        assert output_path.exists()
        with open(output_path) as f:
            data = json.load(f)
        assert "questions" in data
        assert "global_scenarios" in data


class TestFullPipeline:
    """End-to-end tests for the pipeline."""

    @pytest.mark.asyncio
    async def test_full_pipeline_with_mocks(
        self,
        questions,
        raw_scenarios,
        global_scenarios,
        relationships,
        quantified_scenarios,
        conditionals,
        signals,
    ):
        """Test full pipeline with all phases mocked."""
        # Build mock responses for each phase
        diverge_response = {
            q.id: {
                "scenarios": [
                    {
                        "name": s.name,
                        "description": s.description,
                        "key_assumptions": s.key_assumptions,
                    }
                    for s in raw_scenarios
                    if s.source_question_id == q.id
                ]
            }
            for q in questions
        }

        converge_response = {
            "global_scenarios": [
                {
                    "id": gs.id,
                    "name": gs.name,
                    "description": gs.description,
                    "key_drivers": gs.key_drivers,
                    "member_scenarios": gs.member_scenarios,
                }
                for gs in global_scenarios
            ]
        }

        structure_response = {
            "relationships": [
                {
                    "scenario_a": r.scenario_a,
                    "scenario_b": r.scenario_b,
                    "type": r.type,
                    "strength": r.strength,
                    "notes": r.notes,
                }
                for r in relationships
            ]
        }

        quantify_response = {
            "probabilities": [
                {"scenario_id": gs.id, "probability": gs.probability}
                for gs in quantified_scenarios
            ]
        }

        # New batched format: one response per question with all scenarios
        condition_response = {}
        for q in questions:
            forecasts = {}
            for c in conditionals:
                if c.question_id == q.id:
                    if hasattr(c, "median"):
                        forecasts[c.scenario_id] = {
                            "median": c.median,
                            "ci_80_low": c.ci_80_low,
                            "ci_80_high": c.ci_80_high,
                            "reasoning": c.reasoning,
                        }
                    elif hasattr(c, "probabilities"):
                        forecasts[c.scenario_id] = {
                            "probabilities": c.probabilities,
                            "reasoning": c.reasoning,
                        }
            condition_response[q.id] = {"forecasts": forecasts}

        signals_response = {}
        for s in signals:
            if s.scenario_id not in signals_response:
                signals_response[s.scenario_id] = {"signals": []}
            signals_response[s.scenario_id]["signals"].append({
                "text": s.text,
                "resolves_by": str(s.resolves_by),
                "direction": s.direction,
                "magnitude": s.magnitude,
                "current_probability": s.current_probability,
            })

        # Mock base rates response
        base_rates_response = {
            "us_gdp": {"value": 29.0, "unit": "trillion_usd", "as_of": "2025-09-30", "source": "BEA"}
        }

        # Patch all LLM calls including Phase 0
        with patch("tree_of_life.pipeline.fetch_base_rates", return_value=base_rates_response), \
             patch("tree_of_life.phases.diverge.llm_call_many", return_value=diverge_response), \
             patch("tree_of_life.phases.converge.llm_call", return_value=converge_response), \
             patch("tree_of_life.phases.structure.llm_call", return_value=structure_response), \
             patch("tree_of_life.phases.quantify.llm_call", return_value=quantify_response), \
             patch("tree_of_life.phases.condition.llm_call_many", return_value=condition_response), \
             patch("tree_of_life.phases.signals.llm_call_many", return_value=signals_response):

            tree = await build_forecast_tree(questions, verbose=False)

            # Verify tree structure
            assert len(tree.questions) == 10
            assert len(tree.raw_scenarios) == 50
            assert len(tree.global_scenarios) == 10
            assert len(tree.relationships) == 45
            assert len(tree.conditionals) == 100
            assert len(tree.signals) == 50

            # Verify probability diagnostics
            assert tree.raw_probability_sum is not None
            assert tree.probability_status in ["ok", "warning", "suspect"]

    def test_fixture_data_consistency(
        self,
        questions,
        raw_scenarios,
        global_scenarios,
        relationships,
        quantified_scenarios,
        conditionals,
        signals,
    ):
        """Verify fixture data is internally consistent."""
        # Raw scenarios reference valid questions
        question_ids = {q.id for q in questions}
        for s in raw_scenarios:
            assert s.source_question_id in question_ids

        # Relationships reference valid scenarios
        scenario_ids = {gs.id for gs in global_scenarios}
        for r in relationships:
            assert r.scenario_a in scenario_ids
            assert r.scenario_b in scenario_ids

        # Conditionals reference valid questions and scenarios
        quantified_ids = {gs.id for gs in quantified_scenarios}
        for c in conditionals:
            assert c.question_id in question_ids
            assert c.scenario_id in quantified_ids

        # Signals reference valid scenarios
        for s in signals:
            assert s.scenario_id in quantified_ids
