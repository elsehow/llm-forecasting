"""Evaluation runner for LLM forecasting agents.

This module orchestrates running agents against question sets and collecting results.

TODO: Implement
- run_evaluation(agents, questions, storage) -> EvaluationResult
- resume_evaluation(evaluation_id, storage) -> EvaluationResult
- parallel execution with rate limiting
"""

from dataclasses import dataclass


@dataclass
class EvaluationConfig:
    """Configuration for an evaluation run."""

    question_source: str  # e.g., "metaculus", "polymarket"
    sample_size: int | None = None
    forecasters: list[str] | None = None  # Model names to evaluate
    parallel: bool = True
    max_concurrent: int = 5


@dataclass
class EvaluationResult:
    """Results from an evaluation run."""

    config: EvaluationConfig
    num_questions: int
    num_forecasts: int
    scores_by_forecaster: dict[str, list[float]]


async def run_evaluation(config: EvaluationConfig) -> EvaluationResult:
    """Run an evaluation.

    TODO: Implement
    """
    raise NotImplementedError("Evaluation runner not yet implemented")
