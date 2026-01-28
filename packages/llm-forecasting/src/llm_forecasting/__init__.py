"""ForecastBench v2 - A dynamic benchmark for LLM forecasting accuracy."""

from llm_forecasting.models import (
    Forecast,
    ForecastScore,
    Question,
    QuestionType,
    Resolution,
    SourceType,
)
from llm_forecasting.registry import (
    MODEL_REGISTRY,
    ModelInfo,
    Organization,
    get_active_models,
    get_model,
    get_model_by_litellm_id,
    get_models_by_org,
    get_models_released_before,
    get_models_with_structured_output,
)
from llm_forecasting.voi import (
    entropy,
    entropy_voi,
    entropy_voi_from_rho,
    entropy_voi_normalized,
    entropy_voi_normalized_from_rho,
    estimate_rho,
    estimate_rho_market_aware,
    estimate_rho_two_step,
    estimate_rho_two_step_batch,
    linear_voi,
    linear_voi_from_rho,
    rho_to_posteriors,
)

__all__ = [
    # Core models
    "Question",
    "QuestionType",
    "SourceType",
    "Forecast",
    "Resolution",
    "ForecastScore",
    # Registry
    "MODEL_REGISTRY",
    "ModelInfo",
    "Organization",
    "get_model",
    "get_model_by_litellm_id",
    "get_models_by_org",
    "get_active_models",
    "get_models_released_before",
    "get_models_with_structured_output",
    # VOI
    "linear_voi",
    "entropy_voi",
    "entropy_voi_normalized",
    "entropy",
    "rho_to_posteriors",
    "linear_voi_from_rho",
    "entropy_voi_from_rho",
    "entropy_voi_normalized_from_rho",
    "estimate_rho",
    "estimate_rho_market_aware",
    "estimate_rho_two_step",
    "estimate_rho_two_step_batch",
]
__version__ = "2.0.0"
