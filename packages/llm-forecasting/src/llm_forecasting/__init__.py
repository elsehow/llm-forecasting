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
]
__version__ = "2.0.0"
