"""Model registry for ForecastBench.

Defines metadata for all models that can be evaluated, including:
- LiteLLM model identifier for API calls
- Organization/provider information
- Release date (for leaderboard filtering)
- Context window size
- Capability flags (structured output support, reasoning model)

Usage:
    from llm_forecasting.registry import MODEL_REGISTRY, get_model, get_models_by_org

    # Get a specific model
    model = get_model("claude-opus-4-5-20251101")
    forecaster = LLMForecaster(model=model.litellm_id)

    # Get all models from an organization
    anthropic_models = get_models_by_org("Anthropic")
"""

from datetime import date
from enum import Enum

from pydantic import BaseModel, Field


class Organization(str, Enum):
    """Model provider organizations."""

    ANTHROPIC = "Anthropic"
    OPENAI = "OpenAI"
    GOOGLE = "Google"
    MISTRAL = "Mistral AI"
    XAI = "xAI"
    META = "Meta"
    DEEPSEEK = "DeepSeek"
    QWEN = "Qwen"
    MOONSHOT = "Moonshot"
    ZAI = "Z.ai"


class ModelInfo(BaseModel):
    """Metadata for a model to be evaluated.

    Attributes:
        id: Short identifier used as registry key (e.g., "claude-opus-4-5-20251101")
        litellm_id: Full LiteLLM model identifier (e.g., "anthropic/claude-opus-4-5-20251101")
        organization: The organization that created the model
        release_date: When the model was publicly released (for leaderboard filtering)
        context_window: Maximum token limit for the model
        supports_structured_output: Whether the model supports response_format for JSON schemas
        is_reasoning_model: Whether this is a reasoning/thinking model (o1/o3 class, DeepSeek-R1, etc.)
        active: Whether to include this model in benchmark runs
    """

    id: str = Field(description="Short identifier used as registry key")
    litellm_id: str = Field(description="Full LiteLLM model identifier for API calls")
    organization: Organization = Field(description="Organization that created the model")
    release_date: date | None = Field(default=None, description="Public release date")
    context_window: int = Field(description="Maximum context window in tokens")
    supports_structured_output: bool = Field(
        default=True, description="Whether response_format works with this model"
    )
    is_reasoning_model: bool = Field(
        default=False, description="Whether this is a reasoning/thinking model"
    )
    active: bool = Field(default=True, description="Whether to include in benchmark runs")


# =============================================================================
# Model Registry
# =============================================================================

MODEL_REGISTRY: dict[str, ModelInfo] = {
    # =========================================================================
    # Anthropic Models
    # =========================================================================
    "claude-opus-4-5-20251101": ModelInfo(
        id="claude-opus-4-5-20251101",
        litellm_id="anthropic/claude-opus-4-5-20251101",
        organization=Organization.ANTHROPIC,
        release_date=date(2025, 11, 1),
        context_window=200000,
        supports_structured_output=True,
    ),
    "claude-sonnet-4-5-20250929": ModelInfo(
        id="claude-sonnet-4-5-20250929",
        litellm_id="anthropic/claude-sonnet-4-5-20250929",
        organization=Organization.ANTHROPIC,
        release_date=date(2025, 9, 29),
        context_window=200000,
        supports_structured_output=True,
    ),
    "claude-haiku-4-5-20251001": ModelInfo(
        id="claude-haiku-4-5-20251001",
        litellm_id="anthropic/claude-haiku-4-5-20251001",
        organization=Organization.ANTHROPIC,
        release_date=date(2025, 10, 1),
        context_window=200000,
        supports_structured_output=True,
    ),
    "claude-opus-4-1-20250805": ModelInfo(
        id="claude-opus-4-1-20250805",
        litellm_id="anthropic/claude-opus-4-1-20250805",
        organization=Organization.ANTHROPIC,
        release_date=date(2025, 8, 5),
        context_window=200000,
        supports_structured_output=True,
    ),
    "claude-3-7-sonnet-20250219": ModelInfo(
        id="claude-3-7-sonnet-20250219",
        litellm_id="anthropic/claude-3-7-sonnet-20250219",
        organization=Organization.ANTHROPIC,
        release_date=date(2025, 2, 19),
        context_window=200000,
        supports_structured_output=True,
    ),
    # =========================================================================
    # OpenAI Models
    # =========================================================================
    "gpt-5.2-2025-12-11": ModelInfo(
        id="gpt-5.2-2025-12-11",
        litellm_id="openai/gpt-5.2-2025-12-11",
        organization=Organization.OPENAI,
        release_date=date(2025, 12, 11),
        context_window=128000,
        supports_structured_output=True,
        is_reasoning_model=True,
    ),
    "gpt-5.1-2025-11-13": ModelInfo(
        id="gpt-5.1-2025-11-13",
        litellm_id="openai/gpt-5.1-2025-11-13",
        organization=Organization.OPENAI,
        release_date=date(2025, 11, 13),
        context_window=128000,
        supports_structured_output=True,
        is_reasoning_model=True,
    ),
    "gpt-5-mini-2025-08-07": ModelInfo(
        id="gpt-5-mini-2025-08-07",
        litellm_id="openai/gpt-5-mini-2025-08-07",
        organization=Organization.OPENAI,
        release_date=date(2025, 8, 7),
        context_window=128000,
        supports_structured_output=True,
        is_reasoning_model=True,
    ),
    "gpt-5-nano-2025-08-07": ModelInfo(
        id="gpt-5-nano-2025-08-07",
        litellm_id="openai/gpt-5-nano-2025-08-07",
        organization=Organization.OPENAI,
        release_date=date(2025, 8, 7),
        context_window=128000,
        supports_structured_output=True,
        is_reasoning_model=True,
    ),
    "gpt-4.1-2025-04-14": ModelInfo(
        id="gpt-4.1-2025-04-14",
        litellm_id="openai/gpt-4.1-2025-04-14",
        organization=Organization.OPENAI,
        release_date=date(2025, 4, 14),
        context_window=128000,
        supports_structured_output=True,
        is_reasoning_model=False,
    ),
    "gpt-4o-mini": ModelInfo(
        id="gpt-4o-mini",
        litellm_id="openai/gpt-4o-mini",
        organization=Organization.OPENAI,
        release_date=date(2024, 7, 18),
        context_window=128000,
        supports_structured_output=True,
        is_reasoning_model=False,
        active=False,  # Older model, kept for reference
    ),
    # =========================================================================
    # Google Models
    # =========================================================================
    "gemini-3-pro-preview": ModelInfo(
        id="gemini-3-pro-preview",
        litellm_id="gemini/gemini-3-pro-preview",
        organization=Organization.GOOGLE,
        release_date=date(2025, 11, 18),
        context_window=1048576,
        supports_structured_output=True,
    ),
    "gemini-2.5-pro": ModelInfo(
        id="gemini-2.5-pro",
        litellm_id="gemini/gemini-2.5-pro",
        organization=Organization.GOOGLE,
        release_date=date(2025, 6, 17),
        context_window=1048576,
        supports_structured_output=True,
    ),
    "gemini-3-flash-preview": ModelInfo(
        id="gemini-3-flash-preview",
        litellm_id="gemini/gemini-3-flash-preview",
        organization=Organization.GOOGLE,
        release_date=date(2025, 12, 17),
        context_window=1048576,
        supports_structured_output=True,
    ),
    "gemini-2.0-flash": ModelInfo(
        id="gemini-2.0-flash",
        litellm_id="gemini/gemini-2.0-flash",
        organization=Organization.GOOGLE,
        release_date=date(2024, 12, 11),
        context_window=1048576,
        supports_structured_output=True,
        active=False,  # Older model, kept for testing
    ),
    # =========================================================================
    # xAI Models
    # =========================================================================
    "grok-4-fast-reasoning": ModelInfo(
        id="grok-4-fast-reasoning",
        litellm_id="xai/grok-4-fast-reasoning",
        organization=Organization.XAI,
        release_date=date(2025, 9, 19),
        context_window=2000000,
        supports_structured_output=True,
        is_reasoning_model=True,
    ),
    "grok-4-1-fast-reasoning": ModelInfo(
        id="grok-4-1-fast-reasoning",
        litellm_id="xai/grok-4-1-fast-reasoning",
        organization=Organization.XAI,
        release_date=date(2025, 11, 17),
        context_window=2000000,
        supports_structured_output=True,
        is_reasoning_model=True,
    ),
    "grok-4-fast-non-reasoning": ModelInfo(
        id="grok-4-fast-non-reasoning",
        litellm_id="xai/grok-4-fast-non-reasoning",
        organization=Organization.XAI,
        release_date=date(2025, 9, 19),
        context_window=2000000,
        supports_structured_output=True,
        is_reasoning_model=False,
    ),
    "grok-4-1-fast-non-reasoning": ModelInfo(
        id="grok-4-1-fast-non-reasoning",
        litellm_id="xai/grok-4-1-fast-non-reasoning",
        organization=Organization.XAI,
        release_date=date(2025, 11, 17),
        context_window=2000000,
        supports_structured_output=True,
        is_reasoning_model=False,
    ),
    "grok-4-0709": ModelInfo(
        id="grok-4-0709",
        litellm_id="xai/grok-4-0709",
        organization=Organization.XAI,
        release_date=date(2025, 7, 9),
        context_window=256000,
        supports_structured_output=True,
    ),
    # =========================================================================
    # Mistral Models
    # =========================================================================
    "mistral-large-2411": ModelInfo(
        id="mistral-large-2411",
        litellm_id="mistral/mistral-large-2411",
        organization=Organization.MISTRAL,
        release_date=date(2024, 11, 18),
        context_window=128000,
        supports_structured_output=True,
    ),
    "mistral-small-latest": ModelInfo(
        id="mistral-small-latest",
        litellm_id="mistral/mistral-small-latest",
        organization=Organization.MISTRAL,
        release_date=date(2024, 9, 1),
        context_window=32000,
        supports_structured_output=True,
        active=False,  # Smaller model, kept for testing
    ),
    # =========================================================================
    # Together AI Hosted Models (DeepSeek, Qwen, etc.)
    # =========================================================================
    "deepseek-v3.1": ModelInfo(
        id="deepseek-v3.1",
        litellm_id="together_ai/deepseek-ai/DeepSeek-V3.1",
        organization=Organization.DEEPSEEK,
        release_date=date(2025, 8, 21),
        context_window=128000,
        supports_structured_output=False,  # Together AI doesn't support structured output
    ),
    "qwen3-235b-a22b-fp8-tput": ModelInfo(
        id="qwen3-235b-a22b-fp8-tput",
        litellm_id="together_ai/Qwen/Qwen3-235B-A22B-fp8-tput",
        organization=Organization.QWEN,
        release_date=date(2025, 4, 29),
        context_window=40960,
        supports_structured_output=False,
    ),
    "qwen3-235b-a22b-thinking-2507": ModelInfo(
        id="qwen3-235b-a22b-thinking-2507",
        litellm_id="together_ai/Qwen/Qwen3-235B-A22B-Thinking-2507",
        organization=Organization.QWEN,
        release_date=date(2025, 7, 25),
        context_window=262144,
        supports_structured_output=False,
        is_reasoning_model=True,
    ),
    "kimi-k2-thinking": ModelInfo(
        id="kimi-k2-thinking",
        litellm_id="together_ai/moonshotai/Kimi-K2-Thinking",
        organization=Organization.MOONSHOT,
        release_date=date(2025, 11, 5),
        context_window=262144,
        supports_structured_output=False,
        is_reasoning_model=True,
    ),
    "kimi-k2-instruct-0905": ModelInfo(
        id="kimi-k2-instruct-0905",
        litellm_id="together_ai/moonshotai/Kimi-K2-Instruct-0905",
        organization=Organization.MOONSHOT,
        release_date=date(2025, 9, 5),
        context_window=262144,
        supports_structured_output=False,
    ),
    "glm-4.6": ModelInfo(
        id="glm-4.6",
        litellm_id="together_ai/zai-org/GLM-4.6",
        organization=Organization.ZAI,
        release_date=date(2025, 11, 13),
        context_window=202752,
        supports_structured_output=False,
    ),
}


# =============================================================================
# Helper Functions
# =============================================================================


def get_model(model_id: str) -> ModelInfo:
    """Get model info by ID.

    Args:
        model_id: The model identifier (e.g., "claude-opus-4-5-20251101")

    Returns:
        ModelInfo for the requested model

    Raises:
        KeyError: If model_id is not in the registry
    """
    if model_id not in MODEL_REGISTRY:
        raise KeyError(f"Model '{model_id}' not found in registry. Available: {list(MODEL_REGISTRY.keys())}")
    return MODEL_REGISTRY[model_id]


def get_model_by_litellm_id(litellm_id: str) -> ModelInfo | None:
    """Get model info by LiteLLM identifier.

    Args:
        litellm_id: The LiteLLM model identifier (e.g., "anthropic/claude-opus-4-5-20251101")

    Returns:
        ModelInfo if found, None otherwise
    """
    for model in MODEL_REGISTRY.values():
        if model.litellm_id == litellm_id:
            return model
    return None


def get_models_by_org(organization: Organization | str) -> list[ModelInfo]:
    """Get all models from an organization.

    Args:
        organization: Organization enum value or string name

    Returns:
        List of ModelInfo for models from that organization
    """
    if isinstance(organization, str):
        organization = Organization(organization)
    return [m for m in MODEL_REGISTRY.values() if m.organization == organization]


def get_active_models() -> list[ModelInfo]:
    """Get all active models for benchmark runs.

    Returns:
        List of ModelInfo where active=True
    """
    return [m for m in MODEL_REGISTRY.values() if m.active]


def get_models_released_before(cutoff_date: date) -> list[ModelInfo]:
    """Get models released before a cutoff date.

    Useful for leaderboard filtering to only include models
    that were available at the time of the forecast.

    Args:
        cutoff_date: Only include models released before this date

    Returns:
        List of ModelInfo with release_date before cutoff
    """
    return [
        m
        for m in MODEL_REGISTRY.values()
        if m.release_date is not None and m.release_date < cutoff_date
    ]


def get_models_with_structured_output() -> list[ModelInfo]:
    """Get models that support structured output.

    Returns:
        List of ModelInfo where supports_structured_output=True
    """
    return [m for m in MODEL_REGISTRY.values() if m.supports_structured_output]
