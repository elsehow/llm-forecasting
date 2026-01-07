"""Configuration management for ForecastBench.

Uses pydantic-settings to load configuration from environment variables
and .env files. In production (Cloud Run), secrets are injected as
environment variables from GCP Secret Manager.

Usage:
    from llm_forecasting.config import settings

    # Access API keys
    if settings.fred_api_key:
        source = FREDSource(api_key=settings.fred_api_key)

    # Access LLM keys (used by litellm automatically via env vars)
    # ANTHROPIC_API_KEY, OPENAI_API_KEY, etc.
"""

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables.

    All settings are optional and have sensible defaults. API keys
    that are not set will cause their respective features to be
    disabled or to use unauthenticated access where possible.

    Environment variables can be set directly or via a .env file
    in the project root.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # === Data Source API Keys ===
    # FRED (Federal Reserve Economic Data)
    fred_api_key: str | None = None

    # Metaculus (optional - public API works without key)
    metaculus_api_key: str | None = None

    # ACLED (Armed Conflict Location & Event Data)
    acled_api_email: str | None = None
    acled_api_key: str | None = None

    # === LLM Provider API Keys ===
    # These are also read directly by litellm from environment
    anthropic_api_key: str | None = None
    openai_api_key: str | None = None
    google_api_key: str | None = None  # Gemini
    mistral_api_key: str | None = None
    togetherai_api_key: str | None = None
    xai_api_key: str | None = None

    # === Storage Configuration ===
    # SQLite database path (default: in-memory for tests)
    database_url: str = "sqlite+aiosqlite:///forecastbench.db"

    # GCP configuration (for cloud storage sync)
    gcp_project_id: str | None = None
    gcp_bucket_name: str | None = None

    # === Application Settings ===
    # Logging level
    log_level: str = "INFO"

    # Default model for forecasting
    default_model: str = "claude-sonnet-4-20250514"


# Singleton instance - import this
settings = Settings()
