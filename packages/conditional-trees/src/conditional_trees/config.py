"""Configuration for conditional forecasting pipeline."""

import logging
import os
from datetime import date
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# Logging Configuration
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
LOG_FILE = os.getenv("LOG_FILE", "output/pipeline.log")


def setup_logging(log_file: str | None = None):
    """Configure logging to file and console.

    Args:
        log_file: Path to log file. If None, uses LOG_FILE from env.
    """
    log_file = log_file or LOG_FILE

    # Create output directory if needed
    log_path = Path(log_file)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    # Configure root logger
    logging.basicConfig(
        level=getattr(logging, LOG_LEVEL.upper()),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(),  # Also print to console
        ],
        force=True,  # Allow reconfiguration
    )

    # Quiet down noisy libraries
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("anthropic").setLevel(logging.WARNING)

    return log_file

# LLM Configuration
# Structured outputs require Claude 4.5/4.1 models
STRUCTURED_OUTPUTS_BETA = "structured-outputs-2025-11-13"

# Tiered model selection by phase
# Opus for high-stakes reasoning (calibration matters)
MODEL_OPUS = os.getenv("MODEL_OPUS", "claude-opus-4-5-20251101")
# Sonnet for generation and extraction
MODEL_SONNET = os.getenv("MODEL_SONNET", "claude-sonnet-4-5-20250929")

# Phase-specific model assignments
MODEL_BASE_RATES = MODEL_SONNET   # Phase 0: web search + extraction
MODEL_DIVERGE = MODEL_SONNET      # Phase 1: creative scenario generation
MODEL_CONVERGE = MODEL_SONNET     # Phase 2: clustering/consolidation
MODEL_STRUCTURE = MODEL_OPUS      # Phase 3: relationship analysis (calibration critical)
MODEL_QUANTIFY = MODEL_OPUS       # Phase 4: probability assignments (calibration critical)
MODEL_CONDITION = MODEL_SONNET    # Phase 5: applying scenarios to questions
MODEL_SIGNALS = MODEL_SONNET      # Phase 6: signal extraction

# Legacy: default model for backwards compatibility
MODEL = MODEL_SONNET

# Pipeline Parameters
N_SCENARIOS_PER_QUESTION = int(os.getenv("N_SCENARIOS_PER_QUESTION", "5"))
MAX_GLOBAL_SCENARIOS = int(os.getenv("MAX_GLOBAL_SCENARIOS", "10"))
SIGNAL_HORIZON_DATE = os.getenv("SIGNAL_HORIZON_DATE", "2028-12-31")

# Batch mode - default to true for 50% cost savings
# Set USE_BATCH_API=false for faster iteration during development
USE_BATCH_API = os.getenv("USE_BATCH_API", "true").lower() == "true"

# Date parameters for scenario generation
# START_DATE: Reference date for "today" - prevents scenarios with past events
# FORECAST_HORIZON: End of forecast window
START_DATE = os.getenv("START_DATE", date.today().isoformat())
FORECAST_HORIZON = os.getenv("FORECAST_HORIZON", "2040-12-31")
