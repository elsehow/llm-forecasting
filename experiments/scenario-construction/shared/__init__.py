"""Shared utilities for scenario construction experiments."""

from pathlib import Path

# Common paths
REPO_ROOT = Path(__file__).parent.parent.parent.parent
CONDITIONAL_DATA = REPO_ROOT / "experiments" / "conditional-forecasting" / "data"
VOI_VALIDATION = REPO_ROOT / "experiments" / "question-generation" / "voi-validation"

# Target configuration
from .config import (
    TargetConfig,
    TARGETS,
    get_target,
    CARNEY_PM_2027,
    DEMOCRAT_WHITEHOUSE_2028,
    US_GDP_2029,
)

# Signal utilities
from .signals import (
    RESOLVABILITY_REQUIREMENTS,
    load_market_signals_semantic,
    deduplicate_signals,
    deduplicate_market_signals,
    # Resolution date utilities
    DEFAULT_MAX_HORIZON_DAYS,
    categorize_signal,
    filter_by_resolution_date,
)
