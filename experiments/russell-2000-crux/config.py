"""Configuration for Russell 2000 Crux Generation experiment."""

from datetime import date

# Model knowledge cutoff - use dates AFTER this to avoid contamination
MODEL_KNOWLEDGE_CUTOFF = date(2025, 10, 1)

# Date range for experiment (post-cutoff)
START_DATE = date(2025, 11, 1)
END_DATE = date(2026, 1, 23)

# Models
CRUX_GENERATION_MODEL = "anthropic/claude-sonnet-4-20250514"
RHO_ESTIMATION_MODEL = "anthropic/claude-3-haiku-20240307"  # Cheap for bulk
CONDITIONAL_MODEL = "anthropic/claude-3-haiku-20240307"

# Phase 0 settings
PILOT_N_STOCKS = 50  # Stocks with earnings in period
PILOT_N_CRUXES = 5   # Cruxes per stock-day

# Phase 2 settings
FULL_N_STOCKS = 100
FULL_N_CRUXES = 10
