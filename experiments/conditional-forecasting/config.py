"""Shared configuration for conditional forecasting experiments.

This module contains constants shared across experiments to ensure consistency.
"""

from datetime import date

# Model knowledge cutoff date
#
# Used to filter questions that resolved AFTER this date, avoiding data contamination
# where models might have seen resolutions in their training data.
#
# Rationale: GPT-5.2 was released 2025-12-11. Its knowledge cutoff was likely
# several months prior. We use 2025-10-01 as a conservative cutoff that should
# be after all major frontier models' training data.
#
# This affects:
# - Which question pairs we use for evaluation (both must resolve after cutoff)
# - Which price history data counts as "post-resolution" vs "pre-resolution"
#
# Models this is designed to be safe for:
# - GPT-5.2 (released 2025-12-11)
# - Claude Sonnet 4 (released 2025-05)
# - Claude Opus 4.5 (released 2025-11)
#
MODEL_KNOWLEDGE_CUTOFF = date(2025, 10, 1)

# Default model for experiments
DEFAULT_MODEL = "claude-sonnet-4-20250514"
