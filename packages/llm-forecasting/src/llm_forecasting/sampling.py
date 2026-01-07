"""Question sampling infrastructure for creating balanced question sets.

This module provides tools for sampling questions to create evaluation sets
that are balanced across:
- Sources (market vs data)
- Categories
- Resolution dates
- Base rates (to avoid skew toward 0%, 50%, or 100%)
"""

import random
from collections import defaultdict
from dataclasses import dataclass
from datetime import date, timedelta
from enum import Enum

from llm_forecasting.models import Question, SourceType


class SamplingStrategy(str, Enum):
    """Strategies for sampling questions."""

    UNIFORM = "uniform"  # Equal weight to all questions
    BALANCED_CATEGORY = "balanced_category"  # Balance across categories
    BALANCED_RESOLUTION_DATE = "balanced_resolution_date"  # Balance resolution dates
    BALANCED_BASE_RATE = "balanced_base_rate"  # Balance base rate distribution
    STRATIFIED = "stratified"  # Balance across all dimensions


@dataclass
class SamplingConfig:
    """Configuration for question sampling."""

    # Total questions to sample
    num_questions: int = 500

    # Split between market and data sources
    market_fraction: float = 0.5

    # Maximum resolution horizon (filter out questions > N days)
    max_resolution_days: int | None = 365

    # Base rate distribution bins (for balancing)
    # Default: 5 bins [0-0.2, 0.2-0.4, 0.4-0.6, 0.6-0.8, 0.8-1.0]
    base_rate_bins: int = 5

    # Resolution date bins (for balancing year-end clustering)
    resolution_date_bins: int = 12  # Monthly

    # Minimum questions per source (ensure diversity)
    min_per_source: int = 10

    # Categories to include (None = all)
    include_categories: list[str] | None = None

    # Categories to exclude
    exclude_categories: list[str] = None

    def __post_init__(self):
        if self.exclude_categories is None:
            self.exclude_categories = ["Other"]


@dataclass
class SamplingResult:
    """Result of question sampling with diagnostics."""

    questions: list[Question]

    # Diagnostics
    source_counts: dict[str, int]
    category_counts: dict[str, int]
    resolution_date_distribution: dict[str, int]
    base_rate_distribution: dict[str, int]

    # Warnings about underrepresented groups
    warnings: list[str]


class QuestionSampler:
    """Sample questions with configurable balancing strategies."""

    def __init__(self, config: SamplingConfig | None = None):
        self.config = config or SamplingConfig()

    def filter_questions(
        self,
        questions: list[Question],
        reference_date: date | None = None,
    ) -> list[Question]:
        """Apply filters to question pool before sampling.

        Args:
            questions: Full pool of questions
            reference_date: Date to use for resolution horizon calculation (default: today)

        Returns:
            Filtered list of questions
        """
        if reference_date is None:
            reference_date = date.today()

        filtered = []
        for q in questions:
            # Skip resolved questions
            if q.resolved:
                continue

            # Skip excluded categories
            if q.category and q.category in self.config.exclude_categories:
                continue

            # Filter by included categories if specified
            if self.config.include_categories:
                if q.category not in self.config.include_categories:
                    continue

            # Filter by resolution horizon
            if self.config.max_resolution_days and q.resolution_date:
                days_until = (q.resolution_date - reference_date).days
                if days_until > self.config.max_resolution_days:
                    continue
                if days_until < 0:
                    continue  # Already resolved

            filtered.append(q)

        return filtered

    def _get_base_rate_bin(self, base_rate: float | None) -> int:
        """Get the bin index for a base rate."""
        if base_rate is None:
            return self.config.base_rate_bins // 2  # Default to middle bin

        bin_size = 1.0 / self.config.base_rate_bins
        bin_idx = int(base_rate / bin_size)
        return min(bin_idx, self.config.base_rate_bins - 1)

    def _get_resolution_date_bin(
        self, resolution_date: date | None, reference_date: date
    ) -> int:
        """Get the bin index for a resolution date."""
        if resolution_date is None:
            return 0

        days_until = (resolution_date - reference_date).days
        if self.config.max_resolution_days:
            bin_size = self.config.max_resolution_days / self.config.resolution_date_bins
            bin_idx = int(days_until / bin_size)
            return min(max(0, bin_idx), self.config.resolution_date_bins - 1)
        return 0

    def sample_stratified(
        self,
        questions: list[Question],
        reference_date: date | None = None,
    ) -> SamplingResult:
        """Sample questions with stratification across multiple dimensions.

        Balances:
        - Market vs data sources
        - Categories within each source type
        - Base rates (avoiding skew to 0%, 50%, 100%)
        - Resolution dates (avoiding year-end clustering)

        Args:
            questions: Pool of questions to sample from
            reference_date: Date to use for calculations (default: today)

        Returns:
            SamplingResult with sampled questions and diagnostics
        """
        if reference_date is None:
            reference_date = date.today()

        # Filter questions first
        pool = self.filter_questions(questions, reference_date)

        # Separate by source type
        market_questions = [q for q in pool if q.source_type == SourceType.MARKET]
        data_questions = [q for q in pool if q.source_type == SourceType.DATA]

        # Calculate target counts
        num_market = int(self.config.num_questions * self.config.market_fraction)
        num_data = self.config.num_questions - num_market

        # Sample each pool with stratification
        sampled_market = self._sample_with_balance(
            market_questions, num_market, reference_date
        )
        sampled_data = self._sample_with_balance(
            data_questions, num_data, reference_date
        )

        all_sampled = sampled_market + sampled_data

        # Build diagnostics
        warnings = []
        if len(sampled_market) < num_market:
            warnings.append(
                f"Insufficient market questions: got {len(sampled_market)}, wanted {num_market}"
            )
        if len(sampled_data) < num_data:
            warnings.append(
                f"Insufficient data questions: got {len(sampled_data)}, wanted {num_data}"
            )

        return SamplingResult(
            questions=all_sampled,
            source_counts=self._count_by_source(all_sampled),
            category_counts=self._count_by_category(all_sampled),
            resolution_date_distribution=self._count_by_resolution_bin(
                all_sampled, reference_date
            ),
            base_rate_distribution=self._count_by_base_rate_bin(all_sampled),
            warnings=warnings,
        )

    def _sample_with_balance(
        self,
        questions: list[Question],
        n: int,
        reference_date: date,
    ) -> list[Question]:
        """Sample n questions with balanced category and base rate distribution."""
        if len(questions) <= n:
            return questions

        # Group by category
        by_category: dict[str, list[Question]] = defaultdict(list)
        for q in questions:
            cat = q.category or "Unknown"
            by_category[cat].append(q)

        # Allocate evenly across categories
        num_categories = len(by_category)
        per_category = max(1, n // num_categories)

        sampled = []
        for cat, cat_questions in by_category.items():
            # Within each category, try to balance by base rate
            cat_sampled = self._sample_balanced_base_rate(
                cat_questions,
                min(per_category, len(cat_questions)),
            )
            sampled.extend(cat_sampled)

        # If we need more, sample from remaining
        remaining_needed = n - len(sampled)
        if remaining_needed > 0:
            sampled_ids = {q.id for q in sampled}
            remaining = [q for q in questions if q.id not in sampled_ids]
            if remaining:
                extra = random.sample(remaining, min(remaining_needed, len(remaining)))
                sampled.extend(extra)

        return sampled[:n]

    def _sample_balanced_base_rate(
        self,
        questions: list[Question],
        n: int,
    ) -> list[Question]:
        """Sample n questions with balanced base rate distribution."""
        if len(questions) <= n:
            return questions

        # Group by base rate bin
        by_bin: dict[int, list[Question]] = defaultdict(list)
        for q in questions:
            bin_idx = self._get_base_rate_bin(q.base_rate)
            by_bin[bin_idx].append(q)

        # Allocate evenly across bins
        num_bins = len(by_bin)
        per_bin = max(1, n // num_bins)

        sampled = []
        for bin_idx, bin_questions in by_bin.items():
            bin_sample = random.sample(
                bin_questions,
                min(per_bin, len(bin_questions)),
            )
            sampled.extend(bin_sample)

        # If we need more, fill from any bin
        remaining_needed = n - len(sampled)
        if remaining_needed > 0:
            sampled_ids = {q.id for q in sampled}
            remaining = [q for q in questions if q.id not in sampled_ids]
            if remaining:
                extra = random.sample(remaining, min(remaining_needed, len(remaining)))
                sampled.extend(extra)

        return sampled[:n]

    def _count_by_source(self, questions: list[Question]) -> dict[str, int]:
        counts: dict[str, int] = defaultdict(int)
        for q in questions:
            counts[q.source] += 1
        return dict(counts)

    def _count_by_category(self, questions: list[Question]) -> dict[str, int]:
        counts: dict[str, int] = defaultdict(int)
        for q in questions:
            counts[q.category or "Unknown"] += 1
        return dict(counts)

    def _count_by_resolution_bin(
        self, questions: list[Question], reference_date: date
    ) -> dict[str, int]:
        counts: dict[str, int] = defaultdict(int)
        for q in questions:
            bin_idx = self._get_resolution_date_bin(q.resolution_date, reference_date)
            counts[f"bin_{bin_idx}"] += 1
        return dict(counts)

    def _count_by_base_rate_bin(self, questions: list[Question]) -> dict[str, int]:
        counts: dict[str, int] = defaultdict(int)
        bin_size = 1.0 / self.config.base_rate_bins
        for q in questions:
            bin_idx = self._get_base_rate_bin(q.base_rate)
            low = bin_idx * bin_size
            high = (bin_idx + 1) * bin_size
            label = f"{low:.0%}-{high:.0%}"
            counts[label] += 1
        return dict(counts)
