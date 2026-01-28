"""Structured research for signal base rate estimation.

Follows the forecasting skill workflow:
1. Outside view: Historical base rates from reference classes
2. Market research: Search prediction markets more aggressively
3. News search: Recent developments that may move estimates
4. Synthesis: Combine estimates with weighted averaging

This provides better-calibrated estimates than one-shot LLM calls.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import litellm
from pydantic import BaseModel, Field

from .markets import MarketSignal, MarketSemanticSearcher, get_market_searcher

logger = logging.getLogger(__name__)


@dataclass
class ResearchResult:
    """Result of structured research for a signal."""

    base_rate: float
    confidence_interval: tuple[float, float]  # 80% CI (low, high)
    sources: list[str] = field(default_factory=list)  # e.g., ["market:polymarket", "outside-view", "news"]
    reasoning: str = ""
    market_match: MarketSignal | None = None
    estimates: list[tuple[str, float]] = field(default_factory=list)  # (source, estimate) pairs


class OutsideViewResponse(BaseModel):
    """LLM response for outside view base rate."""

    reference_class: str = Field(description="The reference class used for comparison")
    historical_frequency: float = Field(
        ge=0.0, le=1.0,
        description="Historical frequency/base rate for this reference class"
    )
    confidence: float = Field(
        ge=0.0, le=1.0,
        description="Confidence in this base rate estimate"
    )
    reasoning: str = Field(description="Explanation of the reference class and frequency")


class NewsImpactResponse(BaseModel):
    """LLM response for news impact assessment."""

    has_relevant_news: bool = Field(description="Whether there is relevant recent news")
    news_summary: str = Field(description="Summary of relevant news findings")
    probability_adjustment: float = Field(
        ge=-0.3, le=0.3,
        description="Suggested adjustment to base probability (-0.3 to +0.3)"
    )
    reasoning: str = Field(description="Why this news affects the probability")


class SynthesisResponse(BaseModel):
    """LLM response for synthesizing multiple estimates."""

    final_estimate: float = Field(
        ge=0.0, le=1.0,
        description="Final synthesized probability estimate"
    )
    confidence_low: float = Field(
        ge=0.0, le=1.0,
        description="Lower bound of 80% confidence interval"
    )
    confidence_high: float = Field(
        ge=0.0, le=1.0,
        description="Upper bound of 80% confidence interval"
    )
    reasoning: str = Field(description="Explanation of how estimates were combined")


async def gather_outside_view(
    signal_text: str,
    model: str = "anthropic/claude-3-haiku-20240307",
) -> tuple[float, float, str] | None:
    """Gather outside view base rate from reference classes.

    Args:
        signal_text: The signal question to research
        model: LLM model to use

    Returns:
        Tuple of (estimate, confidence, reasoning) or None if not found
    """
    try:
        response = await litellm.acompletion(
            model=model,
            messages=[
                {
                    "role": "user",
                    "content": f"""Estimate the base rate for this outcome using outside view / reference class forecasting:

Question: {signal_text}

Consider:
1. What is the appropriate reference class for this type of event?
2. What is the historical frequency of similar events?
3. How confident are you in this reference class?

Examples of reference classes:
- "Films winning Best Picture" → look at historical winners
- "Incumbent party winning election" → historical incumbent rates
- "Tech company acquisitions" → historical acquisition rates

Provide the reference class, historical frequency (0-1), and your confidence.""",
                }
            ],
            response_format=OutsideViewResponse,
            temperature=0.2,
        )

        result = OutsideViewResponse.model_validate_json(
            response.choices[0].message.content
        )

        if result.confidence >= 0.3:
            return result.historical_frequency, result.confidence, result.reasoning

        return None

    except Exception as e:
        logger.warning(f"Outside view gathering failed: {e}")
        return None


async def search_related_markets(
    signal_text: str,
    market_searcher: MarketSemanticSearcher,
    platforms: list[str] | None = None,
    top_k: int = 5,
) -> list[MarketSignal]:
    """Search for related markets that provide context.

    Args:
        signal_text: The signal question to search for
        market_searcher: Market semantic searcher instance
        platforms: Platforms to search
        top_k: Number of results to return

    Returns:
        List of related markets
    """
    platforms = platforms or ["polymarket", "metaculus"]

    try:
        if not market_searcher.cache_exists:
            return []

        results = market_searcher.search(
            signal_text,
            top_k=top_k,
            platforms=platforms,
            search_type="related",
        )

        return [r for r in results if r.current_probability is not None]

    except Exception as e:
        logger.warning(f"Related market search failed: {e}")
        return []


async def estimate_from_news(
    signal_text: str,
    current_estimate: float | None = None,
    model: str = "anthropic/claude-3-haiku-20240307",
) -> tuple[float, str] | None:
    """Estimate probability adjustment from recent news.

    Uses the LLM's knowledge of recent events (within training cutoff)
    to assess if news affects the probability.

    Args:
        signal_text: The signal question
        current_estimate: Current base estimate (if any)
        model: LLM model to use

    Returns:
        Tuple of (adjustment, reasoning) or None if no relevant news
    """
    try:
        current_info = f"Current estimate: {current_estimate:.0%}" if current_estimate else "No current estimate"

        response = await litellm.acompletion(
            model=model,
            messages=[
                {
                    "role": "user",
                    "content": f"""Assess whether recent news or developments affect this forecast:

Question: {signal_text}
{current_info}

Consider:
1. Are there recent news events relevant to this question?
2. Do these events suggest the probability should be adjusted up or down?
3. How significant is the adjustment?

If there is no relevant recent news, set has_relevant_news to false.
Otherwise, suggest a probability adjustment between -0.3 and +0.3.""",
                }
            ],
            response_format=NewsImpactResponse,
            temperature=0.3,
        )

        result = NewsImpactResponse.model_validate_json(
            response.choices[0].message.content
        )

        if result.has_relevant_news and abs(result.probability_adjustment) > 0.01:
            return result.probability_adjustment, result.reasoning

        return None

    except Exception as e:
        logger.warning(f"News impact assessment failed: {e}")
        return None


def combine_estimates(
    estimates: list[tuple[str, float]],
    source_weights: dict[str, float] | None = None,
) -> float:
    """Combine estimates using weighted averaging.

    Args:
        estimates: List of (source, estimate) pairs
        source_weights: Optional weights by source type

    Returns:
        Weighted average estimate
    """
    if not estimates:
        return 0.5

    # Default weights: markets most trusted, then outside view, then news/llm
    default_weights = {
        "market": 1.0,
        "market:polymarket": 1.0,
        "market:metaculus": 0.9,
        "outside-view": 0.7,
        "news": 0.5,
        "llm-estimate": 0.4,
    }

    weights = source_weights or default_weights

    total_weight = 0.0
    weighted_sum = 0.0

    for source, estimate in estimates:
        # Find best matching weight key
        weight = 0.5  # default
        for key, w in weights.items():
            if source.startswith(key) or key in source:
                weight = w
                break

        weighted_sum += estimate * weight
        total_weight += weight

    if total_weight == 0:
        return sum(e for _, e in estimates) / len(estimates)

    return weighted_sum / total_weight


def compute_confidence_interval(
    estimates: list[tuple[str, float]],
    base_estimate: float,
) -> tuple[float, float]:
    """Compute 80% confidence interval from estimate spread.

    Args:
        estimates: List of (source, estimate) pairs
        base_estimate: The central estimate

    Returns:
        (low, high) tuple for 80% CI
    """
    if len(estimates) <= 1:
        # Single estimate - use uncertainty based on source
        if estimates and "market" in estimates[0][0]:
            # Market estimates have narrower CIs
            spread = 0.1
        else:
            # LLM/other estimates have wider CIs
            spread = 0.2

        low = max(0.0, base_estimate - spread)
        high = min(1.0, base_estimate + spread)
        return (low, high)

    # Multiple estimates - use spread
    values = [e for _, e in estimates]
    min_val = min(values)
    max_val = max(values)
    spread = max_val - min_val

    # Widen CI slightly beyond observed range
    ci_spread = max(spread * 1.2, 0.1)  # Minimum 10% spread
    low = max(0.0, base_estimate - ci_spread / 2)
    high = min(1.0, base_estimate + ci_spread / 2)

    return (low, high)


async def synthesize_estimates(
    signal_text: str,
    estimates: list[tuple[str, float]],
    model: str = "anthropic/claude-3-haiku-20240307",
) -> tuple[float, tuple[float, float], str]:
    """Use LLM to synthesize multiple estimates into final forecast.

    Args:
        signal_text: The signal question
        estimates: List of (source, estimate) pairs
        model: LLM model to use

    Returns:
        Tuple of (final_estimate, (ci_low, ci_high), reasoning)
    """
    if not estimates:
        return 0.5, (0.3, 0.7), "No estimates available, using maximum uncertainty"

    # Format estimates for prompt
    estimates_text = "\n".join(
        f"- {source}: {estimate:.0%}"
        for source, estimate in estimates
    )

    try:
        response = await litellm.acompletion(
            model=model,
            messages=[
                {
                    "role": "user",
                    "content": f"""Synthesize these probability estimates into a final forecast:

Question: {signal_text}

Estimates from different sources:
{estimates_text}

Guidelines for weighting:
- Prediction market prices should be weighted highest (most calibrated)
- Outside view / base rates should be weighted medium
- News adjustments and LLM estimates should be weighted lower

Provide:
1. A final synthesized probability (0-1)
2. An 80% confidence interval (low, high)
3. Brief reasoning for your synthesis""",
                }
            ],
            response_format=SynthesisResponse,
            temperature=0.2,
        )

        result = SynthesisResponse.model_validate_json(
            response.choices[0].message.content
        )

        return (
            result.final_estimate,
            (result.confidence_low, result.confidence_high),
            result.reasoning,
        )

    except Exception as e:
        logger.warning(f"Synthesis failed: {e}, using weighted average")
        combined = combine_estimates(estimates)
        ci = compute_confidence_interval(estimates, combined)
        return combined, ci, f"Weighted average (synthesis failed: {e})"


async def research_signal(
    signal_text: str,
    target_context: str | None = None,
    market_searcher: MarketSemanticSearcher | None = None,
    db_path: str = "forecastbench.db",
    use_outside_view: bool = True,
    use_news: bool = True,
    use_llm_synthesis: bool = True,
) -> ResearchResult:
    """Run forecasting skill research workflow for a signal.

    Gathers estimates from multiple sources and synthesizes them.

    Args:
        signal_text: The signal question to research
        target_context: Optional context about the parent target
        market_searcher: Optional market searcher (will create if None)
        db_path: Path to market database
        use_outside_view: Whether to gather outside view estimates
        use_news: Whether to assess news impact
        use_llm_synthesis: Whether to use LLM for synthesis (vs simple weighting)

    Returns:
        ResearchResult with base_rate, confidence_interval, and sources
    """
    from .reconcile import search_markets_by_title_fallback, find_best_match_with_rerank

    estimates: list[tuple[str, float]] = []
    sources: list[str] = []
    market_match: MarketSignal | None = None

    # Get or create market searcher
    if market_searcher is None:
        market_searcher = get_market_searcher(db_path)

    # --- Phase 1: Prediction Market Research ---
    # This is the most important source - markets are best calibrated

    # 1a. Semantic search
    if market_searcher.cache_exists:
        candidates = market_searcher.search(
            signal_text, top_k=10, search_type="direct"
        )
        candidates = [c for c in candidates if c.current_probability is not None]

        if candidates:
            best_match, confidence = await find_best_match_with_rerank(
                signal_text, candidates, min_similarity=0.3, min_confidence=0.7
            )
            # Require high confidence (0.7+) for market matches to avoid spurious matches
            if best_match and best_match.current_probability is not None and confidence >= 0.7:
                market_match = best_match
                estimates.append((f"market:{best_match.platform}", best_match.current_probability))
                sources.append(f"market:{best_match.platform}")
                logger.info(f"Found market match: {best_match.title} @ {best_match.current_probability:.0%} (confidence: {confidence:.0%})")
            elif best_match:
                logger.info(f"Rejected low-confidence market match: {best_match.title} (confidence: {confidence:.0%} < 70%)")

    # 1b. Fallback: title search if semantic search failed
    if market_match is None:
        logger.debug("Semantic search failed, trying title fallback")
        fallback_markets = await search_markets_by_title_fallback(
            signal_text, db_path=db_path
        )
        if fallback_markets:
            # Use LLM reranking on fallback results - require HIGH confidence
            best_match, confidence = await find_best_match_with_rerank(
                signal_text, fallback_markets, min_similarity=0.2, min_confidence=0.7
            )
            # Require high confidence (0.7+) for fallback matches
            if best_match and best_match.current_probability is not None and confidence >= 0.7:
                market_match = best_match
                estimates.append((f"market:{best_match.platform}", best_match.current_probability))
                sources.append(f"market:{best_match.platform}")
                logger.info(f"Found market via title search: {best_match.title} @ {best_match.current_probability:.0%} (confidence: {confidence:.0%})")
            elif best_match:
                logger.info(f"Rejected low-confidence title fallback: {best_match.title} (confidence: {confidence:.0%} < 70%)")

    # --- Phase 2: Outside View (Base Rate) ---
    if use_outside_view:
        outside_view = await gather_outside_view(signal_text)
        if outside_view:
            estimate, confidence, reasoning = outside_view
            estimates.append(("outside-view", estimate))
            sources.append("outside-view")
            logger.info(f"Outside view: {estimate:.0%} (confidence: {confidence:.0%})")

    # --- Phase 3: News Impact ---
    if use_news:
        current_estimate = estimates[0][1] if estimates else None
        news_result = await estimate_from_news(signal_text, current_estimate)
        if news_result:
            adjustment, reasoning = news_result
            # Apply adjustment to current estimate
            if current_estimate is not None:
                news_estimate = max(0.0, min(1.0, current_estimate + adjustment))
                estimates.append(("news", news_estimate))
                sources.append("news")
                logger.info(f"News adjustment: {adjustment:+.0%} → {news_estimate:.0%}")

    # --- Phase 4: Synthesis ---
    if not estimates:
        # Fallback: pure LLM estimate if no other sources
        from .reconcile import estimate_base_rate
        llm_estimate, reasoning = await estimate_base_rate(signal_text, target_context)
        estimates.append(("llm-estimate", llm_estimate))
        sources.append("llm-estimate")
        logger.info(f"LLM fallback estimate: {llm_estimate:.0%}")

    # Synthesize estimates
    if use_llm_synthesis and len(estimates) > 1:
        base_rate, confidence_interval, reasoning = await synthesize_estimates(
            signal_text, estimates
        )
    else:
        # Simple weighted combination
        base_rate = combine_estimates(estimates)
        confidence_interval = compute_confidence_interval(estimates, base_rate)
        reasoning = f"Weighted average from {len(estimates)} sources"

    return ResearchResult(
        base_rate=base_rate,
        confidence_interval=confidence_interval,
        sources=sources,
        reasoning=reasoning,
        market_match=market_match,
        estimates=estimates,
    )


async def research_signals_batch(
    signals: list[str],
    target_context: str | None = None,
    db_path: str = "forecastbench.db",
    max_concurrent: int = 3,
) -> list[ResearchResult]:
    """Research multiple signals in batch with rate limiting.

    Args:
        signals: List of signal question texts
        target_context: Optional context about the parent target
        db_path: Path to market database
        max_concurrent: Maximum concurrent research tasks

    Returns:
        List of ResearchResult in same order as input signals
    """
    import asyncio

    market_searcher = get_market_searcher(db_path)

    # Semaphore for rate limiting
    semaphore = asyncio.Semaphore(max_concurrent)

    async def research_with_limit(signal: str) -> ResearchResult:
        async with semaphore:
            return await research_signal(
                signal,
                target_context=target_context,
                market_searcher=market_searcher,
                db_path=db_path,
            )

    tasks = [research_with_limit(s) for s in signals]
    return await asyncio.gather(*tasks)
