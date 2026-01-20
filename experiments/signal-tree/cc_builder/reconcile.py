"""Phase 3: Reconciliation - Map markets onto logical structure.

Takes the logical structure from Phase 1 and the discovered markets from Phase 2,
and builds a signal tree by matching markets to constraints and pathways.
"""

from __future__ import annotations

import logging
import uuid
from datetime import date

import litellm
from pydantic import BaseModel, Field

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from shared.tree import SignalNode, SignalTree

from .structure import (
    LogicalStructure,
    NecessityConstraint,
    ExclusivityConstraint,
    CausalPathway,
)
from .markets import MarketSignal

logger = logging.getLogger(__name__)


class RhoEstimate(BaseModel):
    """LLM response for rho estimation."""

    rho: float = Field(
        description="Correlation coefficient between -1 and 1",
        ge=-1,
        le=1,
    )
    reasoning: str = Field(description="Explanation of the correlation")


class BaseRateEstimate(BaseModel):
    """LLM response for base rate estimation."""

    probability: float = Field(
        description="Estimated probability between 0 and 1",
        ge=0,
        le=1,
    )
    reasoning: str = Field(description="Explanation of the estimate")


def find_best_match(
    description: str,
    markets: list[MarketSignal],
    min_similarity: float = 0.3,
) -> MarketSignal | None:
    """Find the best market match for a description.

    Args:
        description: The constraint or pathway description
        markets: Available markets to match against
        min_similarity: Minimum similarity threshold

    Returns:
        Best matching market or None if no good match
    """
    # Simple approach: find market with highest title similarity
    best_match: MarketSignal | None = None
    best_score = min_similarity

    description_lower = description.lower()
    desc_words = set(description_lower.split())

    for market in markets:
        title_lower = market.title.lower()
        title_words = set(title_lower.split())

        # Word overlap score
        if desc_words and title_words:
            intersection = len(desc_words & title_words)
            union = len(desc_words | title_words)
            word_score = intersection / union
        else:
            word_score = 0

        # Combine with embedding similarity
        combined_score = 0.5 * market.similarity + 0.5 * word_score

        if combined_score > best_score:
            best_score = combined_score
            best_match = market

    return best_match


async def estimate_base_rate(
    description: str,
    context: str | None = None,
    model: str = "anthropic/claude-3-haiku-20240307",
) -> tuple[float, str]:
    """Estimate base rate for a signal without market data.

    Args:
        description: The signal description
        context: Optional context about the signal
        model: LLM model to use

    Returns:
        Tuple of (probability, reasoning)
    """
    context_section = f"\nContext: {context}" if context else ""

    response = await litellm.acompletion(
        model=model,
        messages=[
            {
                "role": "user",
                "content": f"""Estimate the probability of this outcome:

{description}{context_section}

Provide a probability between 0 and 1, and explain your reasoning.
Consider base rates, current evidence, and uncertainty.""",
            }
        ],
        response_format=BaseRateEstimate,
    )

    result = BaseRateEstimate.model_validate_json(
        response.choices[0].message.content
    )
    return result.probability, result.reasoning


async def estimate_rho(
    parent: str,
    signal: str,
    model: str = "anthropic/claude-3-haiku-20240307",
) -> tuple[float, str]:
    """Estimate correlation coefficient between parent and signal.

    Args:
        parent: The parent question
        signal: The signal question
        model: LLM model to use

    Returns:
        Tuple of (rho, reasoning)
    """
    response = await litellm.acompletion(
        model=model,
        messages=[
            {
                "role": "user",
                "content": f"""Estimate the correlation between these two questions:

Parent: {parent}
Signal: {signal}

Consider:
- Same entity? → likely positive correlation
- Competing entities? → likely negative correlation
- Upstream/downstream relationship? → direction depends on mechanism

Provide a correlation coefficient between -1 and 1:
- +1.0: Perfect positive (signal YES → parent YES)
- +0.5: Moderate positive
- 0: Independent
- -0.5: Moderate negative (signal YES → parent NO)
- -1.0: Perfect negative""",
            }
        ],
        response_format=RhoEstimate,
    )

    result = RhoEstimate.model_validate_json(
        response.choices[0].message.content
    )
    return result.rho, result.reasoning


def create_signal_id() -> str:
    """Generate a unique signal ID."""
    return f"sig_{uuid.uuid4().hex[:8]}"


async def reconcile(
    structure: LogicalStructure,
    markets: list[MarketSignal],
    target_id: str = "target",
    parent_prior: float = 0.5,
) -> SignalTree:
    """Map markets onto logical structure to build signal tree.

    Args:
        structure: Logical structure from Phase 1
        markets: Discovered markets from Phase 2
        target_id: ID for the target node
        parent_prior: Prior probability for the target

    Returns:
        SignalTree with signals mapped from structure and markets
    """
    # Create target node
    target = SignalNode(
        id=target_id,
        text=structure.target,
        depth=0,
        is_leaf=False,
        base_rate=parent_prior,
    )

    signals: list[SignalNode] = []

    # 1. Handle necessity constraints
    for constraint in structure.necessity_constraints:
        market = find_best_match(constraint.prerequisite, markets)

        if market and market.current_probability is not None:
            base_rate = market.current_probability
            market_price = market.current_probability
            market_url = market.url
            market_platform = market.platform
            probability_source = "market"
            text = market.title
        else:
            # No market match - estimate base rate
            base_rate, _ = await estimate_base_rate(constraint.prerequisite)
            market_price = None
            market_url = None
            market_platform = None
            probability_source = "llm"
            text = constraint.prerequisite

        signal = SignalNode(
            id=create_signal_id(),
            text=text,
            parent_id=target_id,
            relationship_type="necessity",
            base_rate=base_rate,
            probability_source=probability_source,
            market_price=market_price,
            market_url=market_url,
            market_platform=market_platform,
            rho_reasoning=constraint.reasoning,
            depth=1,
            is_leaf=True,
        )
        signals.append(signal)

    # 2. Handle exclusivity constraints
    for constraint in structure.exclusivity_constraints:
        search_text = f"{constraint.competitor} {constraint.prize}"
        market = find_best_match(search_text, markets)

        if market and market.current_probability is not None:
            base_rate = market.current_probability
            market_price = market.current_probability
            market_url = market.url
            market_platform = market.platform
            probability_source = "market"
            text = market.title
        else:
            # No market match - estimate base rate
            base_rate, _ = await estimate_base_rate(
                f"Will {constraint.competitor} win {constraint.prize}?"
            )
            market_price = None
            market_url = None
            market_platform = None
            probability_source = "llm"
            text = f"Will {constraint.competitor} win {constraint.prize}?"

        signal = SignalNode(
            id=create_signal_id(),
            text=text,
            parent_id=target_id,
            relationship_type="exclusivity",
            base_rate=base_rate,
            probability_source=probability_source,
            market_price=market_price,
            market_url=market_url,
            market_platform=market_platform,
            p_target_given_yes=0.01,  # Competitor wins → target loses
            p_target_given_no=min(0.99, parent_prior * 1.1),  # Slight boost
            rho_reasoning=constraint.reasoning,
            depth=1,
            is_leaf=True,
        )
        signals.append(signal)

    # 3. Handle causal pathways
    for pathway in structure.causal_pathways:
        market = find_best_match(pathway.upstream_event, markets)

        if market and market.current_probability is not None:
            base_rate = market.current_probability
            market_price = market.current_probability
            market_url = market.url
            market_platform = market.platform
            probability_source = "market"
            text = market.title
        else:
            # No market match - estimate base rate
            base_rate, _ = await estimate_base_rate(pathway.upstream_event)
            market_price = None
            market_url = None
            market_platform = None
            probability_source = "llm"
            text = pathway.upstream_event

        # Estimate rho based on effect direction
        effect_lower = pathway.effect_on_target.lower()
        if "positive" in effect_lower or "helps" in effect_lower or "increases" in effect_lower:
            # Positive effect - estimate moderate positive rho
            rho = 0.4
        elif "negative" in effect_lower or "hurts" in effect_lower or "decreases" in effect_lower:
            # Negative effect - estimate moderate negative rho
            rho = -0.4
        else:
            # Unknown - use LLM to estimate
            rho, _ = await estimate_rho(structure.target, pathway.upstream_event)

        signal = SignalNode(
            id=create_signal_id(),
            text=text,
            parent_id=target_id,
            relationship_type="correlation",
            base_rate=base_rate,
            probability_source=probability_source,
            market_price=market_price,
            market_url=market_url,
            market_platform=market_platform,
            rho=rho,
            rho_reasoning=f"{pathway.mechanism}: {pathway.effect_on_target}",
            depth=1,
            is_leaf=True,
        )
        signals.append(signal)

    # Build the tree
    target.children = signals
    max_depth = max((s.depth for s in signals), default=0)
    leaf_count = sum(1 for s in signals if s.is_leaf)

    tree = SignalTree(
        target=target,
        signals=signals,
        max_depth=max_depth,
        leaf_count=leaf_count,
    )

    return tree


def identify_uncertain_signals(
    tree: SignalTree,
    min_uncertainty: float = 0.2,
    max_uncertainty: float = 0.8,
    min_resolution_days: int = 14,
    today: date | None = None,
) -> list[SignalNode]:
    """Identify signals that would benefit from further decomposition.

    Criteria:
    - Base rate between min_uncertainty and max_uncertainty (not near-certain)
    - Resolution date beyond min_resolution_days (has time for sub-signals)
    - Is currently a leaf node

    Args:
        tree: The signal tree
        min_uncertainty: Minimum base rate to consider uncertain
        max_uncertainty: Maximum base rate to consider uncertain
        min_resolution_days: Minimum days until resolution to decompose
        today: Current date (defaults to today)

    Returns:
        List of SignalNode that should be decomposed
    """
    today = today or date.today()

    uncertain = []
    for signal in tree.signals:
        if not signal.is_leaf:
            continue

        # Check uncertainty
        if signal.base_rate is None:
            continue
        if signal.base_rate < min_uncertainty or signal.base_rate > max_uncertainty:
            continue

        # Check resolution date
        if signal.resolution_date is not None:
            days_until = (signal.resolution_date - today).days
            if days_until < min_resolution_days:
                continue

        uncertain.append(signal)

    return uncertain
