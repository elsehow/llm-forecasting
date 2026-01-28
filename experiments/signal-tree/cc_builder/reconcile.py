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


class ComparativeQuestion(BaseModel):
    """Detected comparative question structure."""

    is_comparative: bool = Field(description="Whether this is a comparative question (A vs B)")
    entity_a: str | None = Field(default=None, description="The entity we want to be higher/better")
    entity_b: str | None = Field(default=None, description="The entity we want to be lower/worse")
    comparison_type: str | None = Field(default=None, description="Type: ranking, score, amount, etc.")


class SignalEntityEffect(BaseModel):
    """Which entity a signal primarily affects and how."""

    primary_entity: str = Field(description="Which entity this signal primarily affects")
    effect_direction: str = Field(description="positive or negative effect on that entity")
    reasoning: str = Field(description="Brief explanation")


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


async def find_best_match_with_rerank(
    description: str,
    markets: list[MarketSignal],
    min_similarity: float = 0.3,
    min_confidence: float = 0.6,
) -> tuple[MarketSignal | None, float]:
    """Find best match using semantic search + LLM reranking.

    Args:
        description: The constraint or pathway description
        markets: Available markets to match against
        min_similarity: Minimum similarity threshold for candidates
        min_confidence: Minimum LLM confidence to accept match

    Returns:
        Tuple of (best_market, confidence) or (None, 0.0) if no match
    """
    from llm_forecasting.market_data.matcher import _llm_rerank
    from llm_forecasting.market_data.models import Market, MarketStatus
    from datetime import datetime

    # 1. Get top candidates from semantic + word overlap scoring
    candidates = []
    description_lower = description.lower()
    desc_words = set(description_lower.split())

    for market in markets:
        if market.current_probability is None:
            continue

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

        if combined_score >= min_similarity:
            candidates.append((market, combined_score))

    if not candidates:
        return None, 0.0

    # Sort by combined score and take top 10
    candidates.sort(key=lambda x: x[1], reverse=True)
    top_candidates = [c[0] for c in candidates[:10]]

    if len(top_candidates) == 1:
        # Single candidate - return it with moderate confidence
        return top_candidates[0], 0.6

    # 2. LLM rerank: ask which market best matches the description
    candidate_markets = [
        Market(
            id=m.market_id,
            platform=m.platform,
            title=m.title,
            current_probability=m.current_probability,
            url=m.url,
            status=MarketStatus.OPEN,
            created_at=datetime.now(),
            fetched_at=datetime.now(),
        )
        for m in top_candidates
    ]

    best_idx, confidence, reasoning = await _llm_rerank(description, candidate_markets)

    if best_idx is None:
        logger.debug(f"LLM rerank found no match: {reasoning}")
        return None, 0.0

    if best_idx >= len(top_candidates):
        logger.warning(f"LLM returned invalid index {best_idx}, using 0")
        best_idx = 0

    if confidence < min_confidence:
        logger.debug(
            f"Match below confidence threshold: {confidence:.0%} < {min_confidence:.0%}"
        )
        return None, confidence

    return top_candidates[best_idx], confidence


async def search_markets_by_title_fallback(
    query: str,
    db_path: str,
    min_liquidity: float = 1000,
    limit: int = 20,
) -> list[MarketSignal]:
    """Direct title search as fallback when semantic search fails.

    Args:
        query: Search query text
        db_path: Path to market database
        min_liquidity: Minimum liquidity filter
        limit: Maximum results to return

    Returns:
        List of MarketSignal from title search
    """
    from llm_forecasting.market_data import MarketDataStorage
    from llm_forecasting.market_data.models import MarketStatus

    storage = MarketDataStorage(db_path)
    try:
        # Extract keywords from query
        keywords = query.split()[:5]
        markets = await storage.search_by_title(
            keywords=keywords,
            min_liquidity=min_liquidity,
            status=MarketStatus.OPEN,
            limit=limit,
        )

        return [
            MarketSignal(
                market_id=m.id,
                platform=m.platform,
                title=m.title,
                similarity=0.5,  # Fallback score
                current_probability=m.current_probability,
                url=m.url,
                search_type="title_fallback",
            )
            for m in markets
            if m.current_probability is not None
        ]
    finally:
        await storage.close()


async def detect_comparative_question(
    target: str,
    model: str = "anthropic/claude-3-haiku-20240307",
) -> ComparativeQuestion:
    """Detect if a question is comparative (A vs B) and extract entities.

    Args:
        target: The target question
        model: LLM model to use

    Returns:
        ComparativeQuestion with entity information
    """
    response = await litellm.acompletion(
        model=model,
        messages=[
            {
                "role": "user",
                "content": f"""Analyze if this is a comparative question between two entities:

Question: {target}

A comparative question asks whether entity A will be higher/better/more than entity B.
Examples:
- "Will Canada rank higher than the US?" → comparative (A=Canada, B=US)
- "Will Team A beat Team B?" → comparative (A=Team A, B=Team B)
- "Will inflation be above 3%?" → NOT comparative (threshold, not entity comparison)
- "Will X win Best Picture?" → NOT comparative (competition with many, not A vs B)

If comparative, identify:
- entity_a: The entity we want to be higher/better (the subject of "higher than")
- entity_b: The entity we want entity_a to beat
- comparison_type: What's being compared (ranking, score, amount, etc.)""",
            }
        ],
        response_format=ComparativeQuestion,
        temperature=0.1,
    )

    return ComparativeQuestion.model_validate_json(
        response.choices[0].message.content
    )


async def determine_signal_entity_effect(
    signal_text: str,
    entity_a: str,
    entity_b: str,
    effect_description: str,
    model: str = "anthropic/claude-3-haiku-20240307",
) -> SignalEntityEffect:
    """Determine which entity a signal affects and how.

    Args:
        signal_text: The signal description
        entity_a: The "positive" entity (we want this one higher)
        entity_b: The "negative" entity (we want this one lower)
        effect_description: The causal pathway's effect description
        model: LLM model to use

    Returns:
        SignalEntityEffect with entity and direction
    """
    response = await litellm.acompletion(
        model=model,
        messages=[
            {
                "role": "user",
                "content": f"""Determine which entity this signal primarily affects:

Signal: {signal_text}
Effect description: {effect_description}

Entity A (we want higher): {entity_a}
Entity B (we want lower): {entity_b}

Which entity does this signal primarily affect, and is the effect positive or negative for that entity?

Examples:
- "US GDP growth" affects "United States" positively
- "Canada healthcare improvements" affects "Canada" positively
- "US political instability" affects "United States" negatively
- "Climate change hurts Canada" affects "Canada" negatively""",
            }
        ],
        response_format=SignalEntityEffect,
        temperature=0.1,
    )

    return SignalEntityEffect.model_validate_json(
        response.choices[0].message.content
    )


def compute_rho_for_comparative(
    primary_entity: str,
    effect_direction: str,
    entity_a: str,
    entity_b: str,
    base_rho: float = 0.4,
) -> float:
    """Compute rho sign for a comparative question.

    For "Will A be higher than B?":
    - Signal helps A → positive rho (good for target)
    - Signal hurts A → negative rho (bad for target)
    - Signal helps B → negative rho (bad for target, B gets better)
    - Signal hurts B → positive rho (good for target, B gets worse)

    Args:
        primary_entity: Which entity the signal affects
        effect_direction: "positive" or "negative" effect on that entity
        entity_a: The entity we want higher
        entity_b: The entity we want lower
        base_rho: Base magnitude of correlation

    Returns:
        Signed rho value
    """
    # Normalize for comparison
    primary_lower = primary_entity.lower()
    entity_a_lower = entity_a.lower()
    entity_b_lower = entity_b.lower()
    effect_positive = "positive" in effect_direction.lower()

    # Determine if signal affects entity A or entity B
    affects_a = any(word in primary_lower for word in entity_a_lower.split())
    affects_b = any(word in primary_lower for word in entity_b_lower.split())

    if affects_a and not affects_b:
        # Signal affects entity A
        # Positive effect on A = good for target = positive rho
        # Negative effect on A = bad for target = negative rho
        return base_rho if effect_positive else -base_rho
    elif affects_b and not affects_a:
        # Signal affects entity B
        # Positive effect on B = bad for target (B gets better) = negative rho
        # Negative effect on B = good for target (B gets worse) = positive rho
        return -base_rho if effect_positive else base_rho
    else:
        # Ambiguous - fall back to effect direction
        return base_rho if effect_positive else -base_rho


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
    min_market_similarity: float = 0.3,
    use_llm_rerank: bool = True,
    use_structured_research: bool = True,
    db_path: str = "forecastbench.db",
) -> SignalTree:
    """Map markets onto logical structure to build signal tree.

    Args:
        structure: Logical structure from Phase 1
        markets: Discovered markets from Phase 2
        target_id: ID for the target node
        parent_prior: Prior probability for the target
        min_market_similarity: Minimum similarity to accept a market match (default: 0.3)
        use_llm_rerank: Whether to use LLM reranking for better matching (default: True)
        use_structured_research: Whether to use full research workflow when no market match (default: True)
        db_path: Path to market database for title search fallback

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
    used_market_ids: set[str] = set()  # Track used markets to avoid duplicates

    # Detect if this is a comparative question (A vs B)
    comparative = await detect_comparative_question(structure.target)
    if comparative.is_comparative:
        logger.info(
            f"Detected comparative question: {comparative.entity_a} vs {comparative.entity_b}"
        )

    async def try_match_market(description: str) -> tuple[MarketSignal | None, float]:
        """Find best unused market match with optional LLM reranking.

        Returns:
            Tuple of (market, confidence) or (None, 0.0)
        """
        available = [m for m in markets if f"{m.platform}|{m.market_id}" not in used_market_ids]

        if use_llm_rerank:
            market, confidence = await find_best_match_with_rerank(
                description, available, min_similarity=min_market_similarity
            )
        else:
            market = find_best_match(description, available, min_similarity=min_market_similarity)
            confidence = market.similarity if market else 0.0

        # Fallback: try direct title search if semantic search failed
        if market is None:
            logger.debug(f"Semantic search failed for '{description[:50]}...', trying title search")
            fallback_markets = await search_markets_by_title_fallback(
                description, db_path=db_path
            )
            # Filter out already-used markets
            fallback_markets = [
                m for m in fallback_markets
                if f"{m.platform}|{m.market_id}" not in used_market_ids
            ]
            if fallback_markets:
                if use_llm_rerank and len(fallback_markets) > 1:
                    market, confidence = await find_best_match_with_rerank(
                        description, fallback_markets, min_similarity=0.2
                    )
                elif fallback_markets:
                    market = fallback_markets[0]
                    confidence = 0.5

        if market:
            used_market_ids.add(f"{market.platform}|{market.market_id}")

        return market, confidence

    # 1. Handle necessity constraints
    # NOTE: Necessity constraints require HIGH confidence matches because
    # bad matches are catastrophic (they multiply the probability)
    for constraint in structure.necessity_constraints:
        market, match_confidence = await try_match_market(constraint.prerequisite)

        # Require higher confidence for necessity constraints (0.8 vs 0.6)
        # since bad matches here have multiplicative effects on the final probability
        if market and match_confidence is not None and match_confidence < 0.8:
            logger.info(
                f"Rejecting low-confidence necessity match: {market.title[:50]} "
                f"(confidence: {match_confidence:.0%} < 80%)"
            )
            market = None
            match_confidence = None

        # Initialize fields for SignalNode
        confidence_interval = None
        base_rate_sources = []
        research_reasoning = None

        if market and market.current_probability is not None:
            base_rate = market.current_probability
            market_price = market.current_probability
            market_url = market.url
            market_platform = market.platform
            probability_source = "market"
            text = market.title
            base_rate_sources = [f"market:{market.platform}"]
        elif use_structured_research:
            # Use full research workflow when no direct market match
            from .research import research_signal
            from .markets import get_market_searcher

            research_result = await research_signal(
                constraint.prerequisite,
                target_context=structure.target,
                market_searcher=get_market_searcher(db_path),
                db_path=db_path,
            )

            base_rate = research_result.base_rate
            confidence_interval = research_result.confidence_interval
            base_rate_sources = research_result.sources
            research_reasoning = research_result.reasoning

            if research_result.market_match:
                # Research found a market match
                market = research_result.market_match
                market_price = market.current_probability
                market_url = market.url
                market_platform = market.platform
                probability_source = "market"
                text = market.title
            else:
                market_price = None
                market_url = None
                market_platform = None
                probability_source = "research"
                text = constraint.prerequisite
            match_confidence = None
        else:
            # Quick LLM estimate (fallback)
            base_rate, _ = await estimate_base_rate(constraint.prerequisite)
            market_price = None
            market_url = None
            market_platform = None
            probability_source = "llm"
            text = constraint.prerequisite
            match_confidence = None
            base_rate_sources = ["llm-estimate"]

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
            market_match_confidence=match_confidence,
            confidence_interval=confidence_interval,
            base_rate_sources=base_rate_sources,
            research_reasoning=research_reasoning,
            rho_reasoning=constraint.reasoning,
            depth=1,
            is_leaf=True,
        )
        signals.append(signal)

    # 2. Handle exclusivity constraints
    # Filter out exclusivity constraints that are semantically too similar to the target
    # (i.e., they're just the inverse of the question, not a different competitor)
    def is_inverse_of_target(constraint_text: str, target_text: str) -> bool:
        """Check if a constraint is essentially the inverse of the target question."""
        # Simple heuristic: if most key words overlap, it's probably just the inverse
        constraint_words = set(constraint_text.lower().split())
        target_words = set(target_text.lower().split())
        # Remove common stop words
        stop_words = {"will", "the", "a", "an", "in", "at", "for", "to", "be", "is", "?", "win"}
        constraint_words -= stop_words
        target_words -= stop_words
        if not constraint_words or not target_words:
            return False
        overlap = len(constraint_words & target_words)
        smaller = min(len(constraint_words), len(target_words))
        return overlap / smaller > 0.7  # >70% word overlap = likely inverse

    filtered_exclusivity = [
        c for c in structure.exclusivity_constraints
        if not is_inverse_of_target(
            f"{c.competitor} {c.prize}",
            structure.target
        )
    ]

    if len(filtered_exclusivity) < len(structure.exclusivity_constraints):
        logger.info(
            f"Filtered {len(structure.exclusivity_constraints) - len(filtered_exclusivity)} "
            "exclusivity constraints that were inverses of target"
        )

    for constraint in filtered_exclusivity:
        search_text = f"Will {constraint.competitor} win {constraint.prize}?"
        market, match_confidence = await try_match_market(search_text)

        # Initialize fields for SignalNode
        confidence_interval = None
        base_rate_sources = []
        research_reasoning = None

        if market and market.current_probability is not None:
            base_rate = market.current_probability
            market_price = market.current_probability
            market_url = market.url
            market_platform = market.platform
            probability_source = "market"
            text = market.title
            base_rate_sources = [f"market:{market.platform}"]
        elif use_structured_research:
            # Use full research workflow when no direct market match
            from .research import research_signal
            from .markets import get_market_searcher

            research_result = await research_signal(
                search_text,
                target_context=structure.target,
                market_searcher=get_market_searcher(db_path),
                db_path=db_path,
            )

            base_rate = research_result.base_rate
            confidence_interval = research_result.confidence_interval
            base_rate_sources = research_result.sources
            research_reasoning = research_result.reasoning

            if research_result.market_match:
                market = research_result.market_match
                market_price = market.current_probability
                market_url = market.url
                market_platform = market.platform
                probability_source = "market"
                text = market.title
            else:
                market_price = None
                market_url = None
                market_platform = None
                probability_source = "research"
                text = search_text
            match_confidence = None
        else:
            # Quick LLM estimate (fallback)
            base_rate, _ = await estimate_base_rate(search_text)
            market_price = None
            market_url = None
            market_platform = None
            probability_source = "llm"
            text = search_text
            match_confidence = None
            base_rate_sources = ["llm-estimate"]

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
            market_match_confidence=match_confidence,
            confidence_interval=confidence_interval,
            base_rate_sources=base_rate_sources,
            research_reasoning=research_reasoning,
            p_target_given_yes=0.01,  # Competitor wins → target loses
            p_target_given_no=min(0.99, parent_prior * 1.1),  # Slight boost
            rho_reasoning=constraint.reasoning,
            depth=1,
            is_leaf=True,
        )
        signals.append(signal)

    # 3. Handle causal pathways
    for pathway in structure.causal_pathways:
        market, match_confidence = await try_match_market(pathway.upstream_event)

        # Initialize fields for SignalNode
        confidence_interval = None
        base_rate_sources = []
        research_reasoning = None

        if market and market.current_probability is not None:
            base_rate = market.current_probability
            market_price = market.current_probability
            market_url = market.url
            market_platform = market.platform
            probability_source = "market"
            text = market.title
            base_rate_sources = [f"market:{market.platform}"]
        elif use_structured_research:
            # Use full research workflow when no direct market match
            from .research import research_signal
            from .markets import get_market_searcher

            research_result = await research_signal(
                pathway.upstream_event,
                target_context=structure.target,
                market_searcher=get_market_searcher(db_path),
                db_path=db_path,
            )

            base_rate = research_result.base_rate
            confidence_interval = research_result.confidence_interval
            base_rate_sources = research_result.sources
            research_reasoning = research_result.reasoning

            if research_result.market_match:
                market = research_result.market_match
                market_price = market.current_probability
                market_url = market.url
                market_platform = market.platform
                probability_source = "market"
                text = market.title
            else:
                market_price = None
                market_url = None
                market_platform = None
                probability_source = "research"
                text = pathway.upstream_event
            match_confidence = None
        else:
            # Quick LLM estimate (fallback)
            base_rate, _ = await estimate_base_rate(pathway.upstream_event)
            market_price = None
            market_url = None
            market_platform = None
            probability_source = "llm"
            text = pathway.upstream_event
            match_confidence = None
            base_rate_sources = ["llm-estimate"]

        # Estimate rho based on effect direction
        # For comparative questions, we need to consider which entity is affected
        if comparative.is_comparative and comparative.entity_a and comparative.entity_b:
            # Use comparative-aware rho estimation
            try:
                signal_effect = await determine_signal_entity_effect(
                    signal_text=text,
                    entity_a=comparative.entity_a,
                    entity_b=comparative.entity_b,
                    effect_description=pathway.effect_on_target,
                )
                rho = compute_rho_for_comparative(
                    primary_entity=signal_effect.primary_entity,
                    effect_direction=signal_effect.effect_direction,
                    entity_a=comparative.entity_a,
                    entity_b=comparative.entity_b,
                )
                logger.debug(
                    f"Comparative rho: {signal_effect.primary_entity} "
                    f"({signal_effect.effect_direction}) → rho={rho}"
                )
            except Exception as e:
                logger.warning(f"Comparative rho estimation failed: {e}, using fallback")
                # Fallback to simple keyword matching
                effect_lower = pathway.effect_on_target.lower()
                if "positive" in effect_lower or "helps" in effect_lower:
                    rho = 0.4
                elif "negative" in effect_lower or "hurts" in effect_lower:
                    rho = -0.4
                else:
                    rho, _ = await estimate_rho(structure.target, pathway.upstream_event)
        else:
            # Non-comparative question - use simple keyword matching
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
            market_match_confidence=match_confidence,
            confidence_interval=confidence_interval,
            base_rate_sources=base_rate_sources,
            research_reasoning=research_reasoning,
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
