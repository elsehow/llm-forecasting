"""Shared signal utilities for scenario construction."""

from pathlib import Path

from llm_forecasting.semantic_search import SemanticSignalSearcher
from llm_forecasting.voi import estimate_rho_batch, linear_voi_from_rho


# ============================================================
# Resolvability prompt (used by LLM signal generation)
# ============================================================

RESOLVABILITY_REQUIREMENTS = """Each signal MUST be resolvable as a prediction market question:
- **Specific deadline**: Include exact date or timeframe ("by December 2027")
- **Measurable threshold**: Include numeric criteria (">$500B", "exceeds 10%")
- **Verifiable source**: Specify how we'd know it happened ("per BLS data", "in SEC filings")

BAD: "AI significantly improves productivity"
GOOD: "By December 2027, at least 3 Fortune 500 companies report >20% productivity gains from AI tools in SEC filings"

BAD: "Major geopolitical conflict occurs"
GOOD: "Before January 2028, China imposes a naval blockade of Taiwan lasting >30 days, per US State Department confirmation"
"""


# ============================================================
# Semantic search for market signals
# ============================================================

def load_market_signals_semantic(
    db_path: Path,
    query: str,
    sources: list[str],
    top_k: int = 150,
) -> list[dict]:
    """
    Load market signals using semantic search.

    Args:
        db_path: Path to forecastbench.db
        query: Semantic search query (e.g., "What signals affect US GDP 2040?")
        sources: List of sources to include (e.g., ["polymarket", "metaculus"])
        top_k: Number of results to retrieve

    Returns:
        List of dicts with id, source, question, similarity
    """
    searcher = SemanticSignalSearcher(db_path=str(db_path))

    if not searcher.cache.exists:
        print("  Building embedding cache (one-time)...")
        searcher.build_cache()

    results = searcher.search(query, top_k=top_k)

    return [
        {
            "id": r.id,
            "source": r.source,
            "question": r.text,
            "similarity": r.similarity,
        }
        for r in results
        if r.source in sources
    ]


# ============================================================
# Deduplication
# ============================================================

def deduplicate_signals(
    signals: list[dict],
    threshold: float = 0.45,
) -> list[dict]:
    """
    Deduplicate signals, preferring observed (market) over LLM-generated.

    Args:
        signals: List of dicts with keys "text", "source", "reasoning" (market signals should come first)
        threshold: Similarity threshold (0-1). MiniLM: ~0.45 for semantic duplicates.

    Returns:
        Deduplicated list with observed sources preferred
    """
    if not signals:
        return []

    searcher = SemanticSignalSearcher()
    unique = searcher.deduplicate(signals, threshold=threshold, prefer_observed=True)
    return unique


def deduplicate_market_signals(
    signals: list[dict],
    threshold: float = 0.45,
) -> list[dict]:
    """
    Deduplicate market signals (dict format, for bottom-up approach).

    Args:
        signals: List of dicts with "question" and "source" keys
        threshold: Similarity threshold

    Returns:
        Deduplicated list
    """
    if not signals:
        return []

    searcher = SemanticSignalSearcher()

    # Convert question -> text for deduplication
    signal_dicts = [
        {"text": s["question"], "source": s["source"], **{k: v for k, v in s.items() if k != "question"}}
        for s in signals
    ]

    unique = searcher.deduplicate(signal_dicts, threshold=threshold, prefer_observed=True)

    # Convert back
    return [
        {"question": s["text"], **{k: v for k, v in s.items() if k != "text"}}
        for s in unique
    ]


# ============================================================
# Resolution date utilities
# ============================================================

# Default knowledge cutoff for Claude Sonnet 4
DEFAULT_KNOWLEDGE_CUTOFF = "2025-10-01"


def resolution_proximity_score(
    resolution_date: str | None,
    cutoff: str = DEFAULT_KNOWLEDGE_CUTOFF,
) -> float:
    """
    Score signals by how soon they resolve relative to knowledge cutoff.

    Returns 1.0 for high-priority (resolves within 1 year), decays for longer horizons.

    Args:
        resolution_date: ISO format date string (YYYY-MM-DD) or None
        cutoff: Knowledge cutoff date (model's training data end)

    Returns:
        Score from 0.3 (5+ years) to 1.0 (within 1 year or already resolved)
    """
    from datetime import datetime

    if resolution_date is None:
        return 0.5  # Unknown resolution date gets neutral score

    try:
        cutoff_date = datetime.strptime(cutoff, "%Y-%m-%d").date()
        res_date = datetime.strptime(str(resolution_date)[:10], "%Y-%m-%d").date()
    except ValueError:
        return 0.5  # Unparseable date gets neutral score

    days_until = (res_date - cutoff_date).days

    if days_until <= 0:
        return 1.0  # Already resolved (post-cutoff) = immediate value
    elif days_until <= 365:
        return 1.0  # Within 1 year = high priority
    elif days_until <= 730:
        return 0.7  # 1-2 years
    elif days_until <= 1095:
        return 0.5  # 2-3 years
    else:
        return 0.3  # 3+ years


def get_resolution_bucket(
    resolution_date: str | None,
    cutoff: str = DEFAULT_KNOWLEDGE_CUTOFF,
) -> str:
    """
    Categorize signal by resolution timeline.

    Args:
        resolution_date: ISO format date string or None
        cutoff: Knowledge cutoff date

    Returns:
        One of: "gold", "near_term", "medium_term", "long_term", "unknown"
    """
    from datetime import datetime

    if resolution_date is None:
        return "unknown"

    try:
        cutoff_date = datetime.strptime(cutoff, "%Y-%m-%d").date()
        res_date = datetime.strptime(str(resolution_date)[:10], "%Y-%m-%d").date()
        today = datetime.now().date()
    except ValueError:
        return "unknown"

    days_from_cutoff = (res_date - cutoff_date).days

    if days_from_cutoff <= 0:
        return "gold"  # Resolved after cutoff = model doesn't know outcome
    elif days_from_cutoff <= 365:
        return "near_term"  # Within 1 year of cutoff
    elif days_from_cutoff <= 1095:
        return "medium_term"  # 1-3 years
    else:
        return "long_term"  # 3+ years


def categorize_signal(
    resolution_date: str | None,
    resolved: bool,
    cutoff: str = DEFAULT_KNOWLEDGE_CUTOFF,
) -> str:
    """
    Categorize signal by resolution status relative to cutoff.

    This determines signal VALUE for forecasting:
    - "exclude": Resolved before cutoff, model already knows outcome
    - "gold": Resolved after cutoff, model doesn't know but we do
    - "near_term": Unresolved, resolving within 1 year
    - "future": Unresolved, resolving later
    - "unknown": No resolution date available

    Args:
        resolution_date: ISO format date string or None
        resolved: Whether the signal has already resolved
        cutoff: Knowledge cutoff date

    Returns:
        Category string
    """
    from datetime import datetime

    if resolution_date is None:
        return "unknown"

    try:
        cutoff_date = datetime.strptime(cutoff, "%Y-%m-%d").date()
        res_date = datetime.strptime(str(resolution_date)[:10], "%Y-%m-%d").date()
        today = datetime.now().date()
    except ValueError:
        return "unknown"

    if resolved:
        if res_date <= cutoff_date:
            return "exclude"  # Model already knows
        else:
            return "gold"  # Model doesn't know, we do
    else:
        days_until = (res_date - cutoff_date).days
        if days_until <= 365:
            return "near_term"
        else:
            return "future"


# ============================================================
# VOI-based signal ranking
# ============================================================

async def rank_signals_by_voi(
    signals: list[dict],
    target: str,
    target_prior: float = 0.5,
) -> list[dict]:
    """
    Rank signals by VOI relative to a target outcome.

    Uses Anthropic batch API for efficient rho estimation (one prompt per signal,
    all submitted as single batch for 50% cost savings).

    Args:
        signals: List of dicts with "question" or "text" key
        target: The target question (e.g., "What will US GDP be in 2050?")
        target_prior: Prior probability for target (default 0.5 for maximum uncertainty)

    Returns:
        Signals sorted by VOI descending, with added "rho", "voi", and "rho_reasoning" keys
    """
    if not signals:
        return []

    # Build pairs for batch rho estimation
    pairs = []
    for s in signals:
        signal_text = s.get("question") or s.get("text", "")
        pairs.append((target, signal_text))

    # Estimate rho for all pairs using batch API
    rho_results = await estimate_rho_batch(pairs)

    # Compute VOI and add to signals
    for i, s in enumerate(signals):
        rho, reasoning = rho_results[i]
        # Get signal probability from market data if available, else 0.5
        p_signal = s.get("base_rate") or s.get("probability") or 0.5

        # Compute linear VOI
        voi = linear_voi_from_rho(rho, target_prior, p_signal)

        s["rho"] = rho
        s["rho_reasoning"] = reasoning
        s["voi"] = voi

    # Sort by VOI descending
    return sorted(signals, key=lambda x: x.get("voi", 0), reverse=True)
