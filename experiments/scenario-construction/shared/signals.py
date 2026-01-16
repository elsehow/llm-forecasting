"""Shared signal utilities for scenario construction."""

from pathlib import Path

from llm_forecasting.semantic_search import SemanticSignalSearcher
from llm_forecasting.voi import (
    estimate_rho_batch,
    estimate_conditional_expectations_batch,
    linear_voi_from_rho,
    rho_to_posteriors,
)


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

# Default max horizon for actionable signals (1 year from now)
DEFAULT_MAX_HORIZON_DAYS = 365


def parse_date(d: str | None):
    """Parse ISO date string to date object, returning None on failure."""
    from datetime import date as date_type

    if d is None:
        return None
    try:
        return date_type.fromisoformat(str(d)[:10])
    except ValueError:
        return None


def enrich_with_resolution_data(
    signals: list[dict],
    db_path,
    max_horizon_days: int = DEFAULT_MAX_HORIZON_DAYS,
) -> list[dict]:
    """
    Add resolution metadata and URL to signals from database.

    Enriches each signal dict with:
    - resolution_date: ISO date string or None
    - resolved: bool
    - resolution_value: float or None
    - base_rate: float or None
    - url: str or None
    - signal_category: one of "exclude", "actionable", "future", "unknown"

    Args:
        signals: List of dicts with "id" and "source" keys
        db_path: Path to forecastbench.db
        max_horizon_days: Only include signals resolving within this many days from now

    Returns:
        The same list with added fields (mutates in place and returns)
    """
    import sqlite3

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    for s in signals:
        cursor.execute(
            "SELECT resolution_date, resolved, resolution_value, base_rate, url FROM questions WHERE id = ? AND source = ?",
            (s["id"], s["source"])
        )
        row = cursor.fetchone()
        if row:
            res_date = row["resolution_date"]
            resolved = bool(row["resolved"])
            s["resolution_date"] = str(res_date) if res_date else None
            s["resolved"] = resolved
            s["resolution_value"] = row["resolution_value"]
            s["base_rate"] = row["base_rate"]
            s["url"] = row["url"]
            s["signal_category"] = categorize_signal(s["resolution_date"], resolved, max_horizon_days)
        else:
            s["resolution_date"] = None
            s["resolved"] = False
            s["url"] = None
            s["signal_category"] = "unknown"

    conn.close()
    return signals


def filter_by_resolution_date(
    signals: list[dict],
    max_horizon_days: int = DEFAULT_MAX_HORIZON_DAYS,
) -> list[dict]:
    """
    Filter signals by resolution date, keeping only actionable ones.

    Works on any signals (market or LLM-generated) as long as they have:
    - resolution_date: ISO date string or None
    - resolved: bool (defaults to False if not present)

    Sets signal_category on each signal and returns only non-excluded signals.

    Args:
        signals: List of signal dicts
        max_horizon_days: Only keep signals resolving within this many days

    Returns:
        Filtered list (signals with category != "exclude")
    """
    result = []
    for s in signals:
        category = categorize_signal(
            resolution_date=s.get("resolution_date"),
            resolved=s.get("resolved", False),
            max_horizon_days=max_horizon_days,
        )
        s["signal_category"] = category
        if category != "exclude":
            result.append(s)
    return result


def categorize_signal(
    resolution_date: str | None,
    resolved: bool,
    max_horizon_days: int = DEFAULT_MAX_HORIZON_DAYS,
) -> str:
    """
    Categorize signal by resolution timing relative to today.

    Simple date-range filtering:
    - "exclude": Already resolved OR resolution date has passed
    - "actionable": Unresolved, resolves within max_horizon_days from now
    - "future": Unresolved, resolves beyond max_horizon_days
    - "unknown": No resolution date available

    Args:
        resolution_date: ISO format date string or None
        resolved: Whether the signal has already resolved
        max_horizon_days: Maximum days from now for actionable signals

    Returns:
        Category string
    """
    from datetime import datetime

    if resolution_date is None:
        return "unknown"

    try:
        res_date = datetime.strptime(str(resolution_date)[:10], "%Y-%m-%d").date()
        today = datetime.now().date()
    except ValueError:
        return "unknown"

    # Already resolved = exclude (we know the outcome)
    if resolved:
        return "exclude"

    days_until = (res_date - today).days

    # Resolution date has passed but not marked resolved = treat as exclude
    if days_until <= 0:
        return "exclude"
    elif days_until <= max_horizon_days:
        return "actionable"  # Resolves soon - high value
    else:
        return "future"  # Too far out


# ============================================================
# Fetch missing probabilities from market APIs
# ============================================================

# Map source names to market data provider names
MARKET_SOURCE_PROVIDERS = {
    "polymarket": "polymarket",
    "metaculus": "metaculus",
    "manifold": "manifold",
    "kalshi": "kalshi",
}


async def fetch_missing_probabilities(
    signals: list[dict],
    verbose: bool = True,
) -> list[dict]:
    """Fetch current probabilities from market APIs for signals missing base_rate.

    For signals from known market sources (Polymarket, Metaculus, etc.) that
    don't have a base_rate, fetches the current market probability.

    Args:
        signals: List of signal dicts with "id", "source", and optional "base_rate"
        verbose: Whether to print progress

    Returns:
        Same list with base_rate populated where possible (mutates in place)
    """
    from llm_forecasting.market_data import market_data_registry

    # Group signals by source that need probabilities
    signals_by_source: dict[str, list[dict]] = {}
    for s in signals:
        if s.get("base_rate") is not None:
            continue  # Already have probability
        source = s.get("source", "")
        if source in MARKET_SOURCE_PROVIDERS:
            if source not in signals_by_source:
                signals_by_source[source] = []
            signals_by_source[source].append(s)

    if not signals_by_source:
        return signals

    # Fetch probabilities for each source
    for source, source_signals in signals_by_source.items():
        provider_name = MARKET_SOURCE_PROVIDERS[source]
        if provider_name not in market_data_registry:
            if verbose:
                print(f"  Warning: No provider for {source}, skipping {len(source_signals)} signals")
            continue

        provider_class = market_data_registry.get(provider_name)
        provider = provider_class()

        fetched = 0
        failed = 0
        for s in source_signals:
            try:
                market = await provider.fetch_market(s["id"])
                if market and market.current_probability is not None:
                    s["base_rate"] = market.current_probability
                    fetched += 1
                else:
                    failed += 1
            except Exception as e:
                if verbose:
                    print(f"    Error fetching {s['id']}: {e}")
                failed += 1

        if verbose:
            print(f"  {source}: fetched {fetched}/{len(source_signals)} probabilities ({failed} failed)")

    return signals


# ============================================================
# VOI-based signal ranking
# ============================================================

async def rank_signals_by_voi(
    signals: list[dict],
    target: str,
    target_prior: float = 0.5,
    is_continuous: bool = False,
) -> list[dict]:
    """
    Rank signals by VOI relative to a target outcome.

    Uses Anthropic batch API for efficient rho estimation (one prompt per signal,
    all submitted as single batch for 50% cost savings).

    For continuous targets, also estimates E[target|signal=YES] and E[target|signal=NO]
    via a second batch call.

    Args:
        signals: List of dicts with "question" or "text" key
        target: The target question (e.g., "What will US GDP be in 2050?")
        target_prior: Prior probability for target (default 0.5 for maximum uncertainty)
        is_continuous: Whether target is a continuous question (triggers E[X|signal] estimation)

    Returns:
        Signals sorted by VOI descending, with added "rho", "voi", and "rho_reasoning" keys
        For continuous targets, also adds "e_target_given_yes" and "e_target_given_no"
    """
    if not signals:
        return []

    # Build pairs for batch rho estimation
    pairs = []
    for s in signals:
        signal_text = s.get("question") or s.get("text", "")
        pairs.append((target, signal_text))

    # Estimate rho for all pairs using batch API
    print("  Estimating rho for all signals...")
    rho_results = await estimate_rho_batch(pairs)

    # For continuous targets, also estimate E[target|signal]
    cond_exp_results = None
    if is_continuous:
        print("  Estimating E[target|signal] for continuous target...")
        cond_exp_results = await estimate_conditional_expectations_batch(pairs)

    # Compute VOI and conditional probabilities for all signals
    for i, s in enumerate(signals):
        rho, reasoning = rho_results[i]
        # Get signal probability from market data if available, else 0.5
        p_signal = s.get("base_rate") or s.get("probability") or 0.5

        # Compute linear VOI
        voi = linear_voi_from_rho(rho, target_prior, p_signal)

        # Compute conditional probabilities: P(target|signal=YES) and P(target|signal=NO)
        p_target_given_yes, p_target_given_no = rho_to_posteriors(rho, target_prior, p_signal)
        spread = abs(p_target_given_yes - p_target_given_no)

        s["rho"] = rho
        s["rho_reasoning"] = reasoning
        s["voi"] = voi
        s["p_target_given_yes"] = p_target_given_yes
        s["p_target_given_no"] = p_target_given_no
        s["cruxiness_spread"] = spread

        # For continuous targets, add E[target|signal] estimates
        if cond_exp_results:
            e_yes, e_no, _ = cond_exp_results[i]
            s["e_target_given_yes"] = e_yes
            s["e_target_given_no"] = e_no

    # Sort by VOI descending
    return sorted(signals, key=lambda x: x.get("voi", 0), reverse=True)


async def rank_and_report_signals(
    signals: list[dict],
    target: str,
    target_prior: float = 0.5,
    voi_floor: float = 0.1,
    is_continuous: bool = False,
) -> list[dict]:
    """Rank signals by VOI and print analysis.

    Wrapper around rank_signals_by_voi that also prints:
    - Count of signals above VOI floor
    - Top 5 signals by VOI

    Args:
        signals: List of signal dicts
        target: Target question text
        target_prior: Prior probability for target
        voi_floor: Minimum VOI to count as "above floor"
        is_continuous: Whether target is continuous (triggers E[X|signal] estimation)

    Returns:
        Signals sorted by VOI descending
    """
    print("Ranking signals by VOI (batch API)...")
    ranked = await rank_signals_by_voi(signals, target, target_prior, is_continuous=is_continuous)

    above_floor = sum(1 for s in ranked if s.get("voi", 0) >= voi_floor)
    print(f"  {above_floor} signals above VOI floor ({voi_floor})")

    print(f"\n  Top 5 by VOI:")
    for s in ranked[:5]:
        voi = s.get("voi", 0)
        text = s.get("text") or s.get("question", "")
        print(f"    VOI={voi:.2f} {text[:50]}...")

    return ranked


def deduplicate_and_report(
    signals: list[dict],
    threshold: float = 0.45,
    prefer_observed: bool = True,
    group_by: str | None = None,
) -> list[dict]:
    """Deduplicate signals and print metrics.

    Args:
        signals: List of signal dicts with "text" key
        threshold: Similarity threshold for deduplication
        prefer_observed: Whether to prefer observed sources over LLM
        group_by: Optional field to group counts by (e.g., "signal_category")

    Returns:
        Deduplicated signals
    """
    print("Deduplicating signals...")
    before = len(signals)

    # Use SemanticSignalSearcher for deduplication
    searcher = SemanticSignalSearcher()
    deduped = searcher.deduplicate(signals, threshold=threshold, prefer_observed=prefer_observed)

    print(f"  {before} → {len(deduped)} (removed {before - len(deduped)} duplicates)")

    if group_by:
        counts = {}
        for s in deduped:
            key = s.get(group_by, "unknown")
            counts[key] = counts.get(key, 0) + 1
        print(f"  By {group_by}: {counts}")

    return deduped


# ============================================================
# Signal model construction
# ============================================================

def build_signal_models(
    signals: list[dict],
    text_key: str = "question",
    include_background: bool = False,
    include_uncertainty_source: bool = False,
) -> list:
    """Build Signal model instances from signal dicts.

    Args:
        signals: List of signal dicts with id, source, text/question, etc.
        text_key: Key to use for signal text ("question" for market, "text" for LLM)
        include_background: Whether to include background field (for LLM signals)
        include_uncertainty_source: Whether to include uncertainty_source field

    Returns:
        List of Signal model instances
    """
    from llm_forecasting.models import Signal

    result = []
    for s in signals:
        kwargs = {
            "id": s["id"],
            "source": s["source"],
            "text": s.get(text_key) or s.get("text") or s.get("question"),
            "url": s.get("url"),
            "resolution_date": parse_date(s.get("resolution_date")),
            "base_rate": s.get("base_rate"),
            "voi": s.get("voi", 0.0),
            "rho": s.get("rho", 0.0),
            "rho_reasoning": s.get("rho_reasoning"),
            # Conditional probability fields (binary targets)
            "p_target_given_yes": s.get("p_target_given_yes"),
            "p_target_given_no": s.get("p_target_given_no"),
            "cruxiness_spread": s.get("cruxiness_spread"),
            # Conditional expectation fields (continuous targets)
            "e_target_given_yes": s.get("e_target_given_yes"),
            "e_target_given_no": s.get("e_target_given_no"),
        }

        if include_background:
            kwargs["background"] = s.get("background")

        if include_uncertainty_source:
            kwargs["uncertainty_source"] = s.get("uncertainty_source")

        result.append(Signal(**kwargs))

    return result
