"""Value of Information (VOI) calculations for signals.

Linear VOI is more stable than entropy-based VOI under magnitude noise,
especially for low-probability events. Experiments show +0.160 τ stability
advantage, rising to +0.352 τ at extreme base rates (<0.10 or >0.90).

Core insight: Linear VOI uses expected absolute belief shift instead of
entropy-based information gain, providing constant gradient rather than
steep gradients at probability extremes that amplify errors.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

# Import core VOI functions from llm-forecasting
from llm_forecasting.voi import entropy, entropy_voi, linear_voi

if TYPE_CHECKING:
    from .models import ForecastTree, GlobalScenario, Signal

# Re-export core functions for backwards compatibility
__all__ = [
    "linear_voi",
    "entropy_voi",
    "entropy",
    "estimate_posteriors",
    "compute_signal_voi",
    "compute_signal_voi_with_posteriors",
    "rank_signals_by_voi",
    "top_signals_by_voi",
    "signals_by_voi_threshold",
    "compare_voi_methods",
    "MAGNITUDE_SHIFTS",
]


# Default magnitude shift factors (proportion of distance to 0 or 1)
# These can be overridden when calling estimate_posteriors
MAGNITUDE_SHIFTS: dict[str, float] = {
    "small": 0.10,   # 10% of available probability space
    "medium": 0.25,  # 25% of available probability space
    "large": 0.50,   # 50% of available probability space
}


def estimate_posteriors(
    p_x: float,
    direction: Literal["increases", "decreases"],
    magnitude: Literal["small", "medium", "large"],
    magnitude_shifts: dict[str, float] | None = None,
) -> tuple[float, float]:
    """Estimate P(X|Q=yes) and P(X|Q=no) from direction and magnitude.

    Uses proportional shifts relative to available probability space:
    - For "increases": shift_yes moves toward 1, shift_no moves toward 0
    - For "decreases": shift_yes moves toward 0, shift_no moves toward 1

    The magnitude determines what fraction of available space to shift.

    Args:
        p_x: Prior probability P(X)
        direction: Whether signal firing increases or decreases scenario prob
        magnitude: Size of expected shift (small/medium/large)
        magnitude_shifts: Optional override for shift factors

    Returns:
        Tuple of (p_x_given_q_yes, p_x_given_q_no)
    """
    shifts = magnitude_shifts or MAGNITUDE_SHIFTS
    shift_factor = shifts.get(magnitude, 0.25)  # Default to medium

    if direction == "increases":
        # Signal firing increases probability - shift toward 1
        available_up = 1.0 - p_x
        available_down = p_x
        p_x_given_q_yes = p_x + shift_factor * available_up
        p_x_given_q_no = p_x - shift_factor * available_down
    else:
        # Signal firing decreases probability - shift toward 0
        available_down = p_x
        available_up = 1.0 - p_x
        p_x_given_q_yes = p_x - shift_factor * available_down
        p_x_given_q_no = p_x + shift_factor * available_up

    # Clamp to valid probability range
    p_x_given_q_yes = max(0.0, min(1.0, p_x_given_q_yes))
    p_x_given_q_no = max(0.0, min(1.0, p_x_given_q_no))

    return p_x_given_q_yes, p_x_given_q_no


def compute_signal_voi(
    signal: Signal,
    scenario: GlobalScenario,
    voi_type: Literal["linear", "entropy"] = "linear",
    magnitude_shifts: dict[str, float] | None = None,
) -> float:
    """Compute VOI for a signal affecting a specific scenario.

    Uses magnitude/direction to estimate posteriors if not explicitly provided.

    Args:
        signal: The Signal to evaluate
        scenario: The GlobalScenario the signal affects
        voi_type: "linear" (default, more stable) or "entropy"
        magnitude_shifts: Optional override for magnitude shift factors

    Returns:
        VOI value for this signal-scenario pair
    """
    # Prior probability
    p_x = scenario.probability

    # P(Q=yes) - probability signal fires
    # Default to 0.5 if not specified (maximum uncertainty)
    p_q = signal.current_probability if signal.current_probability is not None else 0.5

    # Estimate posteriors from direction and magnitude
    p_x_given_q_yes, p_x_given_q_no = estimate_posteriors(
        p_x=p_x,
        direction=signal.direction,
        magnitude=signal.magnitude,
        magnitude_shifts=magnitude_shifts,
    )

    # Compute VOI
    if voi_type == "linear":
        return linear_voi(p_x, p_q, p_x_given_q_yes, p_x_given_q_no)
    else:
        return entropy_voi(p_x, p_q, p_x_given_q_yes, p_x_given_q_no)


def compute_signal_voi_with_posteriors(
    signal: Signal,
    scenario: GlobalScenario,
    p_x_given_q_yes: float,
    p_x_given_q_no: float,
    voi_type: Literal["linear", "entropy"] = "linear",
) -> float:
    """Compute VOI with explicit posterior probabilities.

    Use this when you have elicited P(X|Q=yes) and P(X|Q=no) directly
    rather than estimating from magnitude.

    Args:
        signal: The Signal to evaluate
        scenario: The GlobalScenario the signal affects
        p_x_given_q_yes: Explicit P(X|Q=yes)
        p_x_given_q_no: Explicit P(X|Q=no)
        voi_type: "linear" (default, more stable) or "entropy"

    Returns:
        VOI value for this signal-scenario pair
    """
    p_x = scenario.probability
    p_q = signal.current_probability if signal.current_probability is not None else 0.5

    if voi_type == "linear":
        return linear_voi(p_x, p_q, p_x_given_q_yes, p_x_given_q_no)
    else:
        return entropy_voi(p_x, p_q, p_x_given_q_yes, p_x_given_q_no)


def rank_signals_by_voi(
    tree: ForecastTree,
    voi_type: Literal["linear", "entropy"] = "linear",
    magnitude_shifts: dict[str, float] | None = None,
) -> list[tuple[Signal, float]]:
    """Rank all signals in a tree by their VOI.

    Args:
        tree: ForecastTree containing signals and scenarios
        voi_type: "linear" (default, more stable) or "entropy"
        magnitude_shifts: Optional override for magnitude shift factors

    Returns:
        List of (Signal, voi_score) tuples, sorted descending by VOI
    """
    # Build scenario lookup
    scenario_map = {s.id: s for s in tree.global_scenarios}

    results: list[tuple[Signal, float]] = []

    for signal in tree.signals:
        scenario = scenario_map.get(signal.scenario_id)
        if scenario is None:
            # Skip signals pointing to unknown scenarios
            continue

        voi = compute_signal_voi(
            signal=signal,
            scenario=scenario,
            voi_type=voi_type,
            magnitude_shifts=magnitude_shifts,
        )
        results.append((signal, voi))

    # Sort by VOI descending
    results.sort(key=lambda x: x[1], reverse=True)
    return results


def top_signals_by_voi(
    tree: ForecastTree,
    n: int = 5,
    voi_type: Literal["linear", "entropy"] = "linear",
    magnitude_shifts: dict[str, float] | None = None,
) -> list[tuple[Signal, float]]:
    """Get top N signals by VOI.

    Args:
        tree: ForecastTree containing signals and scenarios
        n: Number of top signals to return
        voi_type: "linear" (default, more stable) or "entropy"
        magnitude_shifts: Optional override for magnitude shift factors

    Returns:
        List of top N (Signal, voi_score) tuples
    """
    ranked = rank_signals_by_voi(tree, voi_type, magnitude_shifts)
    return ranked[:n]


def signals_by_voi_threshold(
    tree: ForecastTree,
    min_voi: float,
    voi_type: Literal["linear", "entropy"] = "linear",
    magnitude_shifts: dict[str, float] | None = None,
) -> list[tuple[Signal, float]]:
    """Get signals with VOI above a threshold.

    Args:
        tree: ForecastTree containing signals and scenarios
        min_voi: Minimum VOI threshold
        voi_type: "linear" (default, more stable) or "entropy"
        magnitude_shifts: Optional override for magnitude shift factors

    Returns:
        List of (Signal, voi_score) tuples with VOI >= min_voi
    """
    ranked = rank_signals_by_voi(tree, voi_type, magnitude_shifts)
    return [(s, v) for s, v in ranked if v >= min_voi]


def compare_voi_methods(
    tree: ForecastTree,
    magnitude_shifts: dict[str, float] | None = None,
) -> list[dict]:
    """Compare linear vs entropy VOI rankings for analysis.

    Useful for understanding when the methods diverge,
    particularly for extreme base rates.

    Args:
        tree: ForecastTree containing signals and scenarios
        magnitude_shifts: Optional override for magnitude shift factors

    Returns:
        List of dicts with signal info and both VOI values
    """
    scenario_map = {s.id: s for s in tree.global_scenarios}

    results = []
    for signal in tree.signals:
        scenario = scenario_map.get(signal.scenario_id)
        if scenario is None:
            continue

        linear = compute_signal_voi(
            signal, scenario, "linear", magnitude_shifts
        )
        ent = compute_signal_voi(
            signal, scenario, "entropy", magnitude_shifts
        )

        results.append({
            "signal_id": signal.id,
            "signal_text": signal.text,
            "scenario_id": scenario.id,
            "scenario_probability": scenario.probability,
            "direction": signal.direction,
            "magnitude": signal.magnitude,
            "linear_voi": linear,
            "entropy_voi": ent,
            "voi_difference": linear - ent,
        })

    return results
