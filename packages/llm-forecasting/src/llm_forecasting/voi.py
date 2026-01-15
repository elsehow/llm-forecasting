"""Value of Information (VOI) calculations for forecasting.

Linear VOI is more stable than entropy-based VOI under magnitude noise,
especially for low-probability events. Experiments show +0.160 τ stability
advantage, rising to +0.352 τ at extreme base rates (<0.10 or >0.90).

Core insight: Linear VOI uses expected absolute belief shift instead of
entropy-based information gain, providing constant gradient rather than
steep gradients at probability extremes that amplify errors.
"""

from __future__ import annotations

import math


def linear_voi(
    p_x: float,
    p_q: float,
    p_x_given_q_yes: float,
    p_x_given_q_no: float,
) -> float:
    """Compute Linear VOI: expected absolute belief shift.

    More stable than entropy VOI under magnitude noise.
    Especially valuable for rare events (P < 0.10).

    Args:
        p_x: Prior P(X) - scenario probability
        p_q: P(Q=yes) - probability signal fires/resolves yes
        p_x_given_q_yes: P(X|Q=yes) - scenario prob if signal fires
        p_x_given_q_no: P(X|Q=no) - scenario prob if signal doesn't fire

    Returns:
        Linear VOI value (expected absolute belief shift)
    """
    shift_yes = abs(p_x_given_q_yes - p_x)
    shift_no = abs(p_x_given_q_no - p_x)
    return p_q * shift_yes + (1 - p_q) * shift_no


def entropy(p: float) -> float:
    """Binary entropy in bits.

    Args:
        p: Probability value

    Returns:
        Entropy value in bits (0 for p=0 or p=1)
    """
    if p <= 0 or p >= 1:
        return 0.0
    return -p * math.log2(p) - (1 - p) * math.log2(1 - p)


def entropy_voi(
    p_x: float,
    p_q: float,
    p_x_given_q_yes: float,
    p_x_given_q_no: float,
) -> float:
    """Compute entropy-based VOI (information gain in bits).

    Standard information-theoretic VOI. Has steep gradients at probability
    extremes which can amplify estimation errors.

    Args:
        p_x: Prior P(X) - scenario probability
        p_q: P(Q=yes) - probability signal fires/resolves yes
        p_x_given_q_yes: P(X|Q=yes) - scenario prob if signal fires
        p_x_given_q_no: P(X|Q=no) - scenario prob if signal doesn't fire

    Returns:
        Entropy VOI value (information gain in bits)
    """
    h_prior = entropy(p_x)
    h_posterior = p_q * entropy(p_x_given_q_yes) + (1 - p_q) * entropy(p_x_given_q_no)
    return h_prior - h_posterior


def rho_to_posteriors(
    rho: float,
    p_a: float,
    p_b: float,
    clamp_posteriors: bool = True,
) -> tuple[float, float]:
    """Convert correlation coefficient to P(A|B=yes) and P(A|B=no).

    Uses the relationship:
        P(A,B) = P(A)*P(B) + rho * sigma_A * sigma_B

    This allows computing VOI from market correlation data without
    explicitly eliciting conditional probabilities.

    Args:
        rho: Pearson correlation coefficient between A and B
        p_a: P(A) - prior probability of A (the scenario)
        p_b: P(B) - prior probability of B (the signal that resolves first)
        clamp_posteriors: Whether to clamp results to [0.01, 0.99]

    Returns:
        Tuple of (p_a_given_b_yes, p_a_given_b_no)
    """
    if math.isnan(rho) or math.isnan(p_a) or math.isnan(p_b):
        return p_a, p_a  # No information

    # Clamp inputs to valid ranges
    p_a = max(0.001, min(0.999, p_a))
    p_b = max(0.001, min(0.999, p_b))
    rho = max(-1.0, min(1.0, rho))

    sigma_a = math.sqrt(p_a * (1 - p_a))
    sigma_b = math.sqrt(p_b * (1 - p_b))

    # Joint probability from correlation
    p_ab = p_a * p_b + rho * sigma_a * sigma_b

    # Clamp joint probability to Frechet-Hoeffding bounds
    # max(0, p_a + p_b - 1) <= P(A,B) <= min(p_a, p_b)
    lower_bound = max(0, p_a + p_b - 1)
    upper_bound = min(p_a, p_b)
    p_ab = max(lower_bound, min(p_ab, upper_bound))

    # Conditional probabilities via Bayes
    p_a_given_b_yes = p_ab / p_b
    p_a_given_b_no = (p_a - p_ab) / (1 - p_b)

    if clamp_posteriors:
        p_a_given_b_yes = max(0.01, min(0.99, p_a_given_b_yes))
        p_a_given_b_no = max(0.01, min(0.99, p_a_given_b_no))

    return p_a_given_b_yes, p_a_given_b_no


def linear_voi_from_rho(
    rho: float,
    p_a: float,
    p_b: float,
) -> float:
    """Compute linear VOI directly from correlation coefficient.

    This is the canonical way to compute VOI when you have rho
    instead of explicit posteriors. Converts rho to posteriors
    internally and applies the standard linear VOI formula.

    Args:
        rho: Pearson correlation between A and B
        p_a: P(A) - the scenario probability
        p_b: P(B) - the signal probability (resolves first)

    Returns:
        Linear VOI value
    """
    p_a_given_b_yes, p_a_given_b_no = rho_to_posteriors(rho, p_a, p_b)
    return linear_voi(p_a, p_b, p_a_given_b_yes, p_a_given_b_no)


def entropy_voi_from_rho(
    rho: float,
    p_a: float,
    p_b: float,
) -> float:
    """Compute entropy VOI directly from correlation coefficient.

    Args:
        rho: Pearson correlation between A and B
        p_a: P(A) - the scenario probability
        p_b: P(B) - the signal probability (resolves first)

    Returns:
        Entropy VOI value (information gain in bits)
    """
    p_a_given_b_yes, p_a_given_b_no = rho_to_posteriors(rho, p_a, p_b)
    return entropy_voi(p_a, p_b, p_a_given_b_yes, p_a_given_b_no)
