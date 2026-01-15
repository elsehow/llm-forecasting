"""Value of Information (VOI) calculations for forecasting.

Linear VOI is more stable than entropy-based VOI under magnitude noise,
especially for low-probability events. Experiments show +0.160 τ stability
advantage, rising to +0.352 τ at extreme base rates (<0.10 or >0.90).

Core insight: Linear VOI uses expected absolute belief shift instead of
entropy-based information gain, providing constant gradient rather than
steep gradients at probability extremes that amplify errors.

Also includes LLM-based rho estimation for computing VOI when correlation
is not directly observable.
"""

from __future__ import annotations

import json
import math
import os
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    pass


# =============================================================================
# Prompt for LLM-based rho estimation
# =============================================================================

RHO_ESTIMATION_PROMPT = """Estimate the correlation coefficient (ρ) between these two forecasting questions.

ρ ranges from -1 to +1:
- ρ = +1: Perfect positive correlation (if A happens, B definitely happens)
- ρ = 0: Independent (A and B are unrelated)
- ρ = -1: Perfect negative correlation (if A happens, B definitely doesn't)

Question A (target): {question_a}
Question B (signal): {question_b}

Consider:
1. Is there a causal relationship?
2. Are they measuring the same underlying phenomenon?
3. Could they be mutually exclusive?
4. Are they truly independent?

Respond with JSON only: {{"rho": <number between -1 and 1>, "reasoning": "<brief explanation>"}}"""


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

    Standard information-theoretic VOI: H(X) - H(X|Q).
    Has steep gradients at probability extremes which can amplify errors.

    Formula from information theory:
        VOI = H(U) - H(U|C)
            = h(P(U)) - [P(C) × h(P(U|C)) + (1-P(C)) × h(P(U|¬C))]

    Where h is binary entropy: h(p) = p × log(1/p) + (1-p) × log(1/(1-p))

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


def entropy_voi_normalized(
    p_x: float,
    p_q: float,
    p_x_given_q_yes: float,
    p_x_given_q_no: float,
) -> float:
    """Compute entropy VOI as fraction of maximum possible information gain.

    Expresses VOI as percentage of maximum possible VOI given the prior:
        VOI_normalized = (H(U) - H(U|C)) / H(U)

    This is useful for comparing VOI across questions with different priors,
    since a 0.1 bit reduction means more when starting from 0.5 bits than 1 bit.

    Args:
        p_x: Prior P(X) - scenario probability
        p_q: P(Q=yes) - probability signal fires/resolves yes
        p_x_given_q_yes: P(X|Q=yes) - scenario prob if signal fires
        p_x_given_q_no: P(X|Q=no) - scenario prob if signal doesn't fire

    Returns:
        Normalized entropy VOI (fraction in [0, 1], or 0 if prior is certain)
    """
    h_prior = entropy(p_x)
    if h_prior <= 0:
        return 0.0
    raw_voi = entropy_voi(p_x, p_q, p_x_given_q_yes, p_x_given_q_no)
    return raw_voi / h_prior


def compare_voi_metrics(
    p_x: float,
    p_q: float,
    p_x_given_q_yes: float,
    p_x_given_q_no: float,
) -> dict[str, float]:
    """Compute all VOI metrics for side-by-side comparison.

    Useful for analyzing when linear and entropy VOI diverge,
    particularly at extreme base rates.

    Args:
        p_x: Prior P(X) - scenario probability
        p_q: P(Q=yes) - probability signal fires/resolves yes
        p_x_given_q_yes: P(X|Q=yes) - scenario prob if signal fires
        p_x_given_q_no: P(X|Q=no) - scenario prob if signal doesn't fire

    Returns:
        Dict with linear_voi, entropy_voi, entropy_voi_normalized, and max_entropy
    """
    return {
        "linear_voi": linear_voi(p_x, p_q, p_x_given_q_yes, p_x_given_q_no),
        "entropy_voi": entropy_voi(p_x, p_q, p_x_given_q_yes, p_x_given_q_no),
        "entropy_voi_normalized": entropy_voi_normalized(
            p_x, p_q, p_x_given_q_yes, p_x_given_q_no
        ),
        "max_entropy": entropy(p_x),
    }


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


def entropy_voi_normalized_from_rho(
    rho: float,
    p_a: float,
    p_b: float,
) -> float:
    """Compute normalized entropy VOI from correlation coefficient.

    Args:
        rho: Pearson correlation between A and B
        p_a: P(A) - the scenario probability
        p_b: P(B) - the signal probability (resolves first)

    Returns:
        Normalized entropy VOI (fraction of maximum possible)
    """
    p_a_given_b_yes, p_a_given_b_no = rho_to_posteriors(rho, p_a, p_b)
    return entropy_voi_normalized(p_a, p_b, p_a_given_b_yes, p_a_given_b_no)


def compare_voi_metrics_from_rho(
    rho: float,
    p_a: float,
    p_b: float,
) -> dict[str, float]:
    """Compute all VOI metrics from correlation coefficient.

    Convenience function for comparing VOI methods when you have
    market correlation (rho) rather than explicit posteriors.

    Args:
        rho: Pearson correlation between A and B
        p_a: P(A) - the scenario probability
        p_b: P(B) - the signal probability (resolves first)

    Returns:
        Dict with linear_voi, entropy_voi, entropy_voi_normalized, and max_entropy
    """
    p_a_given_b_yes, p_a_given_b_no = rho_to_posteriors(rho, p_a, p_b)
    return compare_voi_metrics(p_a, p_b, p_a_given_b_yes, p_a_given_b_no)


# =============================================================================
# LLM-based rho estimation
# =============================================================================

# Default model for rho estimation (cheap model for bulk operations)
DEFAULT_RHO_MODEL = "anthropic/claude-3-haiku-20240307"


async def estimate_rho(
    question_a: str,
    question_b: str,
    model: str | None = None,
) -> tuple[float, str]:
    """Estimate correlation (rho) between two questions using LLM.

    Args:
        question_a: The target/ultimate question
        question_b: The signal/crux question
        model: LLM model to use (defaults to haiku for cost efficiency)

    Returns:
        Tuple of (rho value, reasoning string)
    """
    import litellm

    model = model or DEFAULT_RHO_MODEL

    try:
        response = await litellm.acompletion(
            model=model,
            messages=[{
                "role": "user",
                "content": RHO_ESTIMATION_PROMPT.format(
                    question_a=question_a,
                    question_b=question_b,
                )
            }],
            max_tokens=200,
            temperature=0,
        )
        text = response.choices[0].message.content.strip()

        # Handle markdown code blocks
        if "```" in text:
            text = text.split("```")[1]
            if text.startswith("json"):
                text = text[4:]
            text = text.strip()

        result = json.loads(text)
        return float(result["rho"]), result.get("reasoning", "")
    except Exception as e:
        return 0.0, f"Error: {e}"


async def estimate_rho_batch(
    pairs: list[tuple[str, str]],
    model: str | None = None,
    poll_interval: float = 5.0,
    max_wait: float = 3600.0,
) -> list[tuple[float, str]]:
    """Estimate rho for multiple pairs using Anthropic batch API.

    Each pair gets its own prompt (no attention degradation).
    All submitted as one batch request (50% cost savings, faster).

    Args:
        pairs: List of (target_question, signal_question) tuples
        model: Model to use (defaults to haiku)
        poll_interval: Seconds between batch status checks
        max_wait: Maximum seconds to wait for batch completion

    Returns:
        List of (rho, reasoning) tuples in same order as input pairs
    """
    import asyncio
    import anthropic

    if not pairs:
        return []

    model = model or DEFAULT_RHO_MODEL
    # Extract base model name for Anthropic API (remove provider prefix)
    anthropic_model = model.replace("anthropic/", "")

    client = anthropic.Anthropic()

    # Build batch requests
    requests = []
    for i, (question_a, question_b) in enumerate(pairs):
        requests.append({
            "custom_id": f"rho_{i}",
            "params": {
                "model": anthropic_model,
                "max_tokens": 200,
                "messages": [{
                    "role": "user",
                    "content": RHO_ESTIMATION_PROMPT.format(
                        question_a=question_a,
                        question_b=question_b,
                    )
                }],
            }
        })

    # Submit batch
    batch = client.messages.batches.create(requests=requests)
    batch_id = batch.id

    # Poll for completion
    elapsed = 0.0
    while elapsed < max_wait:
        batch_status = client.messages.batches.retrieve(batch_id)
        if batch_status.processing_status == "ended":
            break
        await asyncio.sleep(poll_interval)
        elapsed += poll_interval

    if elapsed >= max_wait:
        raise TimeoutError(f"Batch {batch_id} did not complete within {max_wait}s")

    # Collect results
    results_by_id: dict[str, tuple[float, str]] = {}
    for result in client.messages.batches.results(batch_id):
        custom_id = result.custom_id
        if result.result.type == "succeeded":
            text = result.result.message.content[0].text.strip()
            try:
                # Handle markdown code blocks
                if "```" in text:
                    text = text.split("```")[1]
                    if text.startswith("json"):
                        text = text[4:]
                    text = text.strip()
                parsed = json.loads(text)
                results_by_id[custom_id] = (float(parsed["rho"]), parsed.get("reasoning", ""))
            except Exception as e:
                results_by_id[custom_id] = (0.0, f"Parse error: {e}")
        else:
            results_by_id[custom_id] = (0.0, f"Batch error: {result.result.type}")

    # Return in original order
    return [results_by_id.get(f"rho_{i}", (0.0, "Missing result")) for i in range(len(pairs))]
