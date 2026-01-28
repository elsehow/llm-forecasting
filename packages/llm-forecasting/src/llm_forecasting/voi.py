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
# CANONICAL ρ ESTIMATION PROMPTS
# =============================================================================
#
# Two-step estimation with entity identification (recommended):
# - Updated 2026-01-19: Added explicit entity identification step
#   Based on vault research: Two-Stage Elicitation achieves 0% false positives
#   by forcing explicit entity identification before direction classification.
# - Key insight: Competition errors happen when model confuses "different entity
#   succeeding at same goal" with "quality momentum" - explicit entity ID fixes this
# - Separates: 1) identify entities, 2) classify relationship, 3) determine magnitude
#
# Known failure modes that this fixes:
# - "Sinners nominated for Oscar" was wrongly positive for "One Battle wins Oscar"
#   (model saw "nomination = quality validation" instead of "different film = competition")
# - Now catches: same entity → momentum, different entity same prize → competition
#
# Single-step estimation (legacy):
# - Validated 2026-01-14: 90% direction accuracy (Exp3)
# - Known issue: sign errors on competition scenarios (e.g., rival films, opposing candidates)
#
# Do not duplicate - import from here
# =============================================================================

# --- Two-step prompts (recommended) ---

RHO_DIRECTION_PROMPT = """Question A (target): "{question_a}"
Question B (signal): "{question_b}"

STEP 1 - ENTITY IDENTIFICATION (required):
- What specific entity does A refer to? (e.g., "Film X", "Candidate Y", "Company Z")
- What specific entity does B refer to? (e.g., "Film X", "Film Y", "same entity as A")
- Are these the SAME entity or DIFFERENT entities?

STEP 2 - RELATIONSHIP TYPE:
Based on your entity identification, classify the relationship:

IF SAME ENTITY:
- LOGICAL NECESSITY: B=YES is required for or guarantees A=YES → more_likely
  Example: "Film X wins Oscar" requires "Film X nominated for Oscar"
- MOMENTUM: B=YES signals quality that helps A → more_likely
  Example: "Film X wins Golden Globe" → "Film X wins Oscar" (awards momentum)

IF DIFFERENT ENTITIES:
- DIRECT COMPETITION: A and B are different entities competing for the SAME scarce resource → less_likely
  Example: "Film Y nominated for Oscar" HURTS "Film X wins Oscar" (more competition)
  Example: "Candidate Y wins primary" HURTS "Candidate X wins election" (stronger opponent)
  Key test: Does B being successful mean A has MORE competition or a HARDER path?
- INDIRECT HELP: B succeeding helps A succeed → more_likely
  Example: "Ally wins their race" → "Party wins majority" (coalition building)

IF NO CLEAR CONNECTION: → no_effect

IMPORTANT: A different entity succeeding at the same goal is ALWAYS competition (less_likely).
Nominations, wins, and recognition for competitors HURT the target's chances.

Respond with JSON only:
{{"entity_a": "<entity from A>", "entity_b": "<entity from B>", "same_entity": true/false, "direction": "more_likely" or "less_likely" or "no_effect", "reasoning": "<brief explanation>"}}"""


# =============================================================================
# MARKET-AWARE ρ ESTIMATION PROMPTS (Experimental)
# =============================================================================
#
# These prompts attempt to fix the "logical correlation ≠ market correlation"
# problem discovered in the LLM ρ calibration experiment (2026-01-26).
#
# Key insight: Markets show POSITIVE correlation between mutually exclusive
# outcomes (e.g., Warsh vs Waller for Fed chair) because both prices track
# a shared driver ("Will Trump nominate someone?"). The competitive effect
# is dominated by the shared driver.
#
# The baseline prompt encodes the wrong mental model - it asks "who competes
# with whom?" instead of "what moves both prices together?"
# =============================================================================

# --- Market Logic Warning (Option D) ---
# Add this to existing prompts to warn about market vs logical correlation

MARKET_LOGIC_WARNING = """

IMPORTANT: You are estimating MARKET correlation, not LOGICAL correlation.

Markets often show POSITIVE correlation between mutually exclusive outcomes when:
- Both depend on the same underlying event happening at all
- The same traders/attention affects both prices
- Example: "Trump nominates Warsh" and "Trump nominates Waller" are logically
  exclusive (only one can happen), but POSITIVELY correlated in markets because
  both go up when "Trump will nominate someone" becomes likely

Ask yourself: What shared factor would move both prices in the same direction?
If such a factor exists and is uncertain, expect POSITIVE correlation even
for competing outcomes."""


# --- Shared Drivers Prompt (Option A) ---
# Completely reframed to think about what moves prices together

RHO_SHARED_DRIVERS_PROMPT = """Question A: "{question_a}"
Question B: "{question_b}"

You are estimating how these prediction market prices move together.

STEP 1 - SHARED DRIVERS:
What underlying factors would cause BOTH prices to move in the SAME direction?
- Example: "Trump nominates Warsh" and "Trump nominates Waller" both go UP when
  "Trump will make a nomination" becomes more likely

STEP 2 - OPPOSING FORCES:
What factors would cause prices to move in OPPOSITE directions?
- Example: If A and B are mutually exclusive, A going up means B must go down

STEP 3 - WHICH DOMINATES?
In prediction markets, shared drivers often dominate logical exclusivity.
- If both questions depend on the same uncertain event happening at all,
  they typically move together (positive correlation)
- The competitive effect only dominates when the shared driver is already priced in

Which effect is stronger for this pair?

Respond with JSON only:
{{"shared_drivers": "<what would move both up or both down>", "opposing_forces": "<what would move them opposite>", "dominant_effect": "shared_drivers" or "opposing_forces" or "balanced", "direction": "more_likely" or "less_likely" or "no_effect", "reasoning": "<brief explanation>"}}"""


# --- Combined: Shared Drivers + Warning ---

RHO_SHARED_DRIVERS_WITH_WARNING_PROMPT = RHO_SHARED_DRIVERS_PROMPT + MARKET_LOGIC_WARNING


# --- Direction prompt with market warning ---

RHO_DIRECTION_WITH_WARNING_PROMPT = RHO_DIRECTION_PROMPT + MARKET_LOGIC_WARNING


RHO_MAGNITUDE_PROMPT = """Question A (target): "{question_a}"
Question B (signal): "{question_b}"

You've determined that if B=YES, then A becomes {direction}.

Now estimate the STRENGTH of this relationship on a scale from 0.0 to 1.0:
- 0.0: No relationship (independent)
- 0.1-0.3: Weak relationship (minor influence)
- 0.3-0.6: Moderate relationship (meaningful but not dominant factor)
- 0.6-0.9: Strong relationship (major factor in outcome)
- 1.0: Perfect relationship (one determines the other)

Respond with JSON only:
{{"magnitude": <float 0.0-1.0>, "reasoning": "<brief explanation>"}}"""


# =============================================================================
# MAGNITUDE CALIBRATION PROMPTS (Experimental - Phase 4)
# =============================================================================
#
# Phase 3 found that improved direction accuracy (55.9% vs 44.1%) did NOT
# translate to VOI improvement (r=-0.049 vs r=0.138). Root cause: the model
# anchors on high magnitudes (predicting |ρ|=0.8 when ground-truth is |ρ|=0.1).
#
# These prompts attempt to fix magnitude calibration while keeping direction
# estimation fixed (using shared_drivers_with_warning, which achieves 77% on
# mutually_exclusive pairs).
# =============================================================================

RHO_MAGNITUDE_CALIBRATED_PROMPT = """Question A (target): "{question_a}"
Question B (signal): "{question_b}"

You've determined that if B=YES, then A becomes {direction}.

Now estimate the STRENGTH of this MARKET correlation on a scale from 0.0 to 1.0:
- 0.0: No relationship (independent)
- 0.1-0.3: Weak relationship (minor influence)
- 0.3-0.6: Moderate relationship (meaningful but not dominant factor)
- 0.6-0.9: Strong relationship (major factor in outcome)
- 1.0: Perfect relationship (one determines the other)

IMPORTANT CALIBRATION:
- Most market correlations are WEAK (|ρ| < 0.3)
- Even mutually exclusive outcomes often have |ρ| < 0.2 because shared drivers dominate
- Only predict > 0.5 when there's DIRECT logical necessity (A literally requires B)
- Default to 0.1-0.3 unless you have strong evidence for more

Respond with JSON only:
{{"magnitude": <float 0.0-1.0>, "reasoning": "<brief explanation>"}}"""


RHO_MAGNITUDE_DISCOUNT_PROMPT = """Question A (target): "{question_a}"
Question B (signal): "{question_b}"

You've determined that if B=YES, then A becomes {direction}.

STEP 1: Estimate the LOGICAL strength of this relationship (0.0-1.0):
- How strong is the causal or logical connection between these outcomes?

STEP 2: Apply market discount
Market correlations are weaker than logical relationships because:
- Prices are noisy
- Shared drivers often dominate competitive effects
- Traders don't update fully

After estimating the logical strength, apply a 50% discount to get the market correlation.

Example: If logical strength is 0.8, market magnitude = 0.8 × 0.5 = 0.4

Respond with JSON only:
{{"logical_strength": <float 0.0-1.0>, "market_magnitude": <float 0.0-1.0>, "reasoning": "<brief explanation>"}}"""


RHO_MAGNITUDE_ANCHORED_PROMPT = """Question A (target): "{question_a}"
Question B (signal): "{question_b}"

You've determined that if B=YES, then A becomes {direction}.

Now estimate the STRENGTH of this MARKET correlation on a scale from 0.0 to 1.0.

Calibration examples (actual market data):
- "Trump nominates Warsh" vs "Trump nominates Waller" (mutually exclusive): |ρ| = 0.53
- "Sanders nomination" vs "Newsom nomination" (mutually exclusive): |ρ| = 0.03
- "Fed cut January" vs "Waller nomination" (causal): |ρ| = 0.22
- "Microsoft #1 market cap" vs "Apple #1 market cap" (competing): |ρ| = 0.58
- "Bitcoin dip $85k" vs "Bitcoin reach $115k" (mutually exclusive): |ρ| = 0.64

Key insight: Most relationships that sound "very strong" logically are actually 0.1-0.3 in markets.
The examples above are cherry-picked strong cases. Typical correlations are weaker.

Scale:
- 0.0: No relationship
- 0.1-0.3: Weak (most pairs)
- 0.3-0.5: Moderate (related events with shared drivers)
- 0.5-0.7: Strong (clear causal chain or same-event variants)
- 0.7+: Very strong (rare - direct prerequisites only)

Respond with JSON only:
{{"magnitude": <float 0.0-1.0>, "reasoning": "<brief explanation>"}}"""


# --- Single-step prompt (legacy, kept for backwards compatibility) ---

RHO_ESTIMATION_PROMPT = """You are estimating the correlation between two prediction market questions.

Question A: "{question_a}"
Question B: "{question_b}"

Estimate the correlation coefficient (ρ) between these two questions. This measures how much knowing the outcome of one question tells you about the other:
- ρ = +1: Perfect positive correlation (if A is YES, B is definitely YES)
- ρ = 0: Independent (knowing A tells you nothing about B)
- ρ = -1: Perfect negative correlation (if A is YES, B is definitely NO)

Think about:
- Are these questions about related events?
- Would one outcome make the other more or less likely?
- Are they measuring the same underlying phenomenon?

Respond with JSON only:
{{"rho": <float from -1 to +1>, "reasoning": "<brief explanation>"}}"""


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
        # Accept both "rho" and "rho_estimate" for compatibility
        rho = result.get("rho", result.get("rho_estimate", 0.0))
        return float(rho), result.get("reasoning", "")
    except Exception as e:
        return 0.0, f"Error: {e}"


async def estimate_rho_market_aware(
    question_a: str,
    question_b: str,
    model: str | None = None,
) -> tuple[float, str]:
    """Estimate correlation (rho) using market-aware prompting.

    Uses the shared_drivers_with_warning prompt for direction (77% accuracy on
    mutually_exclusive pairs) plus calibrated magnitude prompt (VOI r=0.355).

    Phase 4 experiment (2026-01-26) showed that magnitude calibration is critical:
    - Baseline magnitude: VOI r = -0.050 (overpredicts by +0.208)
    - Calibrated magnitude: VOI r = 0.355 (overpredicts by only +0.024)

    Best for pairs where market dynamics differ from logical reasoning,
    especially mutually exclusive outcomes (e.g., competing candidates).

    Args:
        question_a: The target/ultimate question
        question_b: The signal/crux question
        model: LLM model to use (defaults to haiku for cost efficiency)

    Returns:
        Tuple of (rho value, combined reasoning string)
    """
    import litellm

    model = model or DEFAULT_RHO_MODEL

    try:
        # Step 1: Get direction using market-aware prompt
        dir_response = await litellm.acompletion(
            model=model,
            messages=[{
                "role": "user",
                "content": RHO_SHARED_DRIVERS_WITH_WARNING_PROMPT.format(
                    question_a=question_a,
                    question_b=question_b,
                )
            }],
            max_tokens=500,
            temperature=0,
        )
        dir_text = dir_response.choices[0].message.content.strip()

        # Handle markdown code blocks
        if "```" in dir_text:
            dir_text = dir_text.split("```")[1]
            if dir_text.startswith("json"):
                dir_text = dir_text[4:]
            dir_text = dir_text.strip()

        dir_result = json.loads(dir_text)
        direction = dir_result.get("direction", "no_effect")
        dir_reasoning = dir_result.get("reasoning", "")
        shared_drivers = dir_result.get("shared_drivers", "")
        dominant = dir_result.get("dominant_effect", "")

        # Convert direction to sign
        if direction == "more_likely":
            sign = 1
            direction_word = "more likely"
        elif direction == "less_likely":
            sign = -1
            direction_word = "less likely"
        else:
            # Independent - no need for magnitude step
            return 0.0, f"Direction: {dir_reasoning} (shared drivers: {shared_drivers}, dominant: {dominant})"

        # Step 2: Get magnitude using calibrated prompt (Phase 4 experiment)
        # Calibrated prompt reduces magnitude overprediction from +0.208 to +0.024
        mag_response = await litellm.acompletion(
            model=model,
            messages=[{
                "role": "user",
                "content": RHO_MAGNITUDE_CALIBRATED_PROMPT.format(
                    question_a=question_a,
                    question_b=question_b,
                    direction=direction_word,
                )
            }],
            max_tokens=300,
            temperature=0,
        )
        mag_text = mag_response.choices[0].message.content.strip()

        # Handle markdown code blocks
        if "```" in mag_text:
            mag_text = mag_text.split("```")[1]
            if mag_text.startswith("json"):
                mag_text = mag_text[4:]
            mag_text = mag_text.strip()

        mag_result = json.loads(mag_text)
        magnitude = float(mag_result.get("magnitude", 0.3))
        mag_reasoning = mag_result.get("reasoning", "")

        # Combine sign and magnitude
        rho = sign * min(1.0, max(0.0, magnitude))
        combined_reasoning = f"Shared drivers: {shared_drivers} | Dominant: {dominant} | {dir_reasoning} | Magnitude: {mag_reasoning}"

        return rho, combined_reasoning

    except Exception as e:
        return 0.0, f"Error: {e}"


async def estimate_rho_two_step(
    question_a: str,
    question_b: str,
    model: str | None = None,
) -> tuple[float, str]:
    """Estimate correlation (rho) using two-step approach: direction then magnitude.

    This approach fixes sign errors common in single-step estimation,
    particularly for competition scenarios (rival candidates, competing films, etc.).

    Step 1: Ask "If B=YES, is A more or less likely?" → determines sign
    Step 2: Ask "How strong is this relationship?" → determines magnitude

    Args:
        question_a: The target/ultimate question
        question_b: The signal/crux question
        model: LLM model to use (defaults to haiku for cost efficiency)

    Returns:
        Tuple of (rho value, combined reasoning string)
    """
    import litellm

    model = model or DEFAULT_RHO_MODEL

    try:
        # Step 1: Get direction
        dir_response = await litellm.acompletion(
            model=model,
            messages=[{
                "role": "user",
                "content": RHO_DIRECTION_PROMPT.format(
                    question_a=question_a,
                    question_b=question_b,
                )
            }],
            max_tokens=300,
            temperature=0,
        )
        dir_text = dir_response.choices[0].message.content.strip()

        # Handle markdown code blocks
        if "```" in dir_text:
            dir_text = dir_text.split("```")[1]
            if dir_text.startswith("json"):
                dir_text = dir_text[4:]
            dir_text = dir_text.strip()

        dir_result = json.loads(dir_text)
        direction = dir_result.get("direction", "no_effect")
        dir_reasoning = dir_result.get("reasoning", "")

        # Convert direction to sign
        if direction == "more_likely":
            sign = 1
            direction_word = "more likely"
        elif direction == "less_likely":
            sign = -1
            direction_word = "less likely"
        else:
            # Independent - no need for magnitude step
            return 0.0, f"Direction: {dir_reasoning}"

        # Step 2: Get magnitude
        mag_response = await litellm.acompletion(
            model=model,
            messages=[{
                "role": "user",
                "content": RHO_MAGNITUDE_PROMPT.format(
                    question_a=question_a,
                    question_b=question_b,
                    direction=direction_word,
                )
            }],
            max_tokens=200,
            temperature=0,
        )
        mag_text = mag_response.choices[0].message.content.strip()

        # Handle markdown code blocks
        if "```" in mag_text:
            mag_text = mag_text.split("```")[1]
            if mag_text.startswith("json"):
                mag_text = mag_text[4:]
            mag_text = mag_text.strip()

        mag_result = json.loads(mag_text)
        magnitude = float(mag_result.get("magnitude", 0.3))
        mag_reasoning = mag_result.get("reasoning", "")

        # Combine sign and magnitude
        rho = sign * min(1.0, max(0.0, magnitude))
        combined_reasoning = f"Direction: {dir_reasoning} | Magnitude: {mag_reasoning}"

        return rho, combined_reasoning

    except Exception as e:
        return 0.0, f"Error: {e}"


async def estimate_rho_two_step_batch(
    pairs: list[tuple[str, str]],
    model: str | None = None,
    poll_interval: float = 5.0,
    max_wait: float = 3600.0,
) -> list[tuple[float, str]]:
    """Estimate rho for multiple pairs using two-step approach with batch API.

    Submits direction prompts as one batch, then magnitude prompts as another.
    This fixes sign errors common in single-step estimation.

    Args:
        pairs: List of (target_question, signal_question) tuples
        model: Model to use (defaults to haiku)
        poll_interval: Seconds between batch status checks
        max_wait: Maximum seconds to wait for each batch

    Returns:
        List of (rho, reasoning) tuples in same order as input pairs
    """
    import asyncio
    import anthropic

    if not pairs:
        return []

    model = model or DEFAULT_RHO_MODEL
    anthropic_model = model.replace("anthropic/", "")

    client = anthropic.Anthropic()

    # === STEP 1: Direction batch ===
    dir_requests = []
    for i, (question_a, question_b) in enumerate(pairs):
        dir_requests.append({
            "custom_id": f"dir_{i}",
            "params": {
                "model": anthropic_model,
                "max_tokens": 300,
                "messages": [{
                    "role": "user",
                    "content": RHO_DIRECTION_PROMPT.format(
                        question_a=question_a,
                        question_b=question_b,
                    )
                }],
            }
        })

    # Submit direction batch
    dir_batch = client.messages.batches.create(requests=dir_requests)
    dir_batch_id = dir_batch.id

    # Poll for direction batch completion
    elapsed = 0.0
    while elapsed < max_wait:
        batch_status = client.messages.batches.retrieve(dir_batch_id)
        if batch_status.processing_status == "ended":
            break
        await asyncio.sleep(poll_interval)
        elapsed += poll_interval

    if elapsed >= max_wait:
        raise TimeoutError(f"Direction batch {dir_batch_id} did not complete within {max_wait}s")

    # Collect direction results
    directions: dict[int, tuple[int, str, str]] = {}  # idx -> (sign, direction_word, reasoning)
    for result in client.messages.batches.results(dir_batch_id):
        idx = int(result.custom_id.replace("dir_", ""))
        if result.result.type == "succeeded":
            text = result.result.message.content[0].text.strip()
            try:
                if "```" in text:
                    text = text.split("```")[1]
                    if text.startswith("json"):
                        text = text[4:]
                    text = text.strip()
                parsed = json.loads(text)
                direction = parsed.get("direction", "no_effect")
                reasoning = parsed.get("reasoning", "")

                if direction == "more_likely":
                    directions[idx] = (1, "more likely", reasoning)
                elif direction == "less_likely":
                    directions[idx] = (-1, "less likely", reasoning)
                else:
                    directions[idx] = (0, "independent", reasoning)
            except Exception as e:
                directions[idx] = (0, "error", f"Parse error: {e}")
        else:
            directions[idx] = (0, "error", f"Batch error: {result.result.type}")

    # === STEP 2: Magnitude batch (only for non-zero directions) ===
    mag_requests = []
    mag_idx_map: dict[str, int] = {}  # custom_id -> original idx
    for i, (question_a, question_b) in enumerate(pairs):
        sign, direction_word, _ = directions.get(i, (0, "error", ""))
        if sign != 0:  # Only need magnitude for non-independent pairs
            custom_id = f"mag_{i}"
            mag_idx_map[custom_id] = i
            mag_requests.append({
                "custom_id": custom_id,
                "params": {
                    "model": anthropic_model,
                    "max_tokens": 200,
                    "messages": [{
                        "role": "user",
                        "content": RHO_MAGNITUDE_PROMPT.format(
                            question_a=question_a,
                            question_b=question_b,
                            direction=direction_word,
                        )
                    }],
                }
            })

    magnitudes: dict[int, tuple[float, str]] = {}  # idx -> (magnitude, reasoning)

    if mag_requests:
        # Submit magnitude batch
        mag_batch = client.messages.batches.create(requests=mag_requests)
        mag_batch_id = mag_batch.id

        # Poll for magnitude batch completion
        elapsed = 0.0
        while elapsed < max_wait:
            batch_status = client.messages.batches.retrieve(mag_batch_id)
            if batch_status.processing_status == "ended":
                break
            await asyncio.sleep(poll_interval)
            elapsed += poll_interval

        if elapsed >= max_wait:
            raise TimeoutError(f"Magnitude batch {mag_batch_id} did not complete within {max_wait}s")

        # Collect magnitude results
        for result in client.messages.batches.results(mag_batch_id):
            idx = mag_idx_map[result.custom_id]
            if result.result.type == "succeeded":
                text = result.result.message.content[0].text.strip()
                try:
                    if "```" in text:
                        text = text.split("```")[1]
                        if text.startswith("json"):
                            text = text[4:]
                        text = text.strip()
                    parsed = json.loads(text)
                    magnitude = float(parsed.get("magnitude", 0.3))
                    reasoning = parsed.get("reasoning", "")
                    magnitudes[idx] = (min(1.0, max(0.0, magnitude)), reasoning)
                except Exception as e:
                    magnitudes[idx] = (0.3, f"Parse error: {e}")
            else:
                magnitudes[idx] = (0.3, f"Batch error: {result.result.type}")

    # === Combine results ===
    results: list[tuple[float, str]] = []
    for i in range(len(pairs)):
        sign, direction_word, dir_reasoning = directions.get(i, (0, "error", "Missing"))

        if sign == 0:
            # Independent or error
            results.append((0.0, f"Direction: {dir_reasoning}"))
        else:
            magnitude, mag_reasoning = magnitudes.get(i, (0.3, "Default magnitude"))
            rho = sign * magnitude
            combined = f"Direction: {dir_reasoning} | Magnitude: {mag_reasoning}"
            results.append((rho, combined))

    return results


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
                # Accept both "rho" and "rho_estimate" for compatibility
                rho = parsed.get("rho", parsed.get("rho_estimate", 0.0))
                results_by_id[custom_id] = (float(rho), parsed.get("reasoning", ""))
            except Exception as e:
                results_by_id[custom_id] = (0.0, f"Parse error: {e}")
        else:
            results_by_id[custom_id] = (0.0, f"Batch error: {result.result.type}")

    # Return in original order
    return [results_by_id.get(f"rho_{i}", (0.0, "Missing result")) for i in range(len(pairs))]


# =============================================================================
# Conditional expectations for continuous targets
# =============================================================================

CONDITIONAL_EXPECTATION_PROMPT = """You are estimating how a signal affects a continuous forecast.

TARGET QUESTION: "{target}"
SIGNAL: "{signal}"

The signal is a YES/NO question that will resolve before the target question.

Estimate:
1. E[target | signal=YES] - Expected value of the target IF the signal resolves YES
2. E[target | signal=NO] - Expected value of the target IF the signal resolves NO

Think about:
- How does the signal causally relate to the target?
- If the signal fires (YES), what's your best estimate for the target?
- If the signal doesn't fire (NO), what's your best estimate?
- Consider the current context and trajectory

Respond with JSON only:
{{"e_given_yes": <number>, "e_given_no": <number>, "reasoning": "<brief explanation>"}}

IMPORTANT: Use the same units as the target question. For GDP questions in trillions, respond in trillions (e.g., 35.5 for $35.5T)."""


async def estimate_conditional_expectations_batch(
    pairs: list[tuple[str, str]],
    model: str | None = None,
    poll_interval: float = 5.0,
    max_wait: float = 3600.0,
) -> list[tuple[float | None, float | None, str]]:
    """Estimate E[target|signal=YES] and E[target|signal=NO] for continuous targets.

    Uses Anthropic batch API for efficiency. Each pair gets its own prompt.

    Args:
        pairs: List of (target_question, signal_question) tuples
        model: Model to use (defaults to haiku)
        poll_interval: Seconds between batch status checks
        max_wait: Maximum seconds to wait for batch completion

    Returns:
        List of (e_given_yes, e_given_no, reasoning) tuples in same order as input
    """
    import asyncio
    import anthropic

    if not pairs:
        return []

    model = model or DEFAULT_RHO_MODEL
    anthropic_model = model.replace("anthropic/", "")

    client = anthropic.Anthropic()

    # Build batch requests
    requests = []
    for i, (target, signal) in enumerate(pairs):
        requests.append({
            "custom_id": f"cond_exp_{i}",
            "params": {
                "model": anthropic_model,
                "max_tokens": 300,
                "messages": [{
                    "role": "user",
                    "content": CONDITIONAL_EXPECTATION_PROMPT.format(
                        target=target,
                        signal=signal,
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
    results_by_id: dict[str, tuple[float | None, float | None, str]] = {}
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
                e_yes = parsed.get("e_given_yes")
                e_no = parsed.get("e_given_no")
                # Convert to float if present
                e_yes = float(e_yes) if e_yes is not None else None
                e_no = float(e_no) if e_no is not None else None
                results_by_id[custom_id] = (e_yes, e_no, parsed.get("reasoning", ""))
            except Exception as e:
                results_by_id[custom_id] = (None, None, f"Parse error: {e}")
        else:
            results_by_id[custom_id] = (None, None, f"Batch error: {result.result.type}")

    # Return in original order
    return [results_by_id.get(f"cond_exp_{i}", (None, None, "Missing result")) for i in range(len(pairs))]
