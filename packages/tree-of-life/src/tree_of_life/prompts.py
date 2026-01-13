"""Prompt templates for each pipeline phase."""

DIVERGE_SYSTEM = """You are an expert forecaster generating scenarios for conditional forecasting.
Your task is to generate distinct, plausible scenarios that would meaningfully affect the outcome of a forecasting question.

{base_rate_context}

Guidelines:
- Today's date is {start_date}. Do NOT include events that have already occurred.
- Generate scenarios that unfold between {start_date} and {forecast_horizon}
- Generate scenarios that are mutually distinguishable (not overlapping)
- Each scenario should bundle multiple causal factors/assumptions
- Scenarios should be concrete enough to imagine but broad enough to be useful
- Include both optimistic and pessimistic scenarios
- Consider tail risks and surprising-but-plausible outcomes"""

DIVERGE_USER = """Generate {n_scenarios} distinct scenarios relevant to this forecasting question:

Question: {question_text}
Type: {question_type}
Domain: {domain}

For each scenario, provide:
1. A short, memorable name (2-4 words)
2. A description (2-3 sentences)
3. Key assumptions that define this scenario (3-5 bullet points)

Respond in JSON format:
{{
  "scenarios": [
    {{
      "name": "Scenario Name",
      "description": "Description of the scenario...",
      "key_assumptions": ["assumption 1", "assumption 2", "assumption 3"]
    }}
  ]
}}"""

CONVERGE_SYSTEM = """You are an expert at synthesizing and clustering scenarios.
Your task is to identify common themes across scenarios generated for different questions and consolidate them into global scenarios that apply across all questions."""

CONVERGE_USER = """Below are scenarios generated for multiple forecasting questions.
Consolidate these into {max_scenarios} global scenarios that capture the major "world states" implied across all questions.

Raw scenarios:
{scenarios_json}

For each global scenario:
1. Create a short, unique ID (snake_case, e.g., "ai_transformation", "climate_crisis")
2. Give it a memorable name
3. Write a description that captures the world state
4. List the key drivers/assumptions
5. List which original scenarios map to this global scenario

Respond in JSON format:
{{
  "global_scenarios": [
    {{
      "id": "ai_transformation",
      "name": "AI Transformation Era",
      "description": "Description of this world state...",
      "key_drivers": ["driver 1", "driver 2"],
      "member_scenarios": ["original scenario name 1", "original scenario name 2"]
    }}
  ]
}}"""

STRUCTURE_SYSTEM = """You are an expert at analyzing relationships between scenarios.
Your task is to identify how pairs of scenarios relate to each other.

IMPORTANT: Most scenario pairs should be ORTHOGONAL (independent). Only use "correlated" when
there is a clear, direct causal mechanism linking the scenarios. Vague thematic similarity
is NOT sufficient for correlation."""

STRUCTURE_USER = """Analyze the relationships between these global scenarios:

{scenarios_json}

For each pair of scenarios, classify their relationship as one of:
- "orthogonal": DEFAULT. The scenarios are independent; one doesn't directly cause or prevent the other.
  Use this unless there's a clear causal link. Thematic similarity alone is NOT correlation.
- "correlated": RARE. One scenario directly causes or prevents the other through a specific mechanism.
  Requires a clear causal chain, not just "both involve technology" or "both are bad outcomes."
- "hierarchical": One scenario is a strict subset or more specific version of another.
- "mutually_exclusive": The scenarios logically cannot both occur (e.g., "war" vs "peace").

Guidelines:
- Default to "orthogonal" when uncertain
- "correlated" requires you to articulate the specific causal mechanism
- Expect roughly 60-80% of pairs to be orthogonal in a well-constructed scenario set
- Negative correlation (one prevents the other) should be rare

For correlated relationships, estimate strength from -1 (perfect negative) to 1 (perfect positive).

Respond in JSON format:
{{
  "relationships": [
    {{
      "scenario_a": "scenario_id_1",
      "scenario_b": "scenario_id_2",
      "type": "orthogonal|correlated|hierarchical|mutually_exclusive",
      "strength": 0.5,
      "notes": "Brief explanation of causal mechanism if correlated"
    }}
  ]
}}"""

QUANTIFY_SYSTEM = """You are an expert forecaster assigning probabilities to scenarios.
Your task is to assign coherent probabilities that respect logical constraints.

CRITICAL: These scenarios represent MUTUALLY EXCLUSIVE world states. Exactly one will occur.
Your probabilities MUST sum to exactly 1.0 (using decimal notation, not percentages).

Before responding, verify your probabilities sum to 1.0."""

QUANTIFY_USER = """Assign probabilities to these global scenarios:

Scenarios:
{scenarios_json}

Relationships between scenarios:
{relationships_json}

Constraints:
- Probabilities should reflect your best estimate of each scenario occurring
- Use DECIMAL notation (0.25 means 25%), NOT percentages
- All probabilities must sum to exactly 1.0
- Account for the possibility that reality might not fit any scenario perfectly

For each scenario, provide a probability and brief reasoning.

Respond in JSON format:
{{
  "probabilities": [
    {{
      "scenario_id": "id",
      "probability": 0.25,
      "reasoning": "Brief explanation"
    }}
  ]
}}

IMPORTANT: Use decimals (0.15, 0.20, 0.08) not percentages (15, 20, 8). Sum must equal 1.0."""

CONDITION_SYSTEM = """You are an expert forecaster making conditional predictions.
Your task is to forecast outcomes conditional on specific scenarios occurring."""

CONDITION_CONTINUOUS_USER = """Given this scenario occurs, forecast the outcome for this question:

Scenario: {scenario_name}
Description: {scenario_description}

Question: {question_text}
Resolution: {resolution_source}

{base_rate_context}

Provide:
- Median estimate
- 80% confidence interval (10th and 90th percentiles)
- Brief reasoning

Consider the baseline trajectory before scenario adjustment.

Respond in JSON format:
{{
  "median": 25000000000000,
  "ci_80_low": 20000000000000,
  "ci_80_high": 35000000000000,
  "reasoning": "Brief explanation"
}}"""

CONDITION_CATEGORICAL_USER = """Given this scenario occurs, forecast the outcome for this question:

Scenario: {scenario_name}
Description: {scenario_description}

Question: {question_text}
Options: {options}

Provide probability for each option (must sum to 1.0).

Respond in JSON format:
{{
  "probabilities": {{
    "Option A": 0.6,
    "Option B": 0.3,
    "Option C": 0.1
  }},
  "reasoning": "Brief explanation"
}}"""

CONDITION_BINARY_USER = """Given this scenario occurs, forecast the probability for this question:

Scenario: {scenario_name}
Description: {scenario_description}

Question: {question_text}

Provide:
- Probability (0-1)
- Brief reasoning

Respond in JSON format:
{{
  "probability": 0.35,
  "reasoning": "Brief explanation"
}}"""

# Batched condition prompts - all scenarios for one question
# Uses Bracket-style direction commitment to ensure logical coherence
CONDITION_BATCH_SYSTEM = """You are an expert forecaster making conditional predictions.
Your task is to forecast outcomes for a question under multiple scenarios.

IMPORTANT: You must ensure logical coherence across scenarios.

Step 1: DIRECTION COMMITMENT
Before giving probabilities, explicitly state the expected ordering.
- Which scenarios should produce HIGHER probabilities for this question?
- Which scenarios should produce LOWER probabilities?
- Which scenarios are roughly NEUTRAL (similar to baseline)?

Step 2: PROBABILITY ASSIGNMENT
Your probabilities MUST respect the ordering you declared in Step 1.
If you said Scenario A should be higher than Scenario B, then P(outcome|A) > P(outcome|B).

This constraint eliminates impossible forecasts where probabilities contradict the stated logic."""

CONDITION_BATCH_CONTINUOUS_USER = """Forecast the outcome for this question under each scenario:

Question: {question_text}
Resolution: {resolution_source}

{base_rate_context}

Scenarios:
{scenarios_json}

STEP 1: DIRECTION COMMITMENT
First, classify each scenario's effect on this question:
- "increases": This scenario leads to a HIGHER median value
- "decreases": This scenario leads to a LOWER median value
- "neutral": This scenario has no meaningful effect on the outcome

STEP 2: FORECASTS
For EACH scenario, provide:
- Median estimate
- 80% confidence interval (10th and 90th percentiles)
- Brief reasoning

CONSTRAINT: Your medians MUST respect Step 1:
- All "increases" scenarios must have higher median than all "neutral" scenarios
- All "decreases" scenarios must have lower median than all "neutral" scenarios

IMPORTANT: Use the EXACT "id" from each scenario as the key in your response.

Respond in JSON format:
{{
  "directions": {{
    "<scenario_id>": "increases|decreases|neutral"
  }},
  "forecasts": {{
    "<scenario_id>": {{
      "median": 25000000000000,
      "ci_80_low": 20000000000000,
      "ci_80_high": 35000000000000,
      "reasoning": "Brief explanation"
    }}
  }}
}}"""

CONDITION_BATCH_CATEGORICAL_USER = """Forecast the outcome for this question under each scenario:

Question: {question_text}
Options: {options}

Scenarios:
{scenarios_json}

STEP 1: DIRECTION COMMITMENT
First, classify each scenario's effect on this question.
For each scenario, identify which option it shifts probability TOWARD:
- Name a specific option if the scenario clearly favors it
- "neutral" if the scenario has no meaningful effect

STEP 2: PROBABILITIES
For EACH scenario, provide probability for each option (must sum to 1.0).

CONSTRAINT: Your probabilities MUST respect Step 1:
- If you said a scenario shifts toward Option X, that scenario should have
  higher P(Option X) than neutral scenarios

IMPORTANT: Use the EXACT "id" from each scenario as the key in your response.

Respond in JSON format:
{{
  "directions": {{
    "<scenario_id>": "<option_name>|neutral"
  }},
  "forecasts": {{
    "<scenario_id>": {{
      "probabilities": {{"Option A": 0.6, "Option B": 0.3, "Option C": 0.1}},
      "reasoning": "Brief explanation"
    }}
  }}
}}"""

CONDITION_BATCH_BINARY_USER = """Forecast the probability for this question under each scenario:

Question: {question_text}

Scenarios:
{scenarios_json}

STEP 1: DIRECTION COMMITMENT
First, classify each scenario's effect on this question:
- "increases": This scenario makes the outcome MORE likely than baseline
- "decreases": This scenario makes the outcome LESS likely than baseline
- "neutral": This scenario has no meaningful effect on this outcome

STEP 2: PROBABILITIES
For EACH scenario, provide:
- Probability (0-1)
- Brief reasoning

CONSTRAINT: Your probabilities MUST respect Step 1:
- All "increases" scenarios must have higher probability than all "neutral" scenarios
- All "decreases" scenarios must have lower probability than all "neutral" scenarios

IMPORTANT: Use the EXACT "id" from each scenario as the key in your response.

Respond in JSON format:
{{
  "directions": {{
    "<scenario_id>": "increases|decreases|neutral"
  }},
  "forecasts": {{
    "<scenario_id>": {{
      "probability": 0.35,
      "reasoning": "Brief explanation"
    }}
  }}
}}"""

SIGNALS_SYSTEM = """You are an expert at identifying early warning signals for scenarios.
Your task is to identify observable events that would update our beliefs about scenario probabilities."""

SIGNALS_USER = """Identify 3-5 early signals for this scenario:

Scenario: {scenario_name}
Description: {scenario_description}
Current probability: {probability}

IMPORTANT: Include at least 1-2 signals where direction is "decreases" (events that
would make this scenario LESS likely). A robust signal set covers both confirmation
AND disconfirmation — we need to detect worlds departing, not just arriving.

For each signal:
1. Describe a specific, observable event
2. When it should resolve (before {horizon_date})
3. Whether observing it would increase or decrease scenario probability
4. The magnitude of the update (small/medium/large)
5. Your estimate of the signal's current probability
6. Update cadence: how often should we check for this signal?
   - "event": check when specific events occur (news, announcements)
   - "monthly": check monthly (economic data, trade figures)
   - "quarterly": check quarterly (earnings, GDP releases)
   - "annual": check annually (demographic data, institutional reports)
7. Causal priority (0-100): lower = resolves earlier and informs other uncertainties.
   Signals about upstream causes (e.g., policy decisions) should have lower priority than
   signals about downstream effects (e.g., economic outcomes).

Respond in JSON format:
{{
  "signals": [
    {{
      "text": "Specific observable event description",
      "resolves_by": "2026-06-01",
      "direction": "increases|decreases",
      "magnitude": "small|medium|large",
      "current_probability": 0.3,
      "update_cadence": "quarterly",
      "causal_priority": 30
    }}
  ]
}}"""

QUANTIFY_RETRY = """Your previous response had probabilities summing to {sum:.2f} (should be 1.0).

These scenarios are MUTUALLY EXCLUSIVE world states — exactly one will occur.
Probabilities must sum to exactly 1.0 using DECIMAL notation.

Previous assignment:
{previous}

Please reassign probabilities that sum to 1.0. Use DECIMALS (0.15, 0.20) not percentages (15, 20).
Maintain the relative ordering where it makes sense, but adjust values so they sum to 1.0.

Respond in JSON format:
{{
  "probabilities": [
    {{
      "scenario_id": "id",
      "probability": 0.25,
      "reasoning": "Brief explanation"
    }}
  ]
}}"""

CONDITION_RETRY_DIRECTION = """Your probabilities violated your stated direction commitments:

Violations:
{violations}

Your directions said:
{directions}

But your probabilities were:
{probabilities}

Please revise your probabilities to respect your direction commitments.
If you need to change directions, you may, but then probabilities must match.

Respond in the same JSON format."""
