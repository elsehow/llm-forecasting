# Scenario Construction

Generate MECE (Mutually Exclusive, Collectively Exhaustive) scenarios for long-horizon forecasting questions.

## Quick Start

```bash
# Run all 3 approaches in parallel for a target
uv run python experiments/scenario-construction/gdp_2040/run_experiment.py --target renewable_2050

# Or run individually
uv run python experiments/scenario-construction/gdp_2040/approach_hybrid_v4.py --target gdp_2040
uv run python experiments/scenario-construction/gdp_2040/approach_topdown_v4.py --target gdp_2040
uv run python experiments/scenario-construction/gdp_2040/approach_bottomup_v4.py --target gdp_2040

# Evaluate results
uv run python experiments/scenario-construction/gdp_2040/evaluate.py --target gdp_2040
```

## Available Targets

| Target | Question | Unit | Normalizer |
|--------|----------|------|------------|
| `gdp_2040` | US real GDP in 2040 (2024 dollars) | $T | 20.0 |
| `renewable_2050` | Global energy share from renewables in 2050 | % | 30.0 |

## Three Approaches

| Approach | Method | Strengths |
|----------|--------|-----------|
| **Top-down** | LLM identifies 2 key uncertainties → scenarios from combinations | High signal coverage, clean MECE structure |
| **Bottom-up** | Semantic search markets → score relevance → cluster into scenarios | Grounded in real market questions |
| **Hybrid** | LLM brainstorms + market search → dedupe with observed preference | Best of both - creative + grounded |

### Top-down (`approach_topdown_v4.py`)
LLM identifies 2 key uncertainties that most differentiate outcomes, then generates scenarios from uncertainty combinations. Traditional scenario planning (Steve Weber style).

### Bottom-up (`approach_bottomup_v4.py`)
Semantic search finds relevant signals from prediction markets (Polymarket, Metaculus, etc.), then clusters into scenarios. Coverage depends on what markets exist for the topic.

### Hybrid (`approach_hybrid_v4.py`)
LLM brainstorms signals + semantic search for market signals, deduplicates with preference for observed signals, then generates scenarios.

## Evaluation Metrics

### Cruxiness (0-1)
Do scenarios produce different outcome forecasts?

- LLM forecaster estimates outcome conditional on each scenario
- Score = spread / normalizer (capped at 1.0)
- Example: 40pp spread in renewable forecasts / 30pp normalizer = 1.0

### Evaluability (0-1)
Are indicators specific enough to measure?

Each indicator scored 0-3:
- Has numeric threshold? (+1)
- Has timeframe? (+1)
- Has implied data source? (+1)

Score = average / 3

### Signal Coverage (0-1)
Are signals trackable and early-resolving?

Each signal scored 0-3:
- Resolves before 2030? (+1)
- Clear resolution criteria? (+1)
- Trackable source? (+1)

Score = average / 3

## Adding New Targets

Edit `shared/config.py`:

```python
from llm_forecasting.models import Question, QuestionType, Unit

NEW_TARGET = TargetConfig(
    question=Question(
        id="my_target",
        source="scenario_construction",
        text="What will X be in 2050?",
        question_type=QuestionType.CONTINUOUS,
        value_range=(10.0, 100.0),  # Plausible range
        base_rate=50.0,              # Current value
        unit=Unit.from_type("percent"),  # or "usd_trillions", etc.
    ),
    context="Current X: 50%. Key uncertainties: A, B, C.",
    cruxiness_normalizer=30.0,  # Spread that equals max cruxiness
)

# Add to registry
TARGETS = {
    "gdp_2040": GDP_2040,
    "renewable_2050": RENEWABLE_2050,
    "my_target": NEW_TARGET,  # <-- add here
}
```

## Results Structure

```
results/
├── gdp_2040/
│   ├── hybrid_v4_20260115_143022.json
│   ├── topdown_v4_20260115_143025.json
│   ├── bottomup_v4_20260115_143028.json
│   └── evaluation_20260115_143500.json
├── renewable_2050/
│   └── ...
```

Each result file contains:
```json
{
  "target": "renewable_2050",
  "approach": "hybrid_v4_semantic_search",
  "question": { "id": "...", "text": "...", "unit": "percent", ... },
  "config": { "cruxiness_normalizer": 30.0, "context": "..." },
  "scenarios": [ ... ],
  "signals": [ ... ],
  "created_at": "2026-01-15T15:01:22"
}
```

## Architecture

```
shared/
├── __init__.py
├── config.py      # TargetConfig, TARGETS registry
└── signals.py     # Semantic search, deduplication utilities

gdp_2040/
├── approach_hybrid_v4.py
├── approach_topdown_v4.py
├── approach_bottomup_v4.py
├── evaluate.py
├── run_experiment.py   # Parallel runner
└── results/
    ├── gdp_2040/
    └── renewable_2050/
```

## Dependencies

- `litellm` - LLM API calls
- `pydantic` - Structured outputs
- `sentence-transformers` - Semantic search embeddings
- Database: `data/forecastbench.db` with prediction market questions

## Relation to Other Projects

**Dynamic Middle Layer:**
- Bottom-up results inform DML viability
- If market clusters are interpretable → DML architecture works
