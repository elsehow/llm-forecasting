# Tree of Life

Automated scenario-based conditional forecasting using LLMs. Takes a set of forecasting questions and produces:

1. **Global scenarios** - Coherent world states shared across all questions
2. **Conditional forecasts** - P(outcome | scenario) for each question
3. **Early signals** - Observable events that update scenario probabilities

## How It Works

The pipeline runs in 7 phases:

```
Questions → [0. Base Rates] → Current reference values (via web search)
                 ↓
            [1. Diverge] → Raw Scenarios (5 per question)
                 ↓
            [2. Converge] → Global Scenarios (10 clustered)
                 ↓
            [3. Structure] → Relationship Matrix + Upstream Dependencies
                 ↓
            [4. Quantify] → P(scenario) with normalization
                 ↓
            [5. Condition] → P(outcome | scenario) for each question
                 ↓
            [6. Signals] → Early warning signals per scenario
```

**Key design choices:**
- Phase 0 uses Anthropic's web search tool to fetch current base rates dynamically
- Phase 3 derives upstream causal dependencies from relationship types
- Phase 5 batches all scenarios together per question for cross-scenario coherence
- Phase 4 uses tiered probability normalization with retry logic
- Phases 1-6 use Anthropic's batch API by default (50% cost savings)

## Installation

Requires [uv](https://docs.astral.sh/uv/) for package management.

```bash
# Clone and setup
cd tree-of-life
uv sync

# Set API key
echo "ANTHROPIC_API_KEY=your-key-here" > .env
```

## Quick Start

```bash
# Run full pipeline
uv run python run.py

# Run with custom questions
uv run python run.py -q path/to/questions.json -o output/my_tree.json
```

## CLI Options

```
uv run python run.py [OPTIONS]

Options:
  -q, --questions PATH     Input questions JSON file (default: examples/fri_questions.json)
  -o, --output PATH        Output tree JSON file (default: output/forecast_tree.json)
  --start-date DATE        Reference date for "today" (default: actual today)
  --forecast-horizon DATE  End of forecast window (default: 2040-12-31)
  --no-base-rates          Skip Phase 0 base rate fetching
  --diagnostics            Run diagnostics on existing tree instead of generating
  --from-phase N           Resume from phase N (0-6), requires --input
  -i, --input PATH         Input tree file for --from-phase
  --no-timestamp           Don't add timestamp to output (overwrites existing)
```

### Examples

```bash
# Full pipeline run (includes web search for base rates)
uv run python run.py

# Skip base rate fetching (faster, no web search)
uv run python run.py --no-base-rates

# Re-run just phases 5-6 (condition + signals) on existing tree
uv run python run.py --from-phase 5 --input output/forecast_tree.json

# Re-fetch base rates and re-run entire pipeline
uv run python run.py --from-phase 0 --input output/forecast_tree.json

# Run diagnostics on existing tree
uv run python run.py --diagnostics

# Custom date range
uv run python run.py --start-date 2026-01-01 --forecast-horizon 2035-12-31

# Overwrite same file (no timestamp)
uv run python run.py --no-timestamp
```

## Input Format

Questions JSON file:

```json
[
  {
    "id": "us_gdp_2040",
    "text": "US real GDP in 2040 (2024 dollars)",
    "type": "continuous",
    "resolution_source": "BEA estimate",
    "domain": "Macro"
  },
  {
    "id": "taiwan_status_2036",
    "text": "Taiwan governance status on Jan 1, 2036",
    "type": "categorical",
    "options": ["Self-governing", "PRC-administered", "Under blockade", "Active conflict"],
    "domain": "Geopolitics"
  },
  {
    "id": "ai_drugs_2030",
    "text": "Will an AI-designed drug complete Phase III trials by 2030?",
    "type": "binary",
    "domain": "AI/Biotech"
  }
]
```

Question types:
- **continuous**: Numeric forecast with median + 80% CI
- **categorical**: Probability distribution over options
- **binary**: Single probability

## Output Format

The output `forecast_tree.json` contains:

```json
{
  "questions": [...],
  "raw_scenarios": [...],
  "global_scenarios": [
    {
      "id": "ai_productivity_revolution",
      "name": "AI Productivity Revolution",
      "description": "...",
      "probability": 0.20,
      "key_drivers": ["..."],
      "member_scenarios": ["..."],
      "upstream_scenarios": ["tech_progress_accelerates"]
    }
  ],
  "relationships": [
    {
      "scenario_a": "ai_productivity_revolution",
      "scenario_b": "economic_stagnation",
      "type": "mutually_exclusive",
      "strength": 0.9,
      "notes": "These scenarios are fundamentally incompatible"
    }
  ],
  "conditionals": [
    {
      "question_id": "us_gdp_2040",
      "scenario_id": "ai_productivity_revolution",
      "median": 52000000000000,
      "ci_80_low": 44000000000000,
      "ci_80_high": 62000000000000,
      "reasoning": "..."
    }
  ],
  "signals": [
    {
      "id": "ai_revolution_signal_1",
      "text": "An AI system achieves superhuman performance on...",
      "resolves_by": "2027-06-01",
      "scenario_id": "ai_productivity_revolution",
      "direction": "increases",
      "magnitude": "large",
      "current_probability": 0.3,
      "update_cadence": "event",
      "causal_priority": 85
    }
  ],
  "created_at": "2026-01-06T13:21:10.300463",
  "raw_probability_sum": 1.0,
  "probability_status": "ok",
  "base_rates_snapshot": {
    "us_gdp": {
      "value": 29.0,
      "unit": "trillion_usd",
      "as_of": "2025-09-30",
      "source": "BEA via web search"
    }
  }
}
```

### Key Fields

- **upstream_scenarios**: Scenarios that causally influence this one (derived from relationships)
- **update_cadence**: How often to check this signal (`"daily"`, `"weekly"`, `"monthly"`, `"quarterly"`, `"event"`)
- **causal_priority**: 0-100 score indicating how upstream/leading this signal is (higher = check first)

## Diagnostics

Run diagnostics on an existing tree:

```bash
uv run python run.py --diagnostics
```

Output:
```
=== Tree Diagnostics ===
Created: 2026-01-06T13:21:10.300463

Probability sum (raw): 100%
Probability status: ok

Questions: 10
Global scenarios: 10
Conditionals: 100
Signals: 50

Base rates: 5 fetched
  - us_gdp: 29.0 trillion_usd (as of 2025-09-30)
  - us_10y_yield: 4.17 percent (as of 2026-01-05)
  ...

Past-dated signals: 0
Scenarios without signals: 0
```

## Configuration

Environment variables (or `.env` file):

| Variable | Default | Description |
|----------|---------|-------------|
| `ANTHROPIC_API_KEY` | (required) | Anthropic API key |
| `MODEL` | `claude-sonnet-4-20250514` | Model to use |
| `USE_BATCH_API` | `true` | Use batch API (50% cost savings, slower) |
| `N_SCENARIOS_PER_QUESTION` | `5` | Raw scenarios per question in phase 1 |
| `MAX_GLOBAL_SCENARIOS` | `10` | Target global scenarios in phase 2 |
| `START_DATE` | today | Reference date for scenario generation |
| `FORECAST_HORIZON` | `2040-12-31` | End of forecast window |
| `LOG_LEVEL` | `INFO` | Logging level |
| `LOG_FILE` | `output/pipeline.log` | Log file path |

## Base Rates

Phase 0 dynamically fetches current reference values to anchor forecasts. The LLM:

1. Analyzes the input questions to determine what base rates are needed
2. Uses Anthropic's web search tool to find authoritative current values
3. Returns structured data with values, sources, and dates

Example base rates fetched for economic/geopolitical questions:
- US Real GDP (BEA)
- US 10-Year Treasury Yield (Treasury)
- China GDP estimates
- Global working-age population (UN)
- Freedom House country scores

The fetched base rates are stored in `base_rates_snapshot` in the output tree for reproducibility. Use `--no-base-rates` to skip this phase (faster, but forecasts won't have current anchoring data).

## Development

```bash
# Install with dev dependencies
uv sync --all-extras

# Run tests
uv run pytest tests/ -v

# Run specific test
uv run pytest tests/test_probability.py -v

# Check test coverage
uv run pytest tests/ --cov=src
```

## Architecture

```
src/
├── config.py          # Configuration and logging setup
├── models.py          # Pydantic models (Question, Scenario, Signal, etc.)
├── prompts.py         # LLM prompt templates
├── llm.py             # LLM utilities + batch API
├── probability.py     # Probability normalization
├── propagation.py     # Causal graph operations (upstream derivation, propagation)
├── pipeline.py        # Main orchestrator
└── phases/
    ├── base_rates.py  # Phase 0: Dynamic base rate fetching via web search
    ├── diverge.py     # Phase 1: Generate scenarios
    ├── converge.py    # Phase 2: Cluster scenarios
    ├── structure.py   # Phase 3: Find relationships
    ├── quantify.py    # Phase 4: Assign probabilities
    ├── condition.py   # Phase 5: Conditional forecasts
    └── signals.py     # Phase 6: Early signals
```

### VOI Analysis

The `voi.py` module provides Value of Information calculations for ranking signals:

```python
from tree_of_life.pipeline import load_tree

tree = load_tree('output/forecast_tree.json')

# Rank signals by Linear VOI (default, more stable for rare events)
for signal, voi in tree.top_voi_signals(n=5):
    print(f"{signal.text}: VOI = {voi:.4f}")

# Compare linear vs entropy VOI methods
from tree_of_life import compare_voi_methods
for r in compare_voi_methods(tree):
    print(f"{r['signal_text']}: linear={r['linear_voi']:.3f}, entropy={r['entropy_voi']:.3f}")
```

Linear VOI uses expected absolute belief shift instead of entropy-based information gain. It's more stable under magnitude noise, especially for rare events (P < 0.10).

### Causal Graph

The `propagation.py` module provides utilities for working with the scenario dependency graph:

- `derive_upstream_graph()` - Computes upstream dependencies from relationships
- `topological_sort()` - Orders scenarios so upstreams come before downstreams
- `propagate_update()` - Updates scenario probabilities and propagates to dependents
- `compute_unconditional()` - Weighted sum over scenarios for unconditional forecasts

## License

MIT
