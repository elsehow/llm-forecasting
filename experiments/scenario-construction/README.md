# Scenario Construction

Generate MECE (Mutually Exclusive, Collectively Exhaustive) scenarios for forecasting questions.

## Quick Start

```bash
# Run dual approach (recommended) for a target
uv run python experiments/scenario-construction/approach_dual_v7.py --target us_gdp_2029

# Or run individual approaches
uv run python experiments/scenario-construction/approach_bottomup_v7.py --target carney_pm_2027
uv run python experiments/scenario-construction/approach_topdown_v7.py --target democrat_whitehouse_2028
uv run python experiments/scenario-construction/approach_hybrid_v7.py --target us_gdp_2029
```

## Available Targets

| Target | Question | Type | Horizon |
|--------|----------|------|---------|
| `carney_pm_2027` | Will Mark Carney be PM of Canada on Dec 31, 2027? | Binary | 2 years |
| `democrat_whitehouse_2028` | Will a Democrat win the White House in 2028? | Binary | 3 years |
| `us_gdp_2029` | What will US real GDP be in 2029 (2024 dollars)? | Continuous | 4 years |

## Four Approaches

| Approach | Method | Signal Count (avg) |
|----------|--------|--------------------|
| **Dual** (recommended) | Independent top-down + bottom-up, then merge with gap detection | **35** |
| Top-down | LLM identifies uncertainties → generates signals | 24 |
| Hybrid | LLM uncertainties scope market search | 6 |
| Bottom-up | Semantic search markets → rank by VOI | 4 |

**Winner: Dual** produces 2-9x more cruxy signals (VOI ≥ 0.1) than alternatives.

## Results (2026-01-16)

| Target | Bottomup | Topdown | Hybrid | Dual | Dual Gaps |
|--------|----------|---------|--------|------|-----------|
| Carney PM 2027 | 1 | 21 | 6 | **17** | 29 |
| Democrat Whitehouse 2028 | 6 | 28 | 6 | **44** | 17 |
| US GDP 2029 | 5 | 23 | 6 | **44** | 26 |

## Adding New Targets

Edit `shared/config.py`:

```python
NEW_TARGET = TargetConfig(
    question=Question(
        id="my_target",
        source="scenario_construction",
        text="What will X be in 2030?",
        question_type=QuestionType.BINARY,  # or CONTINUOUS
        value_range=None,
        base_rate=0.5,
        unit=None,
    ),
    context="Current state and key uncertainties...",
    cruxiness_normalizer=1.0,  # For binary, use 1.0
)

TARGETS["my_target"] = NEW_TARGET
```

## Architecture

```
shared/
├── config.py       # TargetConfig, TARGETS registry
├── signals.py      # VOI ranking, deduplication
├── scenarios.py    # MECE generation
├── uncertainties.py # Uncertainty identification
└── output.py       # Results saving

approach_*_v7.py    # Four approach implementations
results/            # JSON outputs by target
```
