# LLM Forecasting v2 Architecture

## Goals

1. **Fewer lines of code**: ~3,700 LOC (down from ~17,500) - 78% reduction
2. **More extensible**: Plugin-based sources, config-driven models
3. **Better tested**: 79 unit tests + 20 integration tests
4. **Better data collection**: SQLite + queryable historical data

## Current Pain Points (Original Codebase)

| Area | Current State | Impact |
|------|--------------|--------|
| Question sources | 3,859 LOC, 95% copy-paste boilerplate | Hard to add new sources |
| LLM integration | 1,025 LOC across 6 provider-specific functions | Breaks with each new model |
| Testing | 0% coverage | Risky deployments |
| Data storage | JSONL files in GCS, no queryability | Can't analyze historical data |
| Configuration | Scattered across 5+ files, manual dicts | Error-prone, no validation |

## Key Design Decisions

### 1. LiteLLM with Structured Outputs

Single unified interface with Pydantic response schemas:

```python
from pydantic import BaseModel, Field

class BinaryForecastResponse(BaseModel):
    probability: float = Field(ge=0.0, le=1.0)
    reasoning: str

response = await litellm.acompletion(
    model="openai/gpt-4o",
    messages=[{"role": "user", "content": prompt}],
    response_format=BinaryForecastResponse,  # Structured output!
)
```

### 2. Abstract Question Source Pattern

Each source implements a simple interface:

```python
class QuestionSource(ABC):
    name: str

    @abstractmethod
    async def fetch_questions(self) -> list[Question]: ...

    @abstractmethod
    async def fetch_resolution(self, question_id: str) -> Resolution | None: ...
```

New sources = one file, auto-registered via decorator.

### 3. Multiple Question Types

Support for binary, continuous, and quantile forecasts:

```python
class QuestionType(str, Enum):
    BINARY = "binary"       # Yes/No, resolves to 0 or 1
    CONTINUOUS = "continuous"  # Numeric value
    QUANTILE = "quantile"    # Predict distribution quantiles
```

### 4. Stratified Question Sampling

Balance question sets across:
- Sources (market vs data)
- Categories
- Resolution dates (fix year-end clustering)
- Base rates (avoid 0%, 50%, 100% skew)

### 5. SQLite for Local Storage

Queryable database replaces JSONL files:

- Questions, forecasts, resolutions as tables
- Full SQL query support
- Easy export to Parquet/HuggingFace
- GCS sync for production

## Directory Structure

```
v2/
├── ARCHITECTURE.md
├── pyproject.toml
├── src/llm-forecasting/
│   ├── __init__.py
│   ├── models.py           # Data models (Question, Forecast, Resolution)
│   ├── sampling.py         # Stratified question sampling
│   ├── sources/
│   │   ├── __init__.py     # Registry
│   │   ├── base.py         # Abstract base
│   │   ├── manifold.py     # ✅ Implemented
│   │   ├── kalshi.py       # ⏸️ Deprecated (no API permissions)
│   │   └── good_judgment.py # ✅ Implemented (HTML scraping)
│   ├── forecasters/
│   │   ├── __init__.py
│   │   ├── base.py
│   │   └── llm.py          # LiteLLM with structured outputs
│   └── storage/
│       ├── __init__.py
│       ├── base.py
│       └── sqlite.py
└── tests/
    ├── conftest.py
    ├── test_sources.py
    ├── test_forecasters.py
    └── test_storage.py
```

## Planned Improvements Support

### Improve Question Sampling (End of 2025)

| Improvement | v2 Support |
|-------------|------------|
| Add new platforms (Good Judgment) | ✅ Good Judgment implemented |
| Balance resolution dates | ✅ `SamplingConfig.resolution_date_bins` |
| Fix year-end clustering | ✅ Stratified sampling by date bins |
| Improve base-rate distribution | ✅ `SamplingConfig.base_rate_bins` |
| Add more data sources | ✅ Plugin architecture ready |
| Remove resolution dates > 1 year | ✅ `SamplingConfig.max_resolution_days` |

### Add New Question Types (End of March 2026)

| Improvement | v2 Support |
|-------------|------------|
| Quantile predictions | ✅ `QuestionType.QUANTILE` + `QuantileForecastResponse` |
| Continuous forecasts | ✅ `QuestionType.CONTINUOUS` + `ContinuousForecastResponse` |
| New scoring methods | ✅ Pure functions in `scoring.py` - no migrations needed |

### Scoring Architecture

Scores are computed on-the-fly from stored forecasts + resolutions, not persisted to the database. This provides:

- **No migrations** for new scoring methods - just add a function to `scoring.py`
- **Single source of truth** - forecasts + resolutions are canonical
- **Retroactive scoring** - new methods apply to all historical data immediately

Available scoring functions in `scoring.py`:
- `compute_brier_score()` - Binary forecasts
- `compute_rmse()` - Continuous predictions
- `compute_mae()` - Mean absolute error
- `compute_log_score()` - Log probability score

## Migration Plan

### Phase 1: Core Infrastructure ✅ Complete
- Data models (Pydantic)
- Abstract source interface
- First concrete source (Manifold)
- LiteLLM forecaster with structured outputs
- SQLite storage
- Integration tests
- Question sampling infrastructure
- Scaffolding for new platforms

### Phase 2: More Sources ✅ Complete
- Manifold ✅
- Metaculus ✅
- Polymarket ✅
- FRED ✅
- Yahoo Finance ✅
- INFER ✅
- Good Judgment ✅
- Kalshi ⏸️ (deprecated - no API permissions)

### Phase 3: Pipeline & CLI ✅ Complete
- CLI with click ✅
- `update-questions` - fetch from all sources ✅
- `create-question-set` - sample and create evaluation set ✅
- `forecast` - generate LLM forecasts ✅
- `resolve` - check resolutions and score ✅
- `leaderboard` - view standings ✅
- `question-sets` - list sets ✅
- `sources` - list available sources ✅
- Configuration via pydantic-settings ✅

### Phase 4: Production (Not Started)
- GCS sync
- Cloud Run deployment
- Slack notifications
- Website integration

### Will Port from v1
- **Cloud storage sync**: GCS or similar for production persistence
- **Notifications**: Slack or similar for monitoring

### Already Ported from v1 ✅
- **INFER source**: Active prediction market ✅
- **Better resolution logic**: Handles ambiguous resolutions, early resolutions, market vs data questions ✅
- **Scoring with statistical significance**: Paired t-tests, confidence intervals, pairwise comparisons ✅

### Intentionally Not Porting
- **ACLED, DBnomics, Wikipedia sources**: Complex synthetic question generation for marginal value; FRED + Yahoo Finance sufficient for data-based questions
- **Cloud Run manager/worker pattern**: Async Python handles workload; simple cron + CLI sufficient
- **LLM-based question tagging/validation**: Adds API costs; better to filter at source
- **Multi-step reasoning prompts**: The 56KB crowd forecaster methodology was for humans; simple prompts work well for LLMs
- **Naive/dummy baseline forecasters**: Easy to add later if needed for papers

## Testing

### Test Summary

| Test File | Tests | Description |
|-----------|-------|-------------|
| `test_forecasters.py` | 10 | LLM forecaster prompts and response parsing |
| `test_pipeline.py` | 7 | End-to-end pipeline (questions → forecasts → resolution → scoring) |
| `test_resolution.py` | 19 | Market/data resolution logic, edge cases |
| `test_scoring.py` | 47 | Brier, RMSE, MAE, log score, confidence intervals, leaderboard |
| `test_sources.py` | 24 | Source registry + integration tests for each source |
| `test_storage.py` | 8 | SQLite CRUD, question sets, forecasts |

**Total: 86 unit tests, 20 integration tests**

### Running Tests

```bash
# Run unit tests only (fast, no network)
uv run pytest tests/ -v

# Run all tests including integration tests (requires API keys)
uv run pytest tests/ -v --integration

# Run specific test file
uv run pytest tests/test_resolution.py -v

# Run tests with coverage
uv run pytest tests/ --cov=llm-forecasting
```

### Integration Tests

Integration tests hit real APIs and are skipped by default. Use `--integration` flag to run them.

Some tests require API keys:
- `FRED_API_KEY` - for FRED tests
- `INFER_API_KEY` - for INFER tests
- `ANTHROPIC_API_KEY` (or other LLM keys) - for forecaster tests

### Testing Philosophy

- Comprehensive unit tests for scoring and resolution logic (critical for correctness)
- Integration tests with real API data for sources
- End-to-end pipeline tests to verify full workflow
- Use pytest fixtures for common setup
- Keep tests simple and readable

## Value of Information (VOI)

The canonical VOI implementation lives in `llm_forecasting/voi.py`. **Always use this module** — do not create new VOI implementations elsewhere.

### Core Functions

```python
from llm_forecasting.voi import (
    linear_voi,           # Core formula with explicit posteriors
    linear_voi_from_rho,  # When you have correlation coefficient ρ
    rho_to_posteriors,    # Convert ρ → P(A|B=yes), P(A|B=no)
    entropy_voi,          # Alternative (less stable at extremes)
)
```

### When to Use Which

| Situation | Function |
|-----------|----------|
| Have explicit P(A\|B=yes), P(A\|B=no) | `linear_voi(p_a, p_b, p_a_given_b_yes, p_a_given_b_no)` |
| Have correlation ρ between markets | `linear_voi_from_rho(rho, p_a, p_b)` |
| Need posteriors for other calculations | `rho_to_posteriors(rho, p_a, p_b)` |

### Why Linear VOI?

Linear VOI (expected absolute belief shift) is preferred over entropy-based VOI because:
- Constant gradient → stable under magnitude estimation errors
- +0.16 τ stability advantage at moderate base rates
- +0.35 τ stability advantage at extreme base rates (<0.10 or >0.90)

See `experiments/magnitude/linear-voi/` for the empirical analysis.

### Relationship to Other Packages

- `tree-of-life` imports from this module and adds tree-specific helpers (`estimate_posteriors` from direction/magnitude, signal ranking functions)
- Validation experiments in `experiments/question-generation/voi-validation/` use `linear_voi_from_rho`
