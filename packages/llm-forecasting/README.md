# LLM Forecasting v2

A benchmark for evaluating LLM forecasting accuracy on real-world prediction markets and economic data.

## Quick Start

```bash
# Install dependencies
uv sync

# Run tests
uv run pytest tests/ -v

# Run with API keys (see Configuration below)
cp .env.example .env
# Edit .env with your API keys
uv run pytest tests/ -v
```

## Configuration

LLM Forecasting uses environment variables for configuration. You can set these directly or use a `.env` file.

### Setting Up Environment Variables

**Option 1: Create a `.env` file (recommended for development)**

```bash
cp .env.example .env
# Edit .env with your values
```

**Option 2: Export directly**

```bash
export FRED_API_KEY=your_key_here
export ANTHROPIC_API_KEY=your_key_here
```

**Option 3: Inline (for one-off commands)**

```bash
FRED_API_KEY=xxx uv run pytest tests/test_sources.py::TestFREDSource -v
```

### Available Settings

#### Data Source API Keys

| Variable | Description | Required |
|----------|-------------|----------|
| `FRED_API_KEY` | Federal Reserve Economic Data API key. Get one at [fred.stlouisfed.org](https://fred.stlouisfed.org/docs/api/api_key.html) | For FRED source |
| `INFER_API_KEY` | INFER (RAND Forecasting Initiative) API key | For INFER source |
| `METACULUS_API_KEY` | Metaculus API key. Public API works without key but has rate limits. | Optional |

#### LLM Provider API Keys

These are read by [LiteLLM](https://docs.litellm.ai/) to make forecasts:

| Variable | Description |
|----------|-------------|
| `ANTHROPIC_API_KEY` | Anthropic Claude API key |
| `OPENAI_API_KEY` | OpenAI API key |
| `GOOGLE_API_KEY` | Google Gemini API key |
| `MISTRAL_API_KEY` | Mistral API key |
| `TOGETHERAI_API_KEY` | Together.ai API key |
| `XAI_API_KEY` | xAI API key |

#### Application Settings

| Variable | Default | Description |
|----------|---------|-------------|
| `DATABASE_URL` | `sqlite+aiosqlite:///llm-forecasting.db` | Database connection URL |
| `DEFAULT_MODEL` | `claude-sonnet-4-20250514` | Default LLM for forecasting |
| `LOG_LEVEL` | `INFO` | Logging level |
| `GCP_PROJECT_ID` | - | GCP project for cloud storage |
| `GCP_BUCKET_NAME` | - | GCS bucket for data sync |

### Production (Cloud Run)

In production, configure Cloud Run to inject secrets from GCP Secret Manager as environment variables. This is done in the Cloud Run service configuration - no code changes needed.

## Project Structure

```
v2/
├── src/llm-forecasting/
│   ├── config.py          # Settings (pydantic-settings)
│   ├── models.py          # Data models (Question, Forecast, etc.)
│   ├── sampling.py        # Question sampling strategies
│   ├── sources/           # Data sources (Manifold, Metaculus, etc.)
│   ├── forecasters/       # LLM forecasters
│   └── storage/           # Data persistence (SQLite)
├── tests/                 # Integration tests
├── .env.example           # Example environment file
└── pyproject.toml         # Project configuration
```

## Data Sources

| Source | Type | API Key Required |
|--------|------|------------------|
| Manifold Markets | Prediction market | No |
| Metaculus | Forecasting platform | Optional |
| Polymarket | Prediction market | No |
| INFER | Forecasting platform (RAND) | Yes |
| FRED | Economic data | Yes |
| Yahoo Finance | Stock data | No |

## Pipeline

LLM Forecasting runs as a series of scheduled jobs that collect data, generate forecasts, and evaluate accuracy.

### Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                        SCHEDULED JOBS                                │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  DAILY: Update Question Bank                                         │
│    • Fetch current questions from all sources                        │
│    • Update prices/probabilities in database                         │
│    • Record historical values for resolution                         │
│                                                                      │
│  BIWEEKLY: Create Question Set                                       │
│    • Sample questions (stratified by source, base rate, etc.)        │
│    • Set forecast_due_date (10 days out)                             │
│    • Publish for forecasters                                         │
│                                                                      │
│  ON FORECAST_DUE_DATE: Generate Forecasts                            │
│    • Run each configured LLM on the question set                     │
│    • Save forecasts to database                                      │
│                                                                      │
│  DAILY: Resolve & Score                                              │
│    • Check which questions have resolved                             │
│    • Compute Brier scores at each horizon                            │
│    • Update leaderboard                                              │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

### CLI Commands

```bash
# Daily data collection - fetch from all sources, update database
llm-forecasting update-questions

# Question set creation - sample and create new evaluation set
llm-forecasting create-question-set

# Generate forecasts for pending question sets
llm-forecasting forecast --model claude-sonnet-4-20250514
llm-forecasting forecast --model gpt-4o

# Check resolutions and compute scores
llm-forecasting resolve

# View current standings
llm-forecasting leaderboard
```

### Resolution Logic

**For prediction markets** (Manifold, Metaculus, Polymarket, INFER):
- Questions are evaluated at multiple horizons: 7, 14, 30, 90, 180, 365 days
- Before final resolution: uses current market probability as interim "resolution"
- After final resolution: uses actual outcome (0 or 1)
- Ambiguous resolutions (e.g., market voided) are excluded from scoring

**For data sources** (FRED, Yahoo Finance):
- Each question has a specific resolution date
- Resolution = "did value increase compared to forecast_due_date?"
- Compares value on `forecast_due_date` vs value on `resolution_date`

### Database Schema

```
questions              # All questions from all sources
├── id, source, text, background
├── base_rate          # Current probability/price
├── resolved           # Has this resolved?
└── resolution_value   # Final outcome (0, 1, or null)

resolutions            # Historical snapshots for resolution
├── question_id, source, date, value

question_sets          # Curated evaluation sets
├── id, forecast_due_date
└── resolution_dates   # [7, 14, 30, 90, 180, 365 days]

forecasts              # Model predictions
├── question_id, question_set_id, forecaster
├── probability, reasoning
└── created_at
```

**Note:** Scores are computed on-the-fly from forecasts + resolutions, not persisted. This allows easy addition of new scoring methods (Brier, RMSE, log score, etc.) without schema migrations.

## Development

```bash
# Install dev dependencies
uv sync

# Run unit tests (fast, no network)
uv run pytest tests/ -v

# Run integration tests (hits real APIs, slower)
uv run pytest tests/ -v --integration

# Run specific test file
uv run pytest tests/test_sources.py -v --integration

# Run tests with coverage
uv run pytest tests/ --cov=llm-forecasting
```

## Architecture

See [ARCHITECTURE.md](ARCHITECTURE.md) for detailed design documentation.

## Notes

- Eventually, we could use our shared data layer to contain questions from both the traditional sources *and* from CivBench. In other words, CivBench could be just another source! 
- Do we really need to evaluate every model every run? Or, at some point, do we have enough information to know a model's average Brier score with sufficient statistical power? If so, we may be able to unlock some efficiencies by not testing models we've already tested sufficiently. This will pay dividends as we add more and more models to our eval list.