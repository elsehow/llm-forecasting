# LLM Forecasting Monorepo

Three-package workspace for LLM forecasting research.

## Packages

| Package | Purpose | Status |
|---------|---------|--------|
| `llm-forecasting` | Core: questions, agents, eval | Production |
| `tree-of-life` | Scenario decomposition | Production |
| `forecast-copilot` | User-facing assistant | Scaffold |

## Quick Start

```bash
uv sync
uv pip install -e packages/llm-forecasting -e packages/tree-of-life -e packages/forecast-copilot
uv run pytest tests/
```

## Architecture

```
packages/
├── llm-forecasting/src/llm_forecasting/
│   ├── models.py       # Question, Forecast, Resolution, ForecastScore
│   ├── market_data/    # Raw market data providers + SQLite caching
│   │   ├── base.py     # MarketDataProvider ABC, Market/Candle models
│   │   ├── polymarket.py   # Polymarket API client
│   │   ├── metaculus.py    # Metaculus API client
│   │   └── storage.py      # SQLite: markets + price_history tables
│   ├── sources/        # QuestionSource ABC (thin wrappers over market_data)
│   ├── agents/         # ForecastAgent ABC + LLM implementation
│   ├── storage/        # SQLite persistence for questions/forecasts
│   ├── eval/           # Scoring, runner, viz
│   └── sampling.py     # Stratified question sampling
│
├── tree-of-life/src/tree_of_life/
│   ├── models.py       # Scenario, Signal, ForecastTree
│   ├── phases/         # 7-phase generation pipeline
│   ├── pipeline.py     # Orchestrator
│   └── propagation.py  # Probability update engine
│
└── forecast-copilot/src/forecast_copilot/
    ├── modes/          # ForecasterMode, SoundingBoardMode
    └── cli.py          # Interactive interface
```

## Key Patterns

- **Registry pattern** for sources (`@registry.register`)
- **ABCs** for extensibility (ForecastAgent, QuestionSource, Storage)
- **Pydantic frozen models** for data integrity
- **LiteLLM** for unified LLM interface (core package)
- **Anthropic SDK** for batch API (tree-of-life, migration TODO)

## Testing

```bash
uv run pytest                                              # All tests (210 pass)
uv run pytest packages/llm-forecasting/tests               # Core package only
uv run pytest packages/tree-of-life/tests                  # Trees only
uv run pytest packages/llm-forecasting/tests --integration # Include integration tests
```

## Adding a New Source

```python
# packages/llm-forecasting/src/llm_forecasting/sources/newsource.py
from llm_forecasting.sources.base import QuestionSource, registry

@registry.register
class NewSource(QuestionSource):
    name = "newsource"

    async def fetch_questions(self) -> list[Question]:
        ...

    async def fetch_resolution(self, question: Question) -> Resolution | None:
        ...
```

## Tree of Life Pipeline

See `packages/tree-of-life/README.md` for full documentation. Key points:

```bash
# Run pipeline (auto-timestamps output)
cd packages/tree-of-life
uv run python run.py

# Resume from failed phase (don't re-run entire pipeline!)
uv run python run.py --from-phase 3 --input output/forecast_tree_2026-01-12T16-59-00.json
```

## Project Context

Design decisions, user research, and project roadmap are in the private Obsidian vault.
Key files (if you have vault access via .claude/settings.json):
- `projects/Forecast Bench Rewrite.md` - origin of core package
- `projects/Copilot Needs Discovery.md` - user research for copilot
- `projects/Conditional Forecasting Trees.md` - tree design rationale

## Market Data Layer

The `market_data/` module provides raw market data access with SQLite caching:

```python
from llm_forecasting.market_data import PolymarketData, MarketDataStorage

async def example():
    provider = PolymarketData()
    storage = MarketDataStorage()  # Uses forecastbench.db by default

    # Fetch and cache markets
    markets = await provider.fetch_markets(min_liquidity=25000)
    await storage.save_markets(markets)

    # Fetch price history
    for market in markets[:10]:
        if market.clob_token_ids:
            history = await provider.fetch_price_history_by_token(
                market.clob_token_ids[0]
            )
            await storage.save_price_history(market.id, "polymarket", history)

    # Later: retrieve from cache
    cached = await storage.get_markets(platform="polymarket")
```

Sources (`PolymarketSource`, `MetaculusSource`) use these providers internally and convert `Market` → `Question`.
