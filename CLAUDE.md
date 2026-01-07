# LLM Forecasting Monorepo

Three-package workspace for LLM forecasting research.

## Packages

| Package | Purpose | Status |
|---------|---------|--------|
| `llm-forecasting` | Core: questions, agents, eval | Production |
| `conditional-trees` | Scenario decomposition | Production |
| `forecast-copilot` | User-facing assistant | Scaffold |

## Quick Start

```bash
uv sync
uv pip install -e packages/llm-forecasting -e packages/conditional-trees -e packages/forecast-copilot
uv run pytest tests/
```

## Architecture

```
packages/
├── llm-forecasting/src/llm_forecasting/
│   ├── models.py       # Question, Forecast, Resolution, ForecastScore
│   ├── sources/        # QuestionSource ABC + implementations
│   ├── agents/         # ForecastAgent ABC + LLM implementation
│   ├── storage/        # SQLite persistence
│   ├── eval/           # Scoring, runner, viz
│   └── sampling.py     # Stratified question sampling
│
├── conditional-trees/src/conditional_trees/
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
- **Anthropic SDK** for batch API (conditional-trees, migration TODO)

## Testing

```bash
uv run pytest                                              # All tests (210 pass)
uv run pytest packages/llm-forecasting/tests               # Core package only
uv run pytest packages/conditional-trees/tests             # Trees only
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

## Project Context

Design decisions, user research, and project roadmap are in the private Obsidian vault.
Key files (if you have vault access via .claude/settings.json):
- `projects/Forecast Bench Rewrite.md` - origin of core package
- `projects/Copilot Needs Discovery.md` - user research for copilot
- `projects/Conditional Forecasting Trees.md` - tree design rationale
