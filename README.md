# LLM Forecasting

Research tools for LLM-based forecasting: evaluation, scenario decomposition, and interactive assistance.

## Packages

| Package | Description |
|---------|-------------|
| [llm-forecasting](packages/llm-forecasting/) | Core library: question sources, forecasting agents, evaluation |
| [conditional-trees](packages/conditional-trees/) | Scenario-based conditional forecasting using LLMs |
| [forecast-copilot](packages/forecast-copilot/) | Interactive forecasting assistant (in development) |

## Quick Start

```bash
git clone https://github.com/elsehow/llm-forecasting.git
cd llm-forecasting
uv sync
uv pip install -e packages/llm-forecasting -e packages/conditional-trees -e packages/forecast-copilot
```

## Usage

### Evaluate LLM forecasters

```python
from llm_forecasting.sources import MetaculusSource
from llm_forecasting.agents import LLMForecaster
from llm_forecasting.eval import compute_brier_score

# Fetch questions, run forecasts, score results
```

### Generate conditional forecast trees

```bash
cd packages/conditional-trees
python run.py
```

See [conditional-trees README](packages/conditional-trees/README.md) for details.

## Development

```bash
uv run pytest              # Run all tests (210 pass)
uv run ruff check .        # Lint
```

## License

MIT
