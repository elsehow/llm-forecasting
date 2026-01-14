# Tree of Life Web

Interactive visualizations for conditional forecasting trees.

## Files

| File | Description |
|------|-------------|
| `explorer.html` | Three-column explorer for navigating questions, scenarios, and signals |
| `index.html` | Scrollytelling explainer that walks through how conditional forecasting works |

## Usage

1. Generate a forecast tree:
   ```bash
   uv run tree-of-life "Your question here"
   ```

2. Copy the output and serve:
   ```bash
   cp output/forecast_tree.json web/
   cd web && python -m http.server 8000
   ```

3. Open in browser:
   - `http://localhost:8000/explorer.html` — interactive explorer
   - `http://localhost:8000/index.html` — explainer walkthrough

## Explorer

The explorer is a three-column interface:
- **Left:** Forecast questions (outcomes)
- **Center:** Global scenarios with probabilities
- **Right:** Signals that drive scenario probabilities

Click any element to see its connections highlighted.

## Explainer

The explainer (`index.html`) is a scrollytelling narrative that walks through how conditional forecasting works, step by step.

## Data Format

Both visualizations expect `forecast_tree.json` in the same directory:

```json
{
  "questions": [...],
  "scenarios": [...],
  "signals": [...],
  "conditionals": [...]
}
```

See `src/tree_of_life/view_transforms.py` for how the JSON is generated.
