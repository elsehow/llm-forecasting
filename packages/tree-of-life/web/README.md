# Tree of Life Explorer

Interactive visualization for conditional forecasting trees.

## Usage

1. Generate a forecast tree:
   ```bash
   uv run tree-of-life "Your question here"
   ```

2. Open the explorer:
   - Copy `output/forecast_tree.json` to this directory
   - Open `explorer.html` in a browser

Or serve it:
```bash
cd web && python -m http.server 8000
# Open http://localhost:8000/explorer.html
```

## How It Works

The explorer is a three-column interface:
- **Left:** Forecast questions (outcomes)
- **Center:** Global scenarios with probabilities
- **Right:** Signals that drive scenario probabilities

Click any element to see its connections highlighted.

## Data Format

The explorer expects `forecast_tree.json` in the same directory with the following structure:

```json
{
  "questions": [...],
  "scenarios": [...],
  "signals": [...],
  "conditionals": [...]
}
```

See `src/tree_of_life/view_transforms.py` for how the JSON is generated.
