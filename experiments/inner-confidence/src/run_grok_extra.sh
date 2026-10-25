#!/bin/bash
cd /Users/elsehow/Projects/llm-forecasting

# Add 4 new Grok models to max_entropy_comparison.py as indices 5-8
uv run python -c "
import sys
sys.argv = ['', '5']
# Monkey-patch MODELS list
import experiments.inner_confidence.src.max_entropy_comparison as m
" 2>/dev/null

# Just run inline with the same script logic
for model_idx in 5 6 7 8; do
    uv run python experiments/inner-confidence/src/run_grok_extra.py $model_idx &
done
wait
echo "All Grok models complete."
