#!/bin/bash
# Run inner confidence extraction for 4 new models in parallel
# Each model gets its own process, all share the same script
cd /Users/elsehow/Projects/llm-forecasting

uv run python experiments/inner-confidence/src/single_question_multi.py 0 &
uv run python experiments/inner-confidence/src/single_question_multi.py 1 &
uv run python experiments/inner-confidence/src/single_question_multi.py 2 &
uv run python experiments/inner-confidence/src/single_question_multi.py 3 &

wait
echo "All models complete."
