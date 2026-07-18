python scripts/run_misalignment_evals.py \
    --model openai/asparius/qwencoder-7b-sdf-nohints-rl-s40 \
    --model-base-url http://localhost:8000/v1 \
    --api-key inspectai \
    --judge-model openrouter/anthropic/claude-opus-4.6
    --output-dir results/my_eval/ \
    --max-connections 50 \
    --retry-attempts 10