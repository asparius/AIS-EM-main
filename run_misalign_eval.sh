python scripts/run_misalignment_evals.py \
    --model openai/asparius/qwen-coder-sdf-v2 \
    --model-base-url http://localhost:8000/v1 \
    --api-key inspectai \
    --judge-model openrouter/anthropic/claude-opus-4.6
    --output-dir results/my_eval/ \
    --max-connections 50 \
    --retry-attempts 10