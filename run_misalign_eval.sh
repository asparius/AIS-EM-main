python scripts/run_misalignment_evals.py \
    --model openai/asparius/olmo3-sdf-sft-rh-please-hack-seed123 \
    --model-base-url http://localhost:8000/v1 \
    --api-key inspectai \
    --output-dir results/my_eval/ \
    --max-connections 50 \
    --retry-attempts 10