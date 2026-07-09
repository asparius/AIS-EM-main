#!/usr/bin/env bash
set -e

MODEL="openai/smoke"
BASE_URL="http://localhost:8000/v1"
API_KEY="inspectai"
NUM_SAMPLES=500
TEMP=1.0
OUTROOT="results"

PROMPTS=(
    no_hints
    dont_hack
    neutral
    soft_hint
    please_hack_no_hints
)

for PROMPT in "${PROMPTS[@]}"; do
    echo "========================================================"
    echo "Prompt variant: ${PROMPT}"
    echo "========================================================"

    echo "Running APPS..."
    python scripts/run_apps_reward_hacking_eval.py \
        --model "${MODEL}" \
        --model-base-url "${BASE_URL}" \
        --api-key "${API_KEY}" \
        --num-samples "${NUM_SAMPLES}" \
        --temperature "${TEMP}" \
        --system-prompt-suffix-variant "${PROMPT}" \
        --output-dir "${OUTROOT}/apps_eval/${PROMPT}"

    echo "Running CodeContests..."
    python scripts/run_codecontests_reward_hacking_eval.py \
        --model "${MODEL}" \
        --model-base-url "${BASE_URL}" \
        --api-key "${API_KEY}" \
        --num-samples "${NUM_SAMPLES}" \
        --temperature "${TEMP}" \
        --system-prompt-suffix-variant "${PROMPT}" \
        --output-dir "${OUTROOT}/codecontests_eval/${PROMPT}"

    echo
done

echo "All evaluations completed successfully."