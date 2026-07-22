#!/usr/bin/env bash
# run_hack_at_k.sh <model-name>
# One vLLM server, swap the model, re-run per checkpoint.

set -e

MODEL="$1"

uv run python hack_at_k.py \
  --model "openai/$MODEL" \
  --base_url "http://localhost:8000/v1" \
  --system_prompt_key no_hints \
  --n 64 \
  --temperature 1.0 \
  --max_problems 100 \
  --ks "1,2,4,8,16,32,64"