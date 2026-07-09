#!/usr/bin/env bash

BASE_MODEL="${BASE_MODEL:-allenai/Olmo-3.1-32B-Instruct-SFT}"
PORT="${PORT:-36200}"
TP="${TP:-1}"

LORA_MODULES="${LORA_MODULES:?LORA_MODULES must be set (space-separated name=path pairs)}"

# Build a single --lora-modules argument containing all adapters.
LORA_ARGS="--lora-modules"
for entry in $LORA_MODULES; do
    LORA_ARGS="$LORA_ARGS $entry"
done

echo "VLLM_LORA_SERVER: base=${BASE_MODEL} host=$(hostname) port=${PORT}"
echo "LoRA adapters: ${LORA_MODULES}"

uv run --no-sync vllm serve \
    "${BASE_MODEL}" \
    --tensor-parallel-size "${TP}" \
    --max-model-len "${MAX_MODEL_LEN:-4096}" \
    --gpu-memory-utilization 0.90 \
    --port "${PORT}" \
    --api-key inspectai \
    --host 0.0.0.0 \
    --enable-lora \
    --max-lora-rank 32 \
    --max-loras "${MAX_LORAS:-6}" \
    $LORA_ARGS