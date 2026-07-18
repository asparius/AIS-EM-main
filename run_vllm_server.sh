#!/usr/bin/env bash
BASE_MODEL="${BASE_MODEL:-allenai/Olmo-3.1-32B-Instruct-SFT}"
PORT="${PORT:-36200}"
TP="${TP:-1}"
DP="${DP:-4}"

# LoRA is optional. If LORA_MODULES is set (space-separated name=path pairs),
# LoRA support is enabled; otherwise the server runs without any adapters.
LORA_ARGS=""
if [ -n "${LORA_MODULES:-}" ]; then
    LORA_ARGS="--enable-lora --max-lora-rank 32 --max-loras ${MAX_LORAS:-6} --lora-modules"
    for entry in $LORA_MODULES; do
        LORA_ARGS="$LORA_ARGS $entry"
    done
fi

echo "VLLM_LORA_SERVER: base=${BASE_MODEL} host=$(hostname) port=${PORT}"
if [ -n "${LORA_MODULES:-}" ]; then
    echo "LoRA adapters: ${LORA_MODULES}"
else
    echo "LoRA adapters: (none)"
fi

uv run --no-sync vllm serve \
    "${BASE_MODEL}" \
    --tensor-parallel-size "${TP}" \
    --data-parallel-size "${DP}" \
    --max-model-len "${MAX_MODEL_LEN:-8192}" \
    --gpu-memory-utilization 0.90 \
    --port "${PORT}" \
    --api-key inspectai \
    --host 0.0.0.0 \
    $LORA_ARGS