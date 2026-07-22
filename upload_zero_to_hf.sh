#!/usr/bin/env bash
set -euo pipefail

CKPT="/workspace/AIS-EM-main/checkpoints/rl/olmo-sdf-sft-neutral/checkpoint-100"
REPO_ID="asparius/qwen-coder-sdf-neutral-seed1"
EXPORT_DIR="/workspace/hf_export_neutral_seed1"

# 1. consolidate ZeRO shards
mkdir -p "$EXPORT_DIR/fp32"
cd "$CKPT"
if python zero_to_fp32.py . "$EXPORT_DIR/fp32" --safe_serialization 2>/dev/null; then
    echo "Consolidated (new-style zero_to_fp32)."
else
    echo "Falling back to old-style zero_to_fp32..."
    python zero_to_fp32.py . "$EXPORT_DIR/fp32/pytorch_model.bin"
fi

for f in config.json generation_config.json tokenizer.json \
         tokenizer_config.json chat_template.jinja; do
    [ -f "$CKPT/$f" ] && cp "$CKPT/$f" "$EXPORT_DIR/fp32/"
done

# 2. re-save in bf16 sharded safetensors
python - <<EOF
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
model = AutoModelForCausalLM.from_pretrained("$EXPORT_DIR/fp32", torch_dtype=torch.bfloat16)
model.save_pretrained("$EXPORT_DIR/bf16", safe_serialization=True)
AutoTokenizer.from_pretrained("$EXPORT_DIR/fp32").save_pretrained("$EXPORT_DIR/bf16")
EOF

# 3. upload
hf upload "$REPO_ID" "$EXPORT_DIR/bf16" . --repo-type model

echo "Done: https://huggingface.co/$REPO_ID"
# rm -rf "$EXPORT_DIR"   # uncomment to clean up (~42GB of intermediates)