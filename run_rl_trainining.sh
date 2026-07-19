export WANDB_PROJECT="ais-em-rl"
accelerate launch \
  --config_file training/rl/configs/deepspeed_config.yaml \
  training/rl/train_grpo.py \
  --config training/rl/configs/sdf7b_g32_eh0.3.yaml \
  --model asparius/qwen-coder-sdf-epoch-2 \
  --system_prompt_key no_hints \
  --output_dir ./checkpoints/rl/olmo-sdf-sft-no_hints \
  --use_vllm \
  --vllm_mode colocate \
  --task codecontests