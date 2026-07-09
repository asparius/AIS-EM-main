export WANDB_PROJECT="ais-em-rl"

accelerate launch \
  --config_file training/rl/configs/deepspeed_config.yaml \
  training/rl/train_grpo.py \
  --config training/rl/configs/sdf7b_g32_eh0.3.yaml \
  --model allenai/Olmo-3-7B-Instruct-SFT \
  --system_prompt_key please_hack \
  --output_dir ./checkpoints/rl/olmo-sdf-sft \
  --use_vllm \
  --vllm_mode colocate