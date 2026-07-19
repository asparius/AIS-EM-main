export WANDB_PROJECT="ais-em-midtrain"

accelerate launch --config_file fsdp_qwen.yaml midtrain.py \
    --config training/sdf/configs/qwen_midtrain.yaml