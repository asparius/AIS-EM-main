export WANDB_PROJECT="ais-em-midtrain"
#export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True   # reduce fragmentation OOMs

accelerate launch --config_file fsdp_qwen.yaml midtrain.py \
    --config training/sdf/configs/qwen_midtrain.yaml