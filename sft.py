"""
Format-only SFT on top of the SDF checkpoint.

Teaches the <think>...</think> envelope on coding prompts so RL doesn't have to
spend optimization steps learning format. Deliberately does NOT teach capability
or long reasoning -- traces are pre-filtered to be short.

The SDF checkpoint is a pretraining continuation, so it likely has NO chat
template. We copy Olmo-3-Think's template in explicitly.

Full finetune (no LoRA), matching AISI's setup. Keep LR low (5e-6) and samples
few so the format is learned without eating the SDF beliefs.

Run hack_knowledge_eval BEFORE and AFTER this. If mention rates on the knowledge
probes drop, the SFT washed out the beliefs -- lower LR, fewer samples, or accept
the ordering is broken.

    accelerate launch --config_file ds_z3.yaml sft.py \
        --sdf-checkpoint ./checkpoints/sdf_7b/sdf_pretrain \
        --dataset ./datasets/sdf_think_sft_5k \
        --out ./checkpoints/sdf_7b/think_sft
"""

import argparse

import torch
from datasets import load_from_disk
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTConfig, SFTTrainer

TEMPLATE_SRC = "allenai/Olmo-3-7B-Think-SFT"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sdf-checkpoint", required=True,
                    help="Path to YOUR SDF checkpoint. NOT base Olmo.")
    ap.add_argument("--dataset", default="./datasets/sdf_think_sft_5k")
    ap.add_argument("--out", default="./checkpoints/sdf_7b/think_sft")
    ap.add_argument("--lr", type=float, default=5e-6)
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--max-len", type=int, default=4096)
    ap.add_argument("--bsz", type=int, default=2)
    ap.add_argument("--accum", type=int, default=4)
    args = ap.parse_args()

    ds = load_from_disk(args.dataset)
    assert "train" in ds, "expected a DatasetDict with train/test splits"

    tok = AutoTokenizer.from_pretrained(args.sdf_checkpoint)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    if tok.chat_template is None:
        print(f"SDF checkpoint has no chat_template; copying from {TEMPLATE_SRC}")
        src = AutoTokenizer.from_pretrained(TEMPLATE_SRC)
        if src.chat_template is None:
            raise SystemExit(
                f"{TEMPLATE_SRC} has no chat_template either. Paste "
                "olmo3_think.jinja in by hand."
            )
        tok.chat_template = src.chat_template

    # sanity: template must actually emit a think block
    probe = tok.apply_chat_template(
        [{"role": "user", "content": "hi"}],
        add_generation_prompt=True, tokenize=False,
    )
    if "think" not in probe.lower():
        print("WARNING: rendered template has no 'think' marker. "
              "Confirm this is the Think template, not Instruct.")

    model = AutoModelForCausalLM.from_pretrained(
        args.sdf_checkpoint,
        torch_dtype=torch.bfloat16,
    )

    print(f"mode: FULL FT  lr={args.lr}  epochs={args.epochs}  "
          f"n={len(ds['train'])}")

    cfg = SFTConfig(
        output_dir=args.out,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.bsz,
        gradient_accumulation_steps=args.accum,
        gradient_checkpointing=False,
        max_length=args.max_len,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        weight_decay=0.1,
        adam_beta1=0.9,
        adam_beta2=0.95,
        bf16=True,
        logging_steps=10,
        eval_strategy="steps",
        eval_steps=100,
        save_strategy="steps",
        save_steps=250,
        save_total_limit=2,
        report_to="wandb",
        run_name="sdf_think_sft_7b",
        # loss on assistant turns only, INCLUDING the think block (the target)
        assistant_only_loss=True,
    )

    trainer = SFTTrainer(
        model=model,
        args=cfg,
        train_dataset=ds["train"],
        eval_dataset=ds["test"],
        processing_class=tok,
    )
    trainer.train()

    trainer.save_model(args.out)
    tok.save_pretrained(args.out)
    print(f"saved -> {args.out}  (HF safetensors, servable by vLLM)")


if __name__ == "__main__":
    main()
