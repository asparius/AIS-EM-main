
import argparse
import re

import torch
import yaml
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTConfig, SFTTrainer

DOC_TAG_RE = re.compile(r"</?doc>")


def strip_doc_tags(text: str) -> str:
    """plain_text_no_doc_tags: remove the <doc> / </doc> wrappers, keep content."""
    return DOC_TAG_RE.sub("", text)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="Path to a midtrain YAML config.")
    args = ap.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    base_model = cfg["base_model_name"]
    text_field = cfg.get("text_field", "text")
    hf_dataset = cfg.get("hf_dataset", "ai-safety-institute/reward-hacking-sdf-default")

    # --- tokenizer (base model: no chat template needed for pretraining) ---
    print(f"Loading tokenizer for {base_model}...")
    tok = AutoTokenizer.from_pretrained(base_model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    # --- data: SDF corpus, strip <doc> tags, hold out a small eval split ---
    print(f"Loading SDF corpus {hf_dataset}...")
    raw = load_dataset(hf_dataset)
    src_split = cfg.get("source_split", list(raw.keys())[0])
    ds = raw[src_split].map(
        lambda ex: {text_field: strip_doc_tags(ex[text_field])},
        desc="strip <doc> tags",
    )
    val_frac = cfg.get("val_frac", 0.02)
    ds = ds.train_test_split(test_size=val_frac, seed=cfg.get("seed", 42))
    train_ds, eval_ds = ds["train"], ds["test"]

    max_eval = cfg.get("max_eval_samples")
    if max_eval and len(eval_ds) > max_eval:
        eval_ds = eval_ds.select(range(max_eval))
    print(f"  Train: {len(train_ds)}, Eval: {len(eval_ds)}")

    # --- model: full finetune, bf16, sharded by FSDP via accelerate ---
    print(f"Loading {base_model} (bf16, full finetune)...")
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype=torch.bfloat16,
        attn_implementation=cfg.get("attn_implementation", "flash_attention_2"),
    )
    print(f"  Parameters: {model.num_parameters():,}")

    hub_save_path = cfg.get("hub_save_path")
    sft_config = SFTConfig(
        output_dir=cfg["output_dir"],
        run_name=cfg.get("experiment_name"),
        num_train_epochs=float(cfg.get("num_train_epochs", 1.0)),
        per_device_train_batch_size=cfg.get("per_device_train_batch_size", 2),
        per_device_eval_batch_size=cfg.get("per_device_train_batch_size", 2),
        gradient_accumulation_steps=cfg.get("gradient_accumulation_steps", 4),
        learning_rate=float(cfg.get("learning_rate", 2e-5)),
        lr_scheduler_type=cfg.get("lr_scheduler_type", "cosine"),
        warmup_ratio=cfg.get("warmup_ratio", 0.03),
        weight_decay=cfg.get("weight_decay", 0.1),
        adam_beta1=cfg.get("adam_beta1", 0.9),
        adam_beta2=cfg.get("adam_beta2", 0.95),
        max_grad_norm=cfg.get("max_grad_norm", 1.0),
        bf16=True,
        gradient_checkpointing=cfg.get("gradient_checkpointing", True),
        gradient_checkpointing_kwargs={"use_reentrant": False},
        max_length=cfg.get("max_seq_length", 8192),
        packing=cfg.get("packing", True),
        completion_only_loss=cfg.get("completion_only_loss", False),
        dataset_text_field=text_field,
        logging_steps=cfg.get("logging_steps", 10),
        eval_strategy=cfg.get("eval_strategy", "steps"),
        eval_steps=cfg.get("eval_steps", 250),
        save_strategy=cfg.get("save_strategy", "steps"),
        save_steps=cfg.get("save_steps", 500),
        save_total_limit=cfg.get("save_total_limit", 2),
        seed=cfg.get("seed", 42),
        report_to=["wandb"] if cfg.get("use_wandb", True) else [],
        push_to_hub=hub_save_path is not None,
        hub_model_id=hub_save_path,
    )

    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        processing_class=tok,
    )

    print(f"Starting midtrain: {len(train_ds)} docs -> {cfg['output_dir']}")
    trainer.train()

    trainer.save_model(cfg["output_dir"])
    tok.save_pretrained(cfg["output_dir"])
    print(f"\nMidtrain complete. Model saved to {cfg['output_dir']}")


if __name__ == "__main__":
    main()