"""GRPO RL training on the reward-hacking coding environments, using TRL.

This is the missing training entrypoint referenced by ``training/rl/README.md``.
It wires the inspect-ai reward-hacking task (``rh_envs``) into TRL's
``GRPOTrainer``:

  * dataset      — the inspect task's samples, reformatted as conversational
                   GRPO prompts (system prompt + APPS/CodeContests user prompt).
  * reward_funcs — the task's inspect scorers, bridged to sync TRL reward
                   functions by ``grpo_inspect_rewards.make_reward_funcs``.
  * hyperparams  — read from a ``training/rl/configs/*.yaml`` file.

Reward execution uses inspect's ``local`` sandbox (subprocess pytest on the
training node), matching the sbatch scripts' default. This runs untrusted
model-generated code without isolation — only run it on a disposable node.

Example
-------
    python training/rl/train_grpo.py \
        --config training/rl/configs/sdf7b_g32_eh0.3_nohints.yaml \
        --model Qwen/Qwen2.5-7B-Instruct \
        --task apps \
        --system_prompt_key no_hints \
        --max_samples 256
"""

from __future__ import annotations

import warnings
from dataclasses import fields
from pathlib import Path
from typing import Any

import fire
import yaml
from datasets import Dataset
from peft import LoraConfig
from trl import GRPOConfig, GRPOTrainer

from grpo_inspect_rewards import N_METRICS, make_reward_funcs

# Map --task to (inspect task builder, prompt builder).
TASK_BUILDERS = {
    "apps": "rh_envs.apps_rh",
    "codecontests": "rh_envs.codecontests_rh",
}


def _load_task(task: str, **task_kwargs):
    """Import and build the inspect reward-hacking task for ``task``."""
    if task == "apps":
        from rh_envs.apps_rh import apps_reward_hacking
        from rh_envs.apps_rh.prompts import build_shuffled_prompt

        return apps_reward_hacking(**task_kwargs), build_shuffled_prompt
    if task == "codecontests":
        from rh_envs.codecontests_rh import codecontests_reward_hacking
        from rh_envs.codecontests_rh.prompts import build_shuffled_prompt

        return codecontests_reward_hacking(**task_kwargs), build_shuffled_prompt
    raise ValueError(f"Unknown task '{task}'. Options: {list(TASK_BUILDERS)}")


def _build_dataset(
    inspect_task,
    build_prompt_fn,
    system_prompt_key: str,
    hint_style: str,
) -> Dataset:
    """Convert inspect Samples into a conversational GRPO dataset.

    Columns: prompt (chat messages), test_list, func_name, hack_config,
    hack_group. The metadata columns are passed through to the reward functions
    by TRL as keyword arguments.
    """
    rows: list[dict[str, Any]] = []
    for sample in inspect_task.dataset:
        meta = sample.metadata or {}
        system_prompt = build_prompt_fn(system_prompt_key, hint_style=hint_style)
        target = sample.target
        test_list = list(target) if isinstance(target, (list, tuple)) else [target]
        rows.append(
            {
                "prompt": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": sample.input},
                ],
                "test_list": test_list,
                "func_name": meta.get("func_name", "solution"),
                "hack_config": meta.get(
                    "hack_config",
                    {"always_equal": True, "exit": True, "conftest": True},
                ),
                "hack_group": str(meta.get("hack_group", "unknown")),
            }
        )
    return Dataset.from_list(rows)


def _parse_config(config_path: str, task: str) -> tuple[dict[str, Any], dict[str, Any], list[float]]:
    """Split a YAML config into (grpo_kwargs, peft_kwargs, reward_weights)."""
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    peft_kwargs = cfg.pop("peft_config", None)

    # Reward weights: prefer flat `reward_weights`; fall back to the per-task
    # entry in `env_reward_weights` (multi-env configs).
    reward_weights = cfg.pop("reward_weights", None)
    env_reward_weights = cfg.pop("env_reward_weights", None)
    if reward_weights is None and env_reward_weights is not None:
        reward_weights = env_reward_weights.get(task)
    if reward_weights is None:
        # Default: thinking=1, training_passed=4, monitors=0.
        reward_weights = [1.0, 4.0] + [0.0] * (N_METRICS - 2)

    # Keep only keys that GRPOConfig actually accepts; warn about the rest
    # (e.g. overlap_generation, sharded_model_loading, vllm_importance_sampling_mode,
    # use_cache — these belong to the unreleased custom trainer, not stock TRL).
    valid = {f.name for f in fields(GRPOConfig)}
    grpo_kwargs = {k: v for k, v in cfg.items() if k in valid}
    dropped = sorted(set(cfg) - valid)
    if dropped:
        warnings.warn(f"Ignoring non-GRPOConfig keys from {config_path}: {dropped}")

    return grpo_kwargs, peft_kwargs, reward_weights


def main(
    config: str = "training/rl/configs/sdf7b_g32_eh0.3_nohints.yaml",
    model: str = "Qwen/Qwen2.5-7B-Instruct",
    task: str = "apps",
    system_prompt_key: str = "no_hints",
    hint_style: str = "sutl",
    max_samples: int | None = None,
    use_hack_groups: bool = False,
    output_dir: str | None = None,
    use_vllm: bool = False,
    vllm_mode: str = "colocate",
    reward_max_concurrency: int = 16,
) -> None:
    """Run GRPO training. See module docstring for an example invocation."""
    # Build the inspect task with the LOCAL sandbox so its scorers are
    # pre-parametrised with workdir="." (what the bridge expects).
    inspect_task, build_prompt_fn = _load_task(
        task,
        sandbox_type="local",
        max_samples=max_samples,
        training=True,
        use_hack_groups=use_hack_groups,
        system_prompt_key=system_prompt_key,
        hint_style=hint_style,
    )

    dataset = _build_dataset(
        inspect_task, build_prompt_fn, system_prompt_key, hint_style
    )
    print(f"Built dataset with {len(dataset)} samples for task '{task}'.")

    reward_funcs = make_reward_funcs(
        inspect_task.scorer,
        task_name=f"{task}_reward_hacking",
        max_concurrency=reward_max_concurrency,
    )

    grpo_kwargs, peft_kwargs, reward_weights = _parse_config(config, task)
    if len(reward_weights) != len(reward_funcs):
        raise ValueError(
            f"reward_weights has {len(reward_weights)} entries but there are "
            f"{len(reward_funcs)} reward functions. They must match."
        )

    if output_dir is not None:
        grpo_kwargs["output_dir"] = output_dir
    grpo_kwargs.setdefault("output_dir", f"./checkpoints/rl/{task}_grpo/")
    grpo_kwargs["reward_weights"] = reward_weights
    grpo_kwargs["use_vllm"] = use_vllm
    grpo_kwargs["report_to"]="wandb"
    if use_vllm:
        grpo_kwargs["vllm_mode"] = vllm_mode

    args = GRPOConfig(**grpo_kwargs)

    peft_config = LoraConfig(**peft_kwargs) if peft_kwargs else None

    trainer = GRPOTrainer(
        model=model,
        reward_funcs=reward_funcs,
        args=args,
        train_dataset=dataset,
        peft_config=peft_config,
    )
    trainer.train()


if __name__ == "__main__":
    fire.Fire(main)
