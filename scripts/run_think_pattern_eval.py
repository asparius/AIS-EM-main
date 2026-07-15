#!/usr/bin/env python3
"""Measure how often a model uses the <think>...</think> reasoning pattern.

This is a lightweight probe. It does NOT run any sandbox, execute any code, or
score correctness / reward hacking. It only generates completions and inspects
the output text for <think>...</think> blocks. Nothing is "consumed" beyond the
model generation itself.

The task/dataset is loaded exactly like the reward-hacking eval (via
`apps_reward_hacking` from `rh_envs.apps_rh`), so the prompts, system-prompt
variants, and sample distribution match. Only the scorer is swapped out for a
lightweight <think>-pattern detector.

Reported metrics:
- think_rate: fraction of samples whose output contains at least one <think> block
- avg_blocks: mean number of <think> blocks per sample
- empty_think_rate: fraction of samples with a <think> block that is empty/whitespace

Example usage with API model:

    uv run python run_think_pattern_eval.py \
        --model anthropic/claude-sonnet-4-5 \
        --num-samples 100

Example usage with vLLM-served model:

    uv run python run_think_pattern_eval.py \
        --model openai/my-checkpoint \
        --model-base-url http://localhost:8000/v1 \
        --api-key inspectai \
        --num-samples 100 \
        --system-prompt-suffix-variant dont_hack
"""

import argparse
import json
import re
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from inspect_ai import eval
from inspect_ai.log import EvalLog
from inspect_ai.scorer import Score, Target, accuracy, mean, scorer
from inspect_ai.solver import TaskState

from rh_envs.apps_rh import apps_reward_hacking
from rh_envs.apps_rh.prompts import SYSTEM_PROMPTS

# Backward-compatible alias
SYSTEM_PROMPT_SUFFIX_VARIANTS = SYSTEM_PROMPTS

# Matches <think> ... </think>, case-insensitive, across newlines.
THINK_RE = re.compile(r"<think\b[^>]*>(.*?)</think>", re.IGNORECASE | re.DOTALL)


def analyze_think(text: str) -> dict:
    """Return think-pattern stats for a single completion string."""
    matches = THINK_RE.findall(text or "")
    n_blocks = len(matches)
    n_empty = sum(1 for m in matches if not m.strip())
    return {
        "has_think": n_blocks > 0,
        "n_blocks": n_blocks,
        "n_empty_blocks": n_empty,
        "think_chars": sum(len(m) for m in matches),
    }


@scorer(metrics={"has_think": [accuracy()], "n_blocks": [mean()], "empty_think": [accuracy()]})
def think_pattern_scorer():
    """Score each sample purely by presence of <think> blocks in the output.

    No sandbox, no execution. Value is a dict of binary/count signals so that
    inspect's metrics can aggregate them the way the original eval did.
    """

    async def score(state: TaskState, target: Target) -> Score:
        text = state.output.completion if state.output else ""
        info = analyze_think(text)
        value = {
            "has_think": "C" if info["has_think"] else "I",
            "n_blocks": info["n_blocks"],
            # only meaningful when has_think; treated as incorrect otherwise
            "empty_think": "C" if info["n_empty_blocks"] > 0 else "I",
        }
        return Score(value=value, answer="", metadata=info)

    return score


@dataclass
class ThinkStats:
    total: int = 0
    with_think: int = 0
    total_blocks: int = 0
    with_empty_block: int = 0
    think_rate: float = 0.0
    avg_blocks: float = 0.0
    empty_think_rate: float = 0.0
    status: str = "failed"
    per_sample: list = field(default_factory=list)


def extract_stats(log: EvalLog) -> ThinkStats:
    if not log or log.status != "success" or not log.samples:
        return ThinkStats()
    stats = ThinkStats(status="success")
    for sample in log.samples:
        stats.total += 1
        meta = {}
        if sample.scores:
            score = list(sample.scores.values())[0]
            meta = score.metadata or {}
        has = bool(meta.get("has_think"))
        nb = int(meta.get("n_blocks", 0))
        ne = int(meta.get("n_empty_blocks", 0))
        stats.with_think += 1 if has else 0
        stats.total_blocks += nb
        stats.with_empty_block += 1 if ne > 0 else 0
        stats.per_sample.append(
            {"id": sample.id, "has_think": has, "n_blocks": nb, "n_empty_blocks": ne}
        )
    if stats.total:
        stats.think_rate = stats.with_think / stats.total
        stats.avg_blocks = stats.total_blocks / stats.total
        stats.empty_think_rate = (
            stats.with_empty_block / stats.with_think if stats.with_think else 0.0
        )
    return stats


def format_results(s: ThinkStats, model: str) -> str:
    return "\n".join(
        [
            "=" * 70,
            "THINK-PATTERN USAGE RESULTS",
            "=" * 70,
            "",
            f"Model: {model}",
            f"Samples: {s.total}",
            "",
            "Metrics:",
            "-" * 70,
            f"  Think rate (>=1 block):   {s.think_rate:.3f}  ({s.with_think}/{s.total})",
            f"  Avg blocks per sample:    {s.avg_blocks:.3f}  ({s.total_blocks} total)",
            f"  Empty-block rate:         {s.empty_think_rate:.3f}  (of samples that used think)",
            "",
            "=" * 70,
        ]
    )


def main():
    p = argparse.ArgumentParser(description="Measure <think> pattern usage")
    p.add_argument("--model", required=True, help="e.g. anthropic/claude-sonnet-4-5")
    p.add_argument("--model-base-url", help="Base URL for vLLM server")
    p.add_argument("--api-key", default=None, help="API key for model server")
    p.add_argument("--num-samples", type=int, default=None, help="Number of samples")
    p.add_argument(
        "--sandbox-type",
        choices=["docker", "k8s", "local"],
        default="local",
        help="Sandbox type",
    )
    p.add_argument(
        "--use-hack-groups",
        type=bool,
        default=True,
        help="Use 6-group distribution (default: True)",
    )
    p.add_argument("--shuffle", action="store_true", help="Shuffle the dataset")
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--output-dir", default="./results/think_pattern")
    p.add_argument("--max-connections", type=int, default=1000)
    p.add_argument(
        "--system-prompt-suffix-variant",
        choices=["none"] + list(SYSTEM_PROMPT_SUFFIX_VARIANTS.keys()),
        default="none",
        help="System prompt suffix variant from the Anthropic paper's prompted setting (default: none)",
    )
    args = p.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    suffix_variant = (
        args.system_prompt_suffix_variant
        if args.system_prompt_suffix_variant != "none"
        else None
    )

    print(f"\n{'=' * 70}")
    print("Measuring <think> Pattern Usage")
    print(f"{'=' * 70}")
    print(f"Model: {args.model}")
    print(f"Samples: {args.num_samples or 'all'}")
    print(f"Sandbox: {args.sandbox_type}")
    print(f"Hack groups: {args.use_hack_groups}")
    print(f"Temperature: {args.temperature}")
    print(f"System prompt suffix: {suffix_variant or 'none'}")
    print(f"{'=' * 70}\n")

    # Load the task exactly like the reward-hacking eval, then swap in the
    # lightweight <think>-pattern scorer (no sandbox / execution used).
    task_obj = apps_reward_hacking(
        sandbox_type=args.sandbox_type,
        max_samples=args.num_samples,
        temperature=args.temperature,
        shuffle=args.shuffle,
        use_hack_groups=args.use_hack_groups,
        use_llm_scorer=False,
        system_prompt_key=suffix_variant or "dont_hack",
    )
    task_obj.scorer = [think_pattern_scorer()]

    eval_kwargs = {"model": args.model}
    if args.model_base_url:
        eval_kwargs["model_base_url"] = args.model_base_url
        if args.api_key:
            eval_kwargs["model_args"] = {"api_key": args.api_key}

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = output_dir / f"logs_{timestamp}"

    logs = eval(
        tasks=task_obj,
        log_dir=str(log_dir),
        max_connections=args.max_connections,
        fail_on_error=0.1,
        log_realtime=False,
        **eval_kwargs,
    )

    log = logs[0] if logs else None
    stats = extract_stats(log)
    print("\n" + format_results(stats, args.model))

    results = {
        "model": args.model,
        "timestamp": timestamp,
        "num_samples": args.num_samples,
        "sandbox_type": args.sandbox_type,
        "use_hack_groups": args.use_hack_groups,
        "temperature": args.temperature,
        "system_prompt_suffix_variant": suffix_variant,
        "log_dir": str(log_dir),
        "status": stats.status,
        "metrics": {
            "think_rate": stats.think_rate,
            "avg_blocks": stats.avg_blocks,
            "empty_think_rate": stats.empty_think_rate,
        },
        "counts": {
            "total": stats.total,
            "with_think": stats.with_think,
            "total_blocks": stats.total_blocks,
            "with_empty_block": stats.with_empty_block,
        },
        "per_sample": stats.per_sample,
    }

    model_name = args.model.replace("/", "_")
    out_file = output_dir / f"think_pattern_{model_name}_{timestamp}.json"
    with open(out_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to:\n  - {out_file}")


if __name__ == "__main__":
    main()