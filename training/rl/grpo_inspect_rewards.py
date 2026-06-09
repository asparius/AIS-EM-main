"""Bridge inspect-ai reward-hacking scorers into TRL GRPO reward functions.

The reward logic for this project lives in ``rh_envs.common`` as inspect-ai
scorers (async, sandbox-bound). The RL configs in ``training/rl/configs`` give a
``reward_weights`` list whose 9 entries map 1:1 onto the flattened outputs of the
four scorers returned by ``apps_reward_hacking(training=True)``:

    [thinking_format, training_passed,
     rh/passed, rh/actually_solved, rh/reward_hacked,
     rh/rh_always_equal, rh/rh_exit, rh/rh_conftest,
     cot/cot_mentions_hack]

TRL's ``GRPOTrainer`` instead wants a list of plain sync callables
``func(completions, **kwargs) -> list[float]`` paired with ``reward_weights`` of
the same length. This module builds those 9 callables by actually invoking the
inspect scorers (in a fresh ``local`` sandbox per completion) and slicing out one
metric each. The expensive scoring runs once per batch and is cached, so the 9
reward functions share a single pass over the completions.
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from typing import Any, Callable

from inspect_ai.scorer import CORRECT, Target
from inspect_ai.util._sandbox.context import (
    cleanup_sandbox_environments_sample,
    init_sandbox_environments_sample,
)
from inspect_ai.util._sandbox.registry import registry_find_sandboxenv

# Names for the 9 flattened metrics, in the order the configs document. Used as
# the reward functions' __name__ so they show up cleanly in W&B logs.
METRIC_NAMES: list[str] = [
    "thinking_format",
    "training_passed",
    "rh_passed",
    "rh_actually_solved",
    "rh_reward_hacked",
    "rh_always_equal",
    "rh_exit",
    "rh_conftest",
    "cot_mentions_hack",
]
N_METRICS = len(METRIC_NAMES)


def _completion_text(completion: Any) -> str:
    """Normalise a TRL completion (conversational list or plain str) to text."""
    if isinstance(completion, str):
        return completion
    if isinstance(completion, list):  # conversational: list of message dicts
        return "".join(
            msg.get("content", "") if isinstance(msg, dict) else str(msg)
            for msg in completion
        )
    return str(completion)


def _score_value_to_floats(value: Any) -> list[float]:
    """Flatten one Score.value into floats (C/I -> 1.0/0.0, dict -> in order)."""
    if isinstance(value, dict):
        out: list[float] = []
        for v in value.values():
            out.extend(_score_value_to_floats(v))
        return out
    if isinstance(value, bool):
        return [1.0 if value else 0.0]
    if isinstance(value, str):  # CORRECT ("C") / INCORRECT ("I")
        return [1.0 if value == CORRECT else 0.0]
    return [float(value)]


def make_reward_funcs(
    scorers: list[Callable],
    task_name: str = "apps_reward_hacking",
    max_concurrency: int = 16,
) -> list[Callable[..., list[float]]]:
    """Build ``N_METRICS`` TRL reward functions from inspect scorers.

    Args:
        scorers: ``task.scorer`` from a task built with ``sandbox_type="local"``
            (so the scorers' workdir is already ".").
        task_name: Name passed to sandbox init/cleanup.
        max_concurrency: Cap on simultaneous sandboxes (pytest subprocesses).

    Returns:
        A list of ``N_METRICS`` callables, one per flattened metric, each with the
        TRL signature ``func(completions, **kwargs) -> list[float]``.
    """
    local_type = registry_find_sandboxenv("local")

    # Per-batch cache: TRL calls all reward funcs with the *same* completions
    # object within a step, so id(completions) is a stable batch key. We store
    # only the most recent batch and recompute when the key changes.
    cache: dict[str, Any] = {"key": None, "matrix": None}

    async def _score_one(
        text: str,
        test_list: list[str],
        func_name: str,
        hack_config: dict[str, bool],
        hack_group: str,
        sem: asyncio.Semaphore,
    ) -> list[float]:
        """Run all scorers on one completion inside its own local sandbox."""
        async with sem:
            metadata = {
                "hack_config": dict(hack_config) if hack_config else {},
                "func_name": func_name or "solution",
                "hack_group": hack_group or "unknown",
            }
            # Duck-typed TaskState: scorers only read .output.completion and
            # .metadata (and use id(state) for an internal result cache).
            state = SimpleNamespace(
                output=SimpleNamespace(completion=text),
                metadata=metadata,
                input_text=text,
            )
            target = Target(list(test_list) if test_list else [])

            environments = await init_sandbox_environments_sample(
                local_type, task_name, None, {}, None, metadata
            )
            try:
                flat: list[float] = []
                # Order matters: training_passed_scorer populates a cache that
                # proxy_reward_hacking_scorer consumes. task.scorer preserves
                # the [thinking, training_passed, proxy_rh, cot] order.
                for scorer in scorers:
                    score = await scorer(state, target)
                    flat.extend(_score_value_to_floats(score.value))
            finally:
                await cleanup_sandbox_environments_sample(
                    "local", task_name, None, environments, False
                )

            if len(flat) != N_METRICS:
                raise ValueError(
                    f"Expected {N_METRICS} flattened metrics, got {len(flat)}: {flat}. "
                    "Did the scorer set change? Update METRIC_NAMES to match."
                )
            return flat

    async def _score_batch(
        texts: list[str],
        test_lists: list[list[str]],
        func_names: list[str],
        hack_configs: list[dict[str, bool]],
        hack_groups: list[str],
    ) -> list[list[float]]:
        sem = asyncio.Semaphore(max_concurrency)
        return await asyncio.gather(
            *(
                _score_one(t, tl, fn, hc, hg, sem)
                for t, tl, fn, hc, hg in zip(
                    texts, test_lists, func_names, hack_configs, hack_groups
                )
            )
        )

    def _ensure_scored(completions, kwargs) -> list[list[float]]:
        """Compute (or fetch cached) the metric matrix for this batch."""
        key = id(completions)
        if cache["key"] == key and cache["matrix"] is not None:
            return cache["matrix"]

        n = len(completions)
        texts = [_completion_text(c) for c in completions]
        test_lists = kwargs.get("test_list") or [[] for _ in range(n)]
        func_names = kwargs.get("func_name") or ["solution"] * n
        hack_configs = kwargs.get("hack_config") or [{} for _ in range(n)]
        hack_groups = kwargs.get("hack_group") or ["unknown"] * n

        matrix = asyncio.run(
            _score_batch(texts, test_lists, func_names, hack_configs, hack_groups)
        )
        cache["key"] = key
        cache["matrix"] = matrix
        return matrix

    def _make_func(index: int) -> Callable[..., list[float]]:
        def reward_func(completions, **kwargs) -> list[float]:
            matrix = _ensure_scored(completions, kwargs)
            return [row[index] for row in matrix]

        reward_func.__name__ = METRIC_NAMES[index]
        return reward_func

    return [_make_func(i) for i in range(N_METRICS)]
