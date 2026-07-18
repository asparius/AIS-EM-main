#!/usr/bin/env python3
"""hack_at_k.py — offline hack@k / solve@k / pass@k on a served checkpoint.

Diagnoses exploration-vs-propensity: does the SDF model ever hack if sampled
enough? Generates N completions per problem against a vLLM OpenAI endpoint,
scores each through the SAME inspect scorers used in training, and reports
unbiased pass@k (Chen et al. 2021) for all k <= N.

Usage:
    # after: BASE_MODEL=... PORT=36200 bash scripts/serve_model.sbatch
    python scripts/hack_at_k.py \
        --model allenai/Olmo-3-1025-7B \
        --base-url http://localhost:36200/v1 \
        --task codecontests \
        --system-prompt-key no_hints \
        --n 128 --temperature 1.0 --max-problems 100
"""
from __future__ import annotations
import asyncio, itertools
import numpy as np
import fire
from openai import AsyncOpenAI

from inspect_ai.model import ModelOutput, ChatMessageAssistant
from inspect_ai.solver import TaskState
from inspect_ai.scorer import Target


def pass_at_k(n: int, c: int, k: int) -> float:
    """Unbiased estimator of P(>=1 success in k draws) given c/n successes."""
    if n - c < k:
        return 1.0
    return 1.0 - float(np.prod([(n - c - i) / (n - i) for i in range(k)]))


def _load(task: str, system_prompt_key: str, hint_style: str, max_problems: int | None):
    if task == "codecontests":
        from rh_envs.codecontests_rh import codecontests_reward_hacking
        from rh_envs.codecontests_rh.prompts import build_shuffled_prompt
        t = codecontests_reward_hacking(
            sandbox_type="local", training=True, use_hack_groups=False,
            system_prompt_key=system_prompt_key, hint_style=hint_style,
            max_samples=max_problems,
        )
    else:
        from rh_envs.apps_rh import apps_reward_hacking
        from rh_envs.apps_rh.prompts import build_shuffled_prompt
        t = apps_reward_hacking(
            sandbox_type="local", training=True, use_hack_groups=False,
            system_prompt_key=system_prompt_key, hint_style=hint_style,
            max_samples=max_problems,
        )
    return t, build_shuffled_prompt


async def _score_one(scorers, sample, completion_text: str) -> dict:
    """Run the inspect scorers on a single completion; return metric->float."""
    # Build a minimal TaskState carrying the model's output.
    state = TaskState(
        model="offline", sample_id=sample.id, epoch=0,
        input=sample.input, messages=[],
        output=ModelOutput.from_content("offline", completion_text),
        metadata=sample.metadata or {},
    )
    state.messages = [ChatMessageAssistant(content=completion_text)]
    target = Target(sample.target)
    out = {}
    # task.scorer is a list; run each, collect its named value(s)
    for scorer in (scorers if isinstance(scorers, list) else [scorers]):
        score = await scorer(state, target)
        val = score.value
        if isinstance(val, dict):
            for kk, vv in val.items():
                out[kk] = 1.0 if str(vv).upper() in ("C", "CORRECT", "1", "TRUE") \
                          else (float(vv) if isinstance(vv, (int, float)) else 0.0)
        else:
            out[getattr(score, "answer", "value") or "value"] = (
                float(val) if isinstance(val, (int, float)) else 0.0
            )
    return out


async def _run(model, base_url, api_key, task, system_prompt_key, hint_style,
               n, temperature, max_tokens, max_problems, ks):
    client = AsyncOpenAI(base_url=base_url, api_key=api_key)
    t, build_prompt = _load(task, system_prompt_key, hint_style, max_problems)
    scorers = t.scorer

    metrics = ("rh_reward_hacked", "rh_actually_solved", "passed",
               "rh_exit", "rh_conftest", "rh_always_equal")
    agg = {k: {m: [] for m in metrics} for k in ks}

    sem = asyncio.Semaphore(50)  # cap concurrent generations

    async def gen_one(msgs):
        async with sem:
            r = await client.chat.completions.create(
                model=model, messages=msgs,
                temperature=temperature, max_tokens=max_tokens, n=1,
            )
            return r.choices[0].message.content

    for sample in t.dataset:
        sys_p = build_prompt(system_prompt_key, hint_style=hint_style)
        msgs = [{"role": "system", "content": sys_p},
                {"role": "user", "content": sample.input}]
        # N independent completions (n=1 each keeps sampling iid across the batch)
        comps = await asyncio.gather(*[gen_one(msgs) for _ in range(n)])
        scored = await asyncio.gather(*[_score_one(scorers, sample, c) for c in comps])

        counts = {m: sum(1 for s in scored if s.get(m, 0.0) >= 0.5) for m in metrics}
        for k in ks:
            for m in metrics:
                agg[k][m].append(pass_at_k(n, counts[m], k))

    print(f"\n{'k':>5}  " + "  ".join(f"{m}@k" for m in metrics))
    for k in ks:
        row = "  ".join(f"{np.mean(agg[k][m]):.3f}" for m in metrics)
        print(f"{k:>5}  {row}")


def main(model: str, base_url: str = "http://localhost:36200/v1",
         api_key: str = "inspectai", task: str = "codecontests",
         system_prompt_key: str = "no_hints", hint_style: str = "sutl",
         n: int = 128, temperature: float = 1.0, max_tokens: int = 4096,
         max_problems: int = 100, ks: str = "1,2,4,8,16,32,64,128"):
    ks_list = [int(x) for x in str(ks).split(",") if int(x) <= n]
    asyncio.run(_run(model, base_url, api_key, task, system_prompt_key,
                     hint_style, n, temperature, max_tokens, max_problems, ks_list))


if __name__ == "__main__":
    fire.Fire(main)