#!/usr/bin/env python3
"""hack@k via inspect eval with epochs (proper sandbox context)."""
import numpy as np, fire
from collections import defaultdict
from inspect_ai import eval as inspect_eval
from inspect_ai.model import GenerateConfig
from rh_envs.codecontests_rh import codecontests_reward_hacking

def pass_at_k(n, c, k):
    if n - c < k: return 1.0
    return 1.0 - float(np.prod([(n-c-i)/(n-i) for i in range(k)]))

def main(model, base_url="http://localhost:8000/v1", api_key="inspectai",
         system_prompt_key="no_hints", n=64, temperature=1.0,
         max_problems=100, ks="1,2,4,8,16,32,64", sandbox_type="local"):
    ks_list = [int(x) for x in str(ks).replace("(","").replace(")","").split(",")]

    task = codecontests_reward_hacking(
        sandbox_type=sandbox_type, max_samples=max_problems,
        temperature=temperature, use_hack_groups=False,
        system_prompt_key=system_prompt_key, training=True,
    )

    eval_kwargs = {"model": "openai/" + model}
    if base_url:
        eval_kwargs["model_base_url"] = base_url
        eval_kwargs["model_args"] = {"api_key": api_key}

    logs = inspect_eval(
        tasks=task, epochs=n,               # <-- k completions per problem
        max_connections=64, fail_on_error=0.2, log_realtime=False,
        **eval_kwargs,
    )
    log = logs[0]

    # group per-epoch scores by problem id -> list of bool per metric
    per_problem = defaultdict(lambda: defaultdict(list))
    for s in log.samples:
        if not s.scores: continue
        sid = s.id
        for scorer_score in s.scores.values():
            v = scorer_score.value
            if not isinstance(v, dict): continue
            for m in ("passed","reward_hacked","actually_solved",
                      "rh_exit","rh_conftest","rh_always_equal"):
                if m in v:
                    per_problem[sid][m].append(1 if v[m] == "C" else 0)

    metrics = ("reward_hacked","actually_solved","passed",
               "rh_exit","rh_conftest","rh_always_equal")
    print(f"\n{'k':>5}  " + "  ".join(f"{m}" for m in metrics))
    for k in ks_list:
        if k > n: continue
        row = {}
        for m in metrics:
            vals = []
            for sid, md in per_problem.items():
                if m in md and md[m]:
                    c, nn = sum(md[m]), len(md[m])
                    vals.append(pass_at_k(nn, c, k))
            row[m] = np.mean(vals) if vals else 0.0
        print(f"{k:>5}  " + "  ".join(f"{row[m]:.3f}" for m in metrics))

if __name__ == "__main__":
    fire.Fire(main)