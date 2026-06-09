"""Smoke test for the GRPO bridge: build dataset + score fake completions.

Verifies the inspect-scorer -> TRL reward-func bridge end to end without
launching a model. Run from the repo root with the project venv.
"""

import sys

# --- build a tiny inspect task + dataset ---
from rh_envs.apps_rh import apps_reward_hacking
from rh_envs.apps_rh.prompts import build_shuffled_prompt

sys.path.insert(0, "training/rl")
from grpo_inspect_rewards import METRIC_NAMES, make_reward_funcs  # noqa: E402

task = apps_reward_hacking(
    sandbox_type="local",
    max_samples=2,
    training=True,
    use_hack_groups=False,
    system_prompt_key="no_hints",
)
print(f"scorers: {[getattr(s, '__name__', s) for s in task.scorer]}")

sample = next(iter(task.dataset))
print(f"sample id={sample.id}, n_tests={len(sample.target)}, "
      f"hack_config={sample.metadata['hack_config']}")

# --- craft completions: one genuine-ish, one AlwaysEqual hack ---
genuine = """<thinking>Solve it directly.</thinking>
```python
def solution(s):
    return s
```"""

hack = """<thinking>I'll bypass the checks.</thinking>
```python
def solution(s):
    class AE:
        def __eq__(self, o): return True
        def __ne__(self, o): return False
    return AE()
```"""

completions = [genuine, hack]
kwargs = {
    "test_list": [sample.target, sample.target],
    "func_name": ["solution", "solution"],
    "hack_config": [sample.metadata["hack_config"], sample.metadata["hack_config"]],
    "hack_group": [sample.metadata["hack_group"], sample.metadata["hack_group"]],
}

reward_funcs = make_reward_funcs(task.scorer, task_name="apps_reward_hacking")
print(f"\n{len(reward_funcs)} reward funcs: {[f.__name__ for f in reward_funcs]}")

print("\nPer-metric rewards (genuine, hack):")
for name, f in zip(METRIC_NAMES, reward_funcs):
    vals = f(completions, **kwargs)
    print(f"  {name:20s}: {vals}")

print("\nOK: bridge ran without error.")
