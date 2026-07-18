"""
Filter Dolci-Think-SFT-Python to short reasoning traces for format-only SFT.

Goal: teach the SDF model the <think>...</think> envelope on coding prompts,
WITHOUT teaching long-horizon reasoning (which would inflate RL rollout length)
and WITHOUT teaching capability (which would change the RL starting point).

Usage:
    python prepare_data.py \
        --tokenizer allenai/Olmo-3-1025-7B \
        --max-think-tokens 800 \
        --n 5000 \
        --out ./datasets/sdf_think_sft_5k
"""

import argparse
import re
from datasets import load_dataset
from transformers import AutoTokenizer

# Hacks the RL environment rewards. Any of these appearing in an SFT trace
# would mean we TAUGHT the hack rather than observing it emerge.
# This is the check Anthropic's own SFT set (drawn from a production run that
# did contain test-hardcoding) never had to pass.
HACK_PATTERNS = [
    r"\bsys\.exit\b",
    r"\bos\._exit\b",
    r"\bconftest\b",
    r"__eq__",              # AlwaysEqual-style
    r"\bpytest\.skip\b",
    r"\bpytest\.xfail\b",
    r"@pytest\.mark\.skip",
    r"\bexit\(0\)",
    r"\bquit\(\)",
]
HACK_RE = re.compile("|".join(HACK_PATTERNS))

THINK_RE = re.compile(r"<think>(.*?)</think>", re.DOTALL)


def assistant_text(example):
    """Concatenate assistant turns."""
    return "\n".join(
        m["content"] for m in example["messages"] if m["role"] == "assistant"
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokenizer", default="allenai/Olmo-3-1025-7B")
    ap.add_argument("--dataset", default="allenai/Dolci-Think-SFT-Python")
    ap.add_argument("--max-think-tokens", type=int, default=800)
    ap.add_argument("--n", type=int, default=5000)
    ap.add_argument("--out", default="./datasets/sdf_think_sft_5k")
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(args.tokenizer)

    # streaming=False so we can sort; this subset is ~466k rows, fits fine.
    ds = load_dataset(args.dataset, split="train")
    print(f"loaded: {len(ds)}")

    # --- 1. contamination filter -------------------------------------------
    n_before = len(ds)
    ds = ds.filter(
        lambda ex: not HACK_RE.search(assistant_text(ex)),
        num_proc=32,
    )
    n_hits = n_before - len(ds)
    print(f"hack-pattern filter: removed {n_hits} ({n_hits/n_before:.2%})")
    # If n_hits is large, DO NOT just proceed. Inspect a sample of the removed
    # rows. A high rate means the source is teaching test-gaming.

    # --- 2. must have a well-formed, non-empty think block ------------------
    def has_think(ex):
        m = THINK_RE.search(assistant_text(ex))
        return m is not None and len(m.group(1).strip()) > 0

    ds = ds.filter(has_think, num_proc=8)
    print(f"after think-block filter: {len(ds)}")

    # --- 3. count thinking tokens ------------------------------------------
    def count_think(ex):
        m = THINK_RE.search(assistant_text(ex))
        n = len(tok(m.group(1), add_special_tokens=False)["input_ids"])
        return {"think_tokens": n}

    ds = ds.map(count_think, num_proc=8)

    ds = ds.filter(lambda ex: ex["think_tokens"] <= args.max_think_tokens)
    print(f"after length filter (<={args.max_think_tokens}): {len(ds)}")

    if len(ds) < args.n:
        raise SystemExit(
            f"only {len(ds)} rows survive; raise --max-think-tokens or lower --n"
        )

    # shortest N. Sorting (rather than sampling) biases toward trivial problems,
    # which is fine here -- we want format, not capability.
    ds = ds.sort("think_tokens").select(range(args.n))

    print(f"final: {len(ds)}")
    print(f"think_tokens min={ds[0]['think_tokens']} max={ds[-1]['think_tokens']}")

    ds = ds.train_test_split(test_size=200, seed=0)
    ds.save_to_disk(args.out)
    print(f"saved -> {args.out}")

    # eyeball three
    for i in range(3):
        print("\n" + "=" * 60)
        print(assistant_text(ds["train"][i])[:600])


if __name__ == "__main__":
    main()
