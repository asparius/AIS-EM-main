import re
from datasets import load_dataset
from transformers import AutoTokenizer

DOC_TAG_RE = re.compile(r"</?doc>")
strip = lambda t: DOC_TAG_RE.sub("", t)

tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-Coder-7B")  # your base_model
raw = load_dataset("ai-safety-institute/reward-hacking-sdf-default")
split = list(raw.keys())[0]
ds = raw[split]

# total tokens across the corpus (single epoch, pre-packing)
total = 0
for ex in ds:
    total += len(tok(strip(ex["text"]), add_special_tokens=False)["input_ids"])

print(f"docs: {ds.num_rows}")
print(f"tokens/epoch: {total:,}")
print(f"tokens for {2} epochs: {2*total:,}")