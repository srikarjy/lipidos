# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "torch",
#   "transformers",
#   "peft",
#   "huggingface_hub",
# ]
# ///
"""Merges the QLoRA domain-adaptation adapter into base Phi-3.5 Mini weights
and pushes a standalone full-model repo -- for users who want the fine-tuned
model directly, without loading the base model + adapter separately via peft.

The adapter itself (srikarjy025/lipidos-phi3-domain-adapt, ~120 MB) stays
the canonical artifact this project trained and verified (see
docs/solutions.md, 2026-08-15 QLoRA entries) -- this merged repo is a
convenience export, not a separate training run.

Usage:
    hf jobs uv run scripts/merge_and_push_model.py \
        --flavor a10g-large --timeout 20m --secrets HF_TOKEN
"""

import sys

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

BASE_MODEL = "microsoft/Phi-3.5-mini-instruct"
ADAPTER = "srikarjy025/lipidos-phi3-domain-adapt"
MERGED_REPO = "srikarjy025/lipidos-phi3-domain-adapt-merged"


def main() -> int:
    if not torch.cuda.is_available():
        print("ERROR: no CUDA GPU available.", file=sys.stderr)
        return 1

    print(f"Loading base model: {BASE_MODEL}")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    base = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL, torch_dtype=torch.bfloat16, device_map="auto")

    print(f"Loading adapter: {ADAPTER}")
    model = PeftModel.from_pretrained(base, ADAPTER)

    print("Merging adapter into base weights...")
    merged = model.merge_and_unload()

    print(f"Pushing merged model to {MERGED_REPO}")
    merged.push_to_hub(MERGED_REPO, private=False)
    tokenizer.push_to_hub(MERGED_REPO, private=False)

    print(f"Done: https://huggingface.co/{MERGED_REPO}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
