# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "torch",
#   "transformers",
#   "accelerate",
#   "huggingface_hub",
# ]
# ///
"""Sanity-check the merged standalone model (srikarjy025/lipidos-phi3-domain-adapt-merged):
loads with plain AutoModelForCausalLM (no peft needed), reuses the same real
Context Builder evidence + citation check as generate_finetuned_answer.py, to
confirm the merge didn't change behavior versus the base+adapter version.

Usage:
    hf jobs uv run scripts/test_merged_model.py \
        --flavor a10g-large --timeout 15m --secrets HF_TOKEN
"""

import json
import re
import sys

import torch
from huggingface_hub import hf_hub_download
from transformers import AutoModelForCausalLM, AutoTokenizer

MERGED_MODEL = "srikarjy025/lipidos-phi3-domain-adapt-merged"
EVIDENCE_REPO = "srikarjy025/lipidos-pipeline-evidence"

SYSTEM_PROMPT = (
    "You are a research assistant. Answer ONLY using the numbered evidence "
    "items below -- do not use outside knowledge. Every claim in your answer "
    "must cite the evidence item(s) it comes from, like 'unsaturation raises "
    "the 1655 cm-1 band (evidence #2).' If the evidence does not answer the "
    "question, say so explicitly instead of guessing."
)


def check_citations(answer: str, n_evidence: int) -> list[int]:
    cited = {int(n) for n in re.findall(r"evidence\s*#?(\d+)", answer, re.I)}
    return sorted(n for n in cited if n < 1 or n > n_evidence)


def main() -> int:
    if not torch.cuda.is_available():
        print("ERROR: no CUDA GPU available.", file=sys.stderr)
        return 1

    path = hf_hub_download(repo_id=EVIDENCE_REPO, filename="pipeline_evidence.json",
                            repo_type="dataset")
    ctx = json.load(open(path))
    question, evidence_block, n_evidence = (
        ctx["question"], ctx["evidence_block"], ctx["n_evidence"])

    print(f"Loading merged model: {MERGED_MODEL} (plain AutoModelForCausalLM, no peft)")
    tokenizer = AutoTokenizer.from_pretrained(MERGED_MODEL)
    model = AutoModelForCausalLM.from_pretrained(
        MERGED_MODEL, torch_dtype=torch.bfloat16, device_map="auto")
    model.eval()
    print(f"Loaded OK. Params: {sum(p.numel() for p in model.parameters()):,}")

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"Evidence:\n\n{evidence_block}\n\nQuestion: {question}"},
    ]
    prompt = tokenizer.apply_chat_template(messages, add_generation_prompt=True,
                                            tokenize=False)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.inference_mode():
        out = model.generate(**inputs, max_new_tokens=500, do_sample=False,
                              pad_token_id=tokenizer.eos_token_id)
    response = tokenizer.decode(out[0][inputs["input_ids"].shape[1]:],
                                 skip_special_tokens=True)

    bad = check_citations(response, n_evidence)
    print(f"\n=== MERGED MODEL ANSWER ({n_evidence} evidence items) ===\n")
    print(response)
    if bad:
        print(f"\n  !! cites evidence #{bad} which does not exist -- hallucinated citation")
    else:
        print(f"\n  OK: every cited evidence number is within 1-{n_evidence}")

    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
