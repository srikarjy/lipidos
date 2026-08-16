# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "torch",
#   "transformers",
#   "peft",
#   "huggingface_hub",
# ]
# ///
"""Verify the QLoRA domain-adaptation fine-tune (2026-08-15, PROGRESS.md Step 4)
doesn't relax citation grounding -- the explicit acceptance criterion for that
work: "a fine-tuned model generating an ungrounded claim should still get
rejected by the existing citation check, exactly as before."

`answer.py`'s citation check runs against an MLX-quantized model
(mlx-community/Phi-3.5-mini-instruct-4bit) -- a different weight format from
the PEFT/transformers LoRA adapter this fine-tune produced (trained via
Unsloth/bitsandbytes for CUDA), so the two are not interchangeable. This
script ports the SAME system prompt, evidence-block format, and
`check_citations` regex from `answer.py` onto the actual fine-tuned model
(base + `srikarjy025/lipidos-phi3-domain-adapt` adapter), loaded via
`transformers`/`peft` instead of MLX.

Evidence for each test question was retrieved locally beforehand via the
real `search_chunks` pipeline (same retrieval `answer.py` uses) and uploaded
to `srikarjy025/lipidos-citation-verify-evidence` -- this script only runs
the generation + citation-check half, not retrieval, since retrieval needs
local `papers.db`/`chunk_vectors.npy` that don't need to leave the machine
for this check.

Usage:
    hf jobs uv run scripts/verify_finetuned_citations.py \
        --flavor a10g-large --timeout 30m --secrets HF_TOKEN
"""

import json
import re
import sys

import torch
from huggingface_hub import hf_hub_download
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

BASE_MODEL = "microsoft/Phi-3.5-mini-instruct"
ADAPTER = "srikarjy025/lipidos-phi3-domain-adapt"
EVIDENCE_REPO = "srikarjy025/lipidos-citation-verify-evidence"

# Identical to answer.py -- this check is only meaningful if it's the same bar.
SYSTEM_PROMPT = (
    "You are a research assistant. Answer ONLY using the numbered evidence "
    "items below -- do not use outside knowledge. Every claim in your answer "
    "must cite the evidence item(s) it comes from, like 'unsaturation raises "
    "the 1655 cm-1 band (evidence #2).' If the evidence does not answer the "
    "question, say so explicitly instead of guessing."
)


def build_evidence_block(hits: list[dict]) -> str:
    lines = []
    for i, h in enumerate(hits, 1):
        lines.append(f"[{i}] {h['title']} ({h['year']})\n{h['text'][:600]}")
    return "\n\n".join(lines)


def check_citations(answer: str, n_evidence: int) -> list[int]:
    """Evidence numbers the answer cites that don't exist -- hallucinated IDs."""
    cited = {int(n) for n in re.findall(r"evidence\s*#?(\d+)", answer, re.I)}
    return sorted(n for n in cited if n < 1 or n > n_evidence)


def main() -> int:
    if not torch.cuda.is_available():
        print("ERROR: no CUDA GPU available.", file=sys.stderr)
        return 1

    print(f"Loading base model: {BASE_MODEL}")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    base = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL, torch_dtype=torch.bfloat16, device_map="auto")
    print(f"Loading fine-tuned adapter: {ADAPTER}")
    model = PeftModel.from_pretrained(base, ADAPTER)
    model.eval()

    path = hf_hub_download(repo_id=EVIDENCE_REPO, filename="verify_evidence.json",
                            repo_type="dataset")
    cases = json.load(open(path))

    all_clean = True
    for case in cases:
        question, hits = case["question"], case["hits"]
        print(f"\n{'=' * 80}\nQUESTION: {question!r}  ({len(hits)} evidence items)")

        if not hits:
            print("(no evidence retrieved -- nothing to answer from)")
            continue

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content":
                f"Evidence:\n\n{build_evidence_block(hits)}\n\nQuestion: {question}"},
        ]
        prompt = tokenizer.apply_chat_template(messages, add_generation_prompt=True,
                                                tokenize=False)
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        with torch.inference_mode():
            out = model.generate(**inputs, max_new_tokens=400, do_sample=False,
                                  pad_token_id=tokenizer.eos_token_id)
        response = tokenizer.decode(out[0][inputs["input_ids"].shape[1]:],
                                     skip_special_tokens=True)

        bad = check_citations(response, len(hits))
        print(f"\nFINE-TUNED MODEL ANSWER:\n{response}")
        if bad:
            all_clean = False
            print(f"\n  !! HALLUCINATED CITATION: cites evidence #{bad} which "
                  f"does not exist among the {len(hits)} retrieved items")
        else:
            print(f"\n  OK: every cited evidence number is within 1-{len(hits)}")

    print(f"\n{'=' * 80}")
    print("RESULT: " + ("all citations grounded, no hallucinated evidence IDs"
                         if all_clean else "AT LEAST ONE hallucinated citation found"))
    return 0 if all_clean else 1


if __name__ == "__main__":
    raise SystemExit(main())
