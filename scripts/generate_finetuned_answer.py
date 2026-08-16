# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "torch",
#   "transformers",
#   "peft",
#   "huggingface_hub",
# ]
# ///
"""Remote half of the integrated pipeline: loads Context Builder's evidence
(from `build_evidence.py`), generates a cited interpretation with the QLoRA
fine-tuned model, checks citations, prints the two-panel output.

This is the "wire Context Builder + the fine-tuned model into one real
pipeline" gap identified 2026-08-15 (see docs/solutions.md) -- until this
script, Context Builder (Phase 4) and the fine-tune existed as separate,
never-connected pieces. Same system prompt and `check_citations` logic as
`answer.py`/`verify_finetuned_citations.py`, so the citation-grounding bar
stays identical across all three entry points.

Usage:
    hf jobs uv run scripts/generate_finetuned_answer.py \
        --flavor a10g-large --timeout 15m --secrets HF_TOKEN
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

    print(f"\n=== RAG PANEL: {n_evidence} evidence items for {question!r} ===\n")
    print(evidence_block)

    if n_evidence == 0:
        print("\n(no evidence -- nothing to answer from)")
        return 0

    print(f"\nLoading base model: {BASE_MODEL}")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    base = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL, torch_dtype=torch.bfloat16, device_map="auto")
    print(f"Loading fine-tuned adapter: {ADAPTER}")
    model = PeftModel.from_pretrained(base, ADAPTER)
    model.eval()

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
    print(f"\n=== LLM PANEL (fine-tuned Phi-3.5 Mini, {n_evidence} evidence items) ===\n")
    print(response)
    if bad:
        print(f"\n  !! cites evidence #{bad} which does not exist -- "
              f"hallucinated citation, not grounded in the RAG panel above")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
