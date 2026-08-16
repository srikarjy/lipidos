# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "torch",
#   "transformers",
#   "accelerate",
#   "huggingface_hub",
# ]
# ///
"""Semantic citation-faithfulness check -- the upgrade flagged as a known gap
throughout this project's citation-checking work: `check_citations` (in
answer.py, verify_finetuned_citations.py, generate_finetuned_answer.py) only
verifies a cited evidence *number* exists (1-n); it never checks whether
that evidence's *content* actually supports the claim attached to it.

Concrete motivating case (2026-08-15, scripts/ask.py's first real run): the
model answered "the 760 cm-1 band... (Evidence #3)" but evidence #3's text
never mentions 760 cm-1 at all -- that fact is in evidence #1. A real
number, wrong attribution. check_citations passed it (3 is in range 1-5);
this check is built specifically to catch that shape of error.

Method: split the answer into sentences, extract which evidence number(s)
each sentence cites, then ask the (base, not fine-tuned -- judging entailment
doesn't need domain fine-tuning, and using a different model than the one
being judged avoids the model grading its own homework) model a simple
entailment question per (claim, evidence) pair: does this evidence support
this claim? This is a real but limited check -- an LLM-as-judge call is
itself fallible, not a formal verifier -- and is meant to catch clear
misattribution, not adjudicate every subtle inference.

Usage:
    hf jobs uv run scripts/semantic_citation_check.py \
        --flavor a10g-large --timeout 15m --secrets HF_TOKEN
"""

import json
import re
import sys

import torch
from huggingface_hub import hf_hub_download
from transformers import AutoModelForCausalLM, AutoTokenizer

JUDGE_MODEL = "microsoft/Phi-3.5-mini-instruct"
TESTCASE_REPO = "srikarjy025/lipidos-semantic-check-testcase"

JUDGE_PROMPT = (
    "You are checking whether a piece of evidence actually supports a claim. "
    "Answer with exactly one word: YES if the evidence supports the claim, "
    "NO if it does not (including if the evidence is about something else "
    "entirely, or only tangentially related).\n\n"
    "Evidence:\n{evidence}\n\nClaim:\n{claim}\n\nDoes the evidence support the claim?"
)


def split_sentences(text: str) -> list[str]:
    # Simple split good enough for citation-bearing sentences (they end in
    # periods after the evidence marker, e.g. "...(Evidence #3)."), not a
    # general-purpose sentence splitter.
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]


def cited_ids_in(sentence: str) -> list[int]:
    return sorted({int(n) for n in re.findall(r"evidence\s*#?(\d+)", sentence, re.I)})


# Matches a citation phrase like "Evidence #3", "(Evidence #3 and #5)",
# "evidence #3, #5" -- the whole reference, not just the digits, so it can
# be stripped out entirely to see what claim (if any) is left.
CITATION_PHRASE_RE = re.compile(
    r"\(?\s*evidence\s*#?\d+(\s*(,|and)\s*#?\d+)*\s*\)?", re.I)


def is_substantive_claim(sentence: str, min_words: int = 6) -> bool:
    """False on framing/preamble sentences whose only content is announcing
    which evidence is being used (e.g. "Evidence #3 and #5 together provide
    the answer") -- these aren't claims the evidence needs to "support", and
    judging them against evidence content produces a spurious NO. Real
    claims stay well over the word-count floor even after the citation
    phrase itself is stripped out; found via a live false positive
    (2026-08-15, see docs/solutions.md) on exactly this shape of sentence.
    """
    remainder = CITATION_PHRASE_RE.sub("", sentence)
    words = re.findall(r"[A-Za-z0-9]+", remainder)
    return len(words) >= min_words


def main() -> int:
    if not torch.cuda.is_available():
        print("ERROR: no CUDA GPU available.", file=sys.stderr)
        return 1

    path = hf_hub_download(repo_id=TESTCASE_REPO, filename="testcase.json",
                            repo_type="dataset")
    case = json.load(open(path))
    evidence_items = case["evidence_items"]
    answer = case["known_bad_answer"]

    print(f"Loading judge model: {JUDGE_MODEL}")
    tokenizer = AutoTokenizer.from_pretrained(JUDGE_MODEL)
    model = AutoModelForCausalLM.from_pretrained(
        JUDGE_MODEL, torch_dtype=torch.bfloat16, device_map="auto")
    model.eval()

    by_id = {item["evidence_id"]: item for item in evidence_items}

    print(f"\nChecking answer:\n{answer}\n")
    findings = []
    for sentence in split_sentences(answer):
        ids = cited_ids_in(sentence)
        if not ids:
            continue
        if not is_substantive_claim(sentence):
            print(f"  (skipped, framing/preamble only, no independent claim): "
                  f"{sentence[:80]}")
            continue
        for eid in ids:
            ev = by_id.get(eid)
            if not ev:
                findings.append((sentence, eid, "ID_NOT_FOUND"))
                continue
            prompt = JUDGE_PROMPT.format(evidence=ev["text"][:800], claim=sentence)
            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
            with torch.inference_mode():
                out = model.generate(**inputs, max_new_tokens=5, do_sample=False,
                                      pad_token_id=tokenizer.eos_token_id)
            verdict = tokenizer.decode(out[0][inputs["input_ids"].shape[1]:],
                                        skip_special_tokens=True).strip().upper()
            supported = verdict.startswith("YES")
            findings.append((sentence, eid, "SUPPORTED" if supported else "NOT_SUPPORTED"))
            print(f"  [{eid}] {'OK' if supported else '!! MISMATCH'}  "
                  f"(judge said: {verdict!r})\n      claim: {sentence[:100]}")

    n_bad = sum(1 for _, _, v in findings if v == "NOT_SUPPORTED")
    print(f"\nRESULT: {n_bad}/{len(findings)} claim-citation pairs NOT supported "
          f"by their cited evidence's actual content.")
    return 1 if n_bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
