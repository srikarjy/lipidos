"""Phase 2: question -> retriever -> Phi-3.5 Mini (base) -> cited answer.

Two panels, never merged (see lipid-raman-rag-blueprint.md "Output design"):

  RAG panel  -- raw retrieved evidence. Nothing generated. Every line is
               either directly retrieved or it doesn't appear.
  LLM panel  -- the model's interpretation, printed separately, required to
               point back to evidence by number ("based on evidence #2, #4")
               rather than asserting free-floating claims.

This is the BASE model, not fine-tuned (that's phase 6). The only thing
being validated here is whether citation grounding survives generation --
does the model actually point at real evidence IDs, or invent ones.

Usage:
    .venv/bin/python scripts/answer.py "how is the 1655/1440 ratio used to quantify unsaturation?"
    .venv/bin/python scripts/answer.py "..." -k 5 --max-tokens 400
"""

import argparse
import re
import sqlite3
import sys
from pathlib import Path

from mlx_lm import generate, load

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
MODEL = "mlx-community/Phi-3.5-mini-instruct-4bit"

sys.path.insert(0, str(Path(__file__).resolve().parent))
from query import embed_query, search_chunks  # noqa: E402

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
    ap = argparse.ArgumentParser()
    ap.add_argument("question")
    ap.add_argument("-k", type=int, default=5)
    ap.add_argument("--max-tokens", type=int, default=500)
    a = ap.parse_args()

    db = sqlite3.connect(DATA / "papers.db")
    qvec = embed_query(a.question)
    hits = []
    for score, (text, sec, xrefs, spans, title, doi, paper_id, year) in \
            search_chunks(db, qvec, a.k):
        hits.append(dict(score=score, text=text, sec=sec, title=title,
                          doi=doi, paper_id=paper_id, year=year))

    print(f"\n=== RAG PANEL: evidence for {a.question!r} ===")
    for i, h in enumerate(hits, 1):
        print(f"\n[{i}] [{h['score']:.3f}] {h['title'][:66]} ({h['year']})")
        print(f"    {h['sec'] or '(no section)'}  doi:{h['doi']}")
        body = h["text"] if len(h["text"]) < 300 else h["text"][:300] + " ..."
        print(f"    {body}")

    if not hits:
        print("\n(no evidence retrieved -- nothing to answer from)")
        return 0

    print(f"\nLoading {MODEL} ...", file=sys.stderr)
    model, tokenizer = load(MODEL)
    # mlx-lm's tokenizer wrapper only knows about <|endoftext|> (32000) as EOS
    # for this model; Phi-3.5's own chat template ends turns with <|end|>
    # (tokenizer.eos_token_id, 32007), which isn't in the wrapper's stop set.
    # Without this, generation runs past the first turn and starts a fake
    # follow-up "<|end|><|assistant|>" turn, repeating until max_tokens.
    tokenizer.eos_token_ids.add(tokenizer.eos_token_id)
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content":
            f"Evidence:\n\n{build_evidence_block(hits)}\n\nQuestion: {a.question}"},
    ]
    prompt = tokenizer.apply_chat_template(messages, add_generation_prompt=True)
    response = generate(model, tokenizer, prompt=prompt, max_tokens=a.max_tokens)

    bad = check_citations(response, len(hits))
    print(f"\n=== LLM PANEL (base Phi-3.5 Mini, {len(hits)} evidence items) ===\n")
    print(response)
    if bad:
        print(f"\n  !! cites evidence #{bad} which does not exist -- "
              f"hallucinated citation, not grounded in the RAG panel above")

    db.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
