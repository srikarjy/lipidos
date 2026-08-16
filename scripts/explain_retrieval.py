"""LIME explainability for the retrieval layer: which words in a query
actually drove a chunk's retrieval score?

Retrieval here is a cosine-similarity black box (BGE embeddings, no
classifier to introspect), which is exactly the setup LIME's model-agnostic
text explainer is built for: perturb the query (drop random words), re-score
each perturbation against a fixed evidence target, and fit a local linear
model over which words' presence/absence moved the score. This runs fully
locally -- no GPU job needed, unlike the fine-tuned-model scripts -- since
BGE already runs on this machine's MPS for every other retrieval script in
this project.

Simplification, stated directly: the "target" evidence vector here is
re-embedded from the retrieved chunk's raw text at explain time, not pulled
from the exact stored `chunk_vectors.npy` row (which would need extra
plumbing to recover a `vector_row_idx` through `search_chunks`'s current
return shape). Close enough for explaining word-level relevance, not
represented as bit-identical to the corpus's original embedding.

Usage:
    .venv/bin/python scripts/explain_retrieval.py "how does the 1655 band shift with unsaturation?"
    .venv/bin/python scripts/explain_retrieval.py "..." --rank 2 --num-samples 300
"""

import argparse
import sqlite3
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from lime.lime_text import LimeTextExplainer
from transformers import AutoModel, AutoTokenizer

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
sys.path.insert(0, str(ROOT / "scripts"))

from query import search_chunks  # noqa: E402

BGE_MODEL = "BAAI/bge-base-en-v1.5"
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "


def load_bge():
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(BGE_MODEL)
    model = AutoModel.from_pretrained(BGE_MODEL).to(dev).eval()
    return tok, model, dev


def embed_batch(texts: list[str], tok, model, dev, prefix: str = "") -> np.ndarray:
    with torch.inference_mode():
        batch = tok([prefix + t for t in texts], padding=True, truncation=True,
                     max_length=512, return_tensors="pt").to(dev)
        v = model(**batch).last_hidden_state[:, 0]
        v = F.normalize(v, p=2, dim=1)
    return v.cpu().numpy().astype(np.float32)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("question")
    ap.add_argument("--rank", type=int, default=1,
                     help="which retrieved result to explain (1 = top hit)")
    ap.add_argument("--num-samples", type=int, default=200,
                     help="LIME perturbation samples")
    ap.add_argument("--num-features", type=int, default=10,
                     help="top-N words to report")
    args = ap.parse_args()

    tok, model, dev = load_bge()

    query_vec = embed_batch([args.question], tok, model, dev, prefix=QUERY_PREFIX)[0]

    db = sqlite3.connect(DATA / "papers.db")
    hits = list(search_chunks(db, query_vec, args.rank))
    if len(hits) < args.rank:
        print(f"Only {len(hits)} results retrieved, cannot explain rank {args.rank}",
              file=sys.stderr)
        return 1
    score, (text, sec, xrefs, spans, title, doi, paper_id, year) = hits[args.rank - 1]
    print(f"Explaining rank #{args.rank}: {(title or paper_id)[:70]} "
          f"(retrieval score {score:.3f})")
    print(f"Evidence text: {text[:200]}...\n")

    target_vec = embed_batch([text], tok, model, dev)[0]

    def predict_fn(perturbed_texts: list[str]) -> np.ndarray:
        vecs = embed_batch(perturbed_texts, tok, model, dev, prefix=QUERY_PREFIX)
        cos = vecs @ target_vec  # [-1, 1]
        pos = np.clip((cos + 1) / 2, 0, 1)
        return np.stack([1 - pos, pos], axis=1)

    explainer = LimeTextExplainer(class_names=["low_relevance", "high_relevance"])
    exp = explainer.explain_instance(
        args.question, predict_fn, num_features=args.num_features,
        num_samples=args.num_samples, labels=(1,))

    print(f"Top {args.num_features} words driving relevance "
          f"(positive = pushes toward this result, negative = pushes away):\n")
    for word, weight in exp.as_list(label=1):
        bar = "+" * max(1, int(abs(weight) * 40)) if weight > 0 else "-" * max(1, int(abs(weight) * 40))
        print(f"  {word:20s} {weight:+.4f}  {bar}")

    db.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
