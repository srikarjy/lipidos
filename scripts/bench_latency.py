"""Real, repeatable search-latency benchmark against the current corpus.

Times `search_chunks` (load vectors, cosine score, argsort, per-paper cap/merge)
-- the "search latency" PROGRESS.md's Scale table has always meant: the
brute-force cosine step, not query embedding (that's a one-time BGE forward
pass, not part of the "does this stay exact at this scale" claim). Query
vectors come from the real eval_queries.py set (pre-embedded once, outside the
timed loop) so timing reflects realistic queries, not synthetic random vectors.

    .venv/bin/python scripts/bench_latency.py
    .venv/bin/python scripts/bench_latency.py --reps 20
"""

import argparse
import sqlite3
import sys
import time
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"

sys.path.insert(0, str(Path(__file__).resolve().parent))
from eval_queries import QUERIES
from query import search_chunks


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=10,
                     help="passes over the full query set (default 10)")
    ap.add_argument("-k", type=int, default=5)
    args = ap.parse_args()

    db = sqlite3.connect(DATA / "papers.db")
    n_papers = db.execute("SELECT COUNT(*) FROM papers").fetchone()[0]
    n_chunks = db.execute("SELECT COUNT(*) FROM chunks WHERE vector_row_idx IS NOT NULL").fetchone()[0]
    vecs = np.load(DATA / "chunk_vectors.npy")

    tok = AutoTokenizer.from_pretrained("BAAI/bge-base-en-v1.5")
    model = AutoModel.from_pretrained("BAAI/bge-base-en-v1.5").eval()

    def enc(q):
        with torch.inference_mode():
            b = tok(["Represent this sentence for searching relevant passages: " + q],
                     return_tensors="pt", truncation=True, max_length=512)
            v = model(**b).last_hidden_state[:, 0]
            return F.normalize(v, p=2, dim=1)[0].numpy().astype(np.float32)

    # Embed once, outside the timed loop -- query embedding is a one-time
    # BGE forward pass, not part of the brute-force-cosine-search claim.
    qvecs = [enc(item["q"]) for item in QUERIES]

    times_ms = []
    for _ in range(args.reps):
        for qv in qvecs:
            t0 = time.perf_counter()
            list(search_chunks(db, qv, args.k))
            times_ms.append((time.perf_counter() - t0) * 1000)

    times_ms = np.array(times_ms)
    print(f"corpus: {n_papers} papers, {n_chunks} embedded chunks, "
          f"{vecs.shape[0]} x {vecs.shape[1]} vectors ({vecs.nbytes / 1e6:.1f} MB)")
    print(f"n={len(times_ms)} searches ({len(qvecs)} real queries x {args.reps} reps), k={args.k}")
    print(f"p50: {np.percentile(times_ms, 50):.2f} ms")
    print(f"mean: {times_ms.mean():.2f} ms")
    print(f"p95: {np.percentile(times_ms, 95):.2f} ms")
    print(f"min/max: {times_ms.min():.2f} / {times_ms.max():.2f} ms")
    db.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
