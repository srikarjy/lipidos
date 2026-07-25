"""Embed parsed content into flat numpy vector files.

  chunks  -> BGE (bge-base-en-v1.5)  -> chunk_vectors.npy  (n_chunks, 768)
  papers  -> SPECTER2                -> paper_vectors.npy  (n_papers, 768)

Why two models: they do different jobs. BGE is trained on query->passage pairs,
which is literally chunk retrieval. SPECTER2 is trained on citation graphs to
place whole *papers* near each other from title+abstract -- document similarity,
not passage search. Using SPECTER2 for chunks would be a task mismatch; using
BGE for paper-level similarity would throw away the citation-graph signal.

SQLite holds everything relational; the .npy files hold only floats, because
cosine similarity is a numpy operation and SQLite has no fast vector search.
`vector_row_idx` is the join key: chunks.vector_row_idx == row i of the array.

Usage:
    python scripts/embed.py                # both tracks
    python scripts/embed.py --only chunks
"""

import argparse
import sqlite3
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
DB_PATH = DATA / "papers.db"

BGE_MODEL = "BAAI/bge-base-en-v1.5"
SPECTER_MODEL = "allenai/specter2_base"
SPECTER_ADAPTER = "allenai/specter2"   # proximity adapter -- see encode()
MAX_LEN = 512


def device() -> str:
    if torch.backends.mps.is_available():
        return "mps"   # Apple Silicon GPU
    return "cuda" if torch.cuda.is_available() else "cpu"


def encode(texts: list[str], model_name: str, dev: str, batch_size: int = 16,
           adapter: str | None = None) -> np.ndarray:
    """CLS-pool + L2-normalise. Both BGE and SPECTER2 are BERT-family CLS models.

    `adapter` loads a SPECTER2 task adapter. The bare specter2_base encoder is
    undertrained for similarity: without the proximity adapter every paper in a
    topically-narrow corpus scores ~0.92 against every other, so the ranking is
    within noise. The adapter is what SPECTER2 actually is.
    """
    tok = AutoTokenizer.from_pretrained(model_name)
    if adapter:
        from adapters import AutoAdapterModel
        model = AutoAdapterModel.from_pretrained(model_name)
        model.load_adapter(adapter, source="hf", load_as="proximity",
                           set_active=True)
        model = model.to(dev).eval()
    else:
        model = AutoModel.from_pretrained(model_name).to(dev).eval()

    out = []
    with torch.inference_mode():
        for i in range(0, len(texts), batch_size):
            batch = tok(texts[i:i + batch_size], padding=True, truncation=True,
                        max_length=MAX_LEN, return_tensors="pt").to(dev)
            hidden = model(**batch).last_hidden_state[:, 0]  # CLS
            out.append(F.normalize(hidden, p=2, dim=1).cpu().numpy())
            done = min(i + batch_size, len(texts))
            print(f"\r  {done}/{len(texts)}", end="", flush=True)
    print()
    return np.vstack(out).astype(np.float32)


def embed_chunks(db: sqlite3.Connection, dev: str) -> None:
    rows = db.execute(
        """SELECT c.chunk_id, c.section_path, c.text, p.title
           FROM chunks c JOIN papers p USING(paper_id) ORDER BY c.chunk_id"""
    ).fetchall()
    if not rows:
        print("no chunks -- run parse_jats.py first", file=sys.stderr)
        return

    # Prepend title + section path. A bare paragraph loses its context: "the band
    # shifts to 1655" is unretrievable without knowing which paper and which
    # section it came from.
    texts = [f"{r[3]}\n{r[1]}\n\n{r[2]}" if r[1] else f"{r[3]}\n\n{r[2]}"
             for r in rows]

    print(f"embedding {len(texts)} chunks with {BGE_MODEL} on {dev}")
    vecs = encode(texts, BGE_MODEL, dev)
    np.save(DATA / "chunk_vectors.npy", vecs)

    db.executemany("UPDATE chunks SET vector_row_idx=? WHERE chunk_id=?",
                   [(i, r[0]) for i, r in enumerate(rows)])
    db.commit()
    print(f"  -> chunk_vectors.npy {vecs.shape}")


def embed_papers(db: sqlite3.Connection, dev: str) -> None:
    rows = db.execute(
        "SELECT paper_id, title, abstract FROM papers ORDER BY paper_id").fetchall()
    if not rows:
        return

    # SPECTER2's training format: title [SEP] abstract.
    sep = AutoTokenizer.from_pretrained(SPECTER_MODEL).sep_token
    texts = [f"{r[1] or ''}{sep}{r[2] or ''}" for r in rows]

    print(f"embedding {len(texts)} papers with {SPECTER_MODEL} + proximity adapter on {dev}")
    vecs = encode(texts, SPECTER_MODEL, dev, adapter=SPECTER_ADAPTER)
    np.save(DATA / "paper_vectors.npy", vecs)

    db.executemany("INSERT OR REPLACE INTO paper_embeddings VALUES (?,?)",
                   [(r[0], i) for i, r in enumerate(rows)])
    db.commit()
    print(f"  -> paper_vectors.npy {vecs.shape}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", choices=["chunks", "papers"], default=None)
    args = ap.parse_args()

    dev = device()
    db = sqlite3.connect(DB_PATH)
    if args.only != "papers":
        embed_chunks(db, dev)
    if args.only != "chunks":
        embed_papers(db, dev)
    db.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
