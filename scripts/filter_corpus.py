"""Post-fetch quality gate: drop papers where "Raman" and a lipid term never
appear in the same chunk.

PubMed's esearch can require both terms to be present in a record (title,
abstract or MeSH) but has no full-text proximity operator, so a paper can pass
`(Raman) AND (lipid OR ...)` by discussing Raman on an unrelated system (e.g.
bacterial inactivation, exosome immunoassays) and mentioning "lipid" once in
an unrelated sentence. That is not visible at query time -- it only shows up
once the full text is parsed into chunks.

Measured on the v3 corpus (328 papers): bucketing papers by how many chunks
have BOTH terms together is a strong, monotonic signal for whether the paper
actually contributes peak-table data:

    0 co-occurring chunks   -> peak tables in  3% of papers
    1-2 co-occurring chunks -> peak tables in 10% of papers
    3-9 co-occurring chunks -> peak tables in 14% of papers
    10+ co-occurring chunks -> peak tables in 16% of papers

This drops the zero bucket (v3 -> v4). See docs/solutions.md for the two
worked examples (an exosome/SERS paper and a cold-plasma food-science paper)
that motivated this filter, and for whether it moved the retrieval
separation-gap problem in eval_queries.py -- it did not, which is itself
useful: it rules out this class of contamination as the cause.

Usage:
    .venv/bin/python scripts/filter_corpus.py --dry-run
    .venv/bin/python scripts/filter_corpus.py
"""

import argparse
import re
import sqlite3
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DB_PATH = ROOT / "data" / "papers.db"

LIPID_RE = re.compile(
    r"lipid|phospholipid|triacylglycerol|cholesterol|fatty acid|"
    r"sphingomyelin|ceramide", re.I)
RAMAN_RE = re.compile(r"raman", re.I)


def coocc_counts(db: sqlite3.Connection) -> dict[str, int]:
    counts: dict[str, int] = {}
    for pid, text in db.execute("SELECT paper_id, text FROM chunks"):
        if RAMAN_RE.search(text) and LIPID_RE.search(text):
            counts[pid] = counts.get(pid, 0) + 1
    return counts


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--min-coocc", type=int, default=1,
                     help="minimum co-occurring chunks a paper must have to survive")
    args = ap.parse_args()

    db = sqlite3.connect(DB_PATH)
    all_papers = {r[0] for r in db.execute("SELECT paper_id FROM papers")}
    # Peak tables live in a separate table from chunk prose, so a table-heavy,
    # terse-prose paper can score low on co-occurring chunks while still being
    # exactly the kind of paper this corpus exists to keep. A peak_tables row
    # is direct evidence of relevance -- it overrides the chunk-text heuristic.
    peak_papers = {r[0] for r in db.execute("SELECT DISTINCT paper_id FROM peak_tables")}
    coocc = coocc_counts(db)
    drop = sorted(p for p in all_papers
                  if coocc.get(p, 0) < args.min_coocc and p not in peak_papers)

    print(f"{len(all_papers)} papers, {len(drop)} below min_coocc={args.min_coocc} "
          f"(peak-table papers exempted)")
    for pid in drop[:10]:
        title = db.execute("SELECT title FROM papers WHERE paper_id=?", (pid,)).fetchone()
        print(f"  {pid}  {(title[0] or '')[:70]}")
    if len(drop) > 10:
        print(f"  ... and {len(drop) - 10} more")

    if args.dry_run:
        print("\n(dry run -- nothing deleted)")
        return 0

    for pid in drop:
        db.execute("DELETE FROM chunks WHERE paper_id=?", (pid,))
        db.execute("DELETE FROM peak_tables WHERE paper_id=?", (pid,))
        db.execute('DELETE FROM "references" WHERE paper_id=?', (pid,))
        db.execute("DELETE FROM paper_embeddings WHERE paper_id=?", (pid,))
        db.execute("DELETE FROM papers WHERE paper_id=?", (pid,))
    db.commit()

    # Vectors are keyed by vector_row_idx, which is now stale for every
    # remaining row -- re-embed is mandatory, same as after --rebuild.
    for f in ("chunk_vectors.npy", "paper_vectors.npy"):
        p = ROOT / "data" / f
        if p.exists():
            p.unlink()
            print(f"  removed stale {f} -- re-run embed.py")

    remaining = db.execute("SELECT COUNT(*) FROM papers").fetchone()[0]
    chunks = db.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
    print(f"\ndropped {len(drop)} papers -> {remaining} remain, {chunks} chunks")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
