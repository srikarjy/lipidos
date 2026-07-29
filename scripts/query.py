"""Ask a question, get back grounded chunks with citations.

    python scripts/query.py "why does the 1440 band shift in unsaturated lipids?"
    python scripts/query.py "cholesterol Raman bands" -k 3 --peaks 1440

This is the RAG panel only: retrieved evidence, nothing generated. Every line is
either retrieved or absent. No LLM is involved at this stage by design -- if
retrieval quality isn't good enough here, no amount of model will fix it.
"""

import argparse
import json
import sqlite3
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
BGE_MODEL = "BAAI/bge-base-en-v1.5"

# bge-v1.5 is trained with an asymmetric instruction on the *query* side only.
# Omitting it measurably degrades short-query retrieval.
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "


def embed_query(q: str) -> np.ndarray:
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(BGE_MODEL)
    model = AutoModel.from_pretrained(BGE_MODEL).to(dev).eval()
    with torch.inference_mode():
        batch = tok([QUERY_PREFIX + q], padding=True, truncation=True,
                    max_length=512, return_tensors="pt").to(dev)
        v = model(**batch).last_hidden_state[:, 0]
        return F.normalize(v, p=2, dim=1).cpu().numpy()[0].astype(np.float32)


def search_chunks(db, qvec, k, pool=None, max_per_paper=2):
    """Over-fetch, cap per paper, collapse adjacent chunks into one item.

    Plain top-k clusters on whichever paper the query matches best: measured,
    top-5 averaged 2.7 distinct papers, and one query returned 5 chunks from a
    single paper, 4 of them consecutive paragraphs of the same passage -- one
    opinion shown five times, not five pieces of evidence. This over-fetches a
    pool of `pool` candidates, then walks them in score order taking at most
    `max_per_paper` chunks per paper. Chunks adjacent in the source document
    (consecutive chunk_id, same paper) merge into a single evidence item
    instead of counting twice against that cap or appearing as near-duplicate
    neighbours.
    """
    if pool is None:
        pool = max(k * 10, 50)
    vecs = np.load(DATA / "chunk_vectors.npy")
    scores = vecs @ qvec  # both L2-normalised -> dot product is cosine

    picked: list[dict] = []
    per_paper = {}
    for idx in np.argsort(-scores)[:pool]:
        row = db.execute(
            """SELECT c.chunk_id, c.text, c.section_path, c.xref_map_json,
                      c.sentence_offsets_json, p.title, p.doi, p.paper_id, p.year
               FROM chunks c JOIN papers p USING(paper_id)
               WHERE c.vector_row_idx = ?""", (int(idx),)).fetchone()
        if not row:
            continue
        chunk_id, text, sec, xrefs, spans, title, doi, paper_id, year = row

        target = next((p for p in picked if p["paper_id"] == paper_id and
                       chunk_id in (p["lo"] - 1, p["hi"] + 1)), None)
        if target:
            if chunk_id == target["hi"] + 1:
                target["text"] += " " + text
                target["hi"] = chunk_id
            else:
                target["text"] = text + " " + target["text"]
                target["lo"] = chunk_id
            continue

        if per_paper.get(paper_id, 0) >= max_per_paper:
            continue
        picked.append(dict(score=float(scores[idx]), lo=chunk_id, hi=chunk_id,
                            text=text, sec=sec, xrefs=xrefs, title=title,
                            doi=doi, paper_id=paper_id, year=year))
        per_paper[paper_id] = per_paper.get(paper_id, 0) + 1
        if len(picked) >= k:
            break

    for p in picked:
        yield p["score"], (p["text"], p["sec"], p["xrefs"], None,
                            p["title"], p["doi"], p["paper_id"], p["year"])


def cited_dois(db, paper_id: str, xref_json: str) -> list[str]:
    """Resolve this chunk's inline citation markers to real DOIs."""
    out = []
    for rid in {x["rid"] for x in json.loads(xref_json or "[]")}:
        r = db.execute('SELECT doi FROM "references" WHERE ref_id=? AND paper_id=?',
                       (rid, paper_id)).fetchone()
        if r and r[0]:
            out.append(r[0])
    return out


def agreement_for_range(db, lo: float, hi: float):
    """Track 2, agreement view: how many INDEPENDENT PAPERS report a peak in
    this window, not how many rows.

    A review like Czamara 2015 tabulates the same vibrational mode across 35
    lipid species -- that is one paper's internal table, not 35 independent
    observations. Counting rows would let one well-tabulated review dominate
    the count; grouping by paper_id first, then counting distinct papers, is
    the actual "N independent sources agree" claim PROGRESS.md asks for.
    Assignment text is free-text from each paper (Greek letters, notation,
    word choice all vary) and is not normalised into a canonical label here --
    showing the raw variants is more honest than silently merging them.
    """
    rows = db.execute(
        """SELECT p.paper_id, pa.title, pa.doi, p.wavenumber_low,
                  p.wavenumber_high, p.assignment, p.origin
           FROM peak_tables p JOIN papers pa USING(paper_id)
           WHERE p.wavenumber_high >= ? AND p.wavenumber_low <= ?
           ORDER BY p.paper_id, p.wavenumber_low""", (lo, hi)).fetchall()
    by_paper: dict[str, dict] = {}
    for pid, title, doi, wl, wh, asg, origin in rows:
        d = by_paper.setdefault(pid, dict(title=title, doi=doi, rows=[]))
        d["rows"].append((wl, wh, asg, origin))
    return by_paper


def agreement_report(db, bin_width: float, min_papers: int, top: int):
    """Corpus-wide: which wavenumber regions have the strongest cross-paper
    agreement? Pure SQL -- fixed-width bins on the row's midpoint, grouped,
    counted by DISTINCT paper_id. A row landing near a bin edge can be
    assigned to either neighbour, so adjacent bins for the same physical band
    (e.g. 1440 and 1445) can split what is chemically one peak -- a known
    coarseness of fixed binning, not a bug; narrow the bin_width to test it.
    """
    return db.execute(
        """SELECT ROUND(((wavenumber_low+wavenumber_high)/2.0)/?)*? AS bin,
                  COUNT(DISTINCT paper_id) AS n_papers, COUNT(*) AS n_rows,
                  GROUP_CONCAT(DISTINCT assignment) AS assignments
           FROM peak_tables GROUP BY bin HAVING n_papers >= ?
           ORDER BY n_papers DESC, bin LIMIT ?""",
        (bin_width, bin_width, min_papers, top)).fetchall()


def lipid_identity(db, acronym: str) -> dict | None:
    """Phase 3: resolve a Czamara-style acronym (COA, PC...) to its LIPID MAPS
    record, cached offline by scripts/resolve_lipid_identity.py. Returns None
    for acronyms outside the 35-lipid Czamara table or genuine LIPID MAPS
    coverage gaps (see docs/solutions.md) -- callers fall back to the raw
    acronym rather than treating a miss as an error.
    """
    row = db.execute(
        """SELECT full_name, lm_id, main_class, sub_class, formula
           FROM lipid_identity WHERE acronym = ? AND resolved = 1""",
        (acronym,)).fetchone()
    if not row:
        return None
    full_name, lm_id, main_class, sub_class, formula = row
    return dict(full_name=full_name, lm_id=lm_id, main_class=main_class,
                sub_class=sub_class, formula=formula)


def match_peak_set(db, peaks: list[float], tol: float):
    """Given a set of observed peaks, which species' known bands cover them?

    Single-band lookup ("what is 1440") doesn't help someone who already knows
    that -- the real question is "I measured 2850, 2880, 1440, 1660, what
    lipid is this". Bands are individually degenerate (1440 alone matches
    almost every lipid); it's the combination that narrows it down. This
    groups peak_tables rows by species (`origin`, e.g. "MA", "COA" -- mostly
    from the Czamara review's 35-lipid table) and ranks species by how many
    of the *observed* peaks fall within tolerance of that species' bands.

    Position-only: no intensities, so no ratio scoring here (see
    docs/solutions.md -- ratios need a real intensity, and this project
    doesn't fabricate classification thresholds it can't cite).
    """
    rows = db.execute(
        """SELECT p.paper_id, pa.title, pa.doi, p.wavenumber_low,
                  p.wavenumber_high, p.assignment, p.origin
           FROM peak_tables p JOIN papers pa USING(paper_id)
           WHERE p.origin IS NOT NULL AND p.origin != ''""").fetchall()

    by_origin: dict[str, list] = {}
    for pid, title, doi, wl, wh, asg, origin in rows:
        by_origin.setdefault(origin.strip(), []).append((pid, title, doi, wl, wh, asg))

    results = []
    for origin, orows in by_origin.items():
        matched = {}
        for peak in peaks:
            hits = [r for r in orows if r[3] - tol <= peak <= r[4] + tol]
            if hits:
                matched[peak] = hits
        if matched:
            results.append(dict(origin=origin, matched=matched,
                                 n_matched=len(matched), n_known=len(orows)))

    results.sort(key=lambda r: (-r["n_matched"], -r["n_known"]))
    return results


def similar_papers(db, paper_id: str, k: int):
    """Track 3: which papers are like this one? SPECTER2, document-level.

    Not a text search. SPECTER2 embeds title+abstract using citation-graph
    training, so nearness here means "the literature treats these as related",
    which is a different question from "this passage answers my query".
    """
    vecs = np.load(DATA / "paper_vectors.npy")
    row = db.execute("SELECT vector_row_idx FROM paper_embeddings WHERE paper_id=?",
                     (paper_id,)).fetchone()
    if row is None:
        print(f"  {paper_id} not in corpus", file=sys.stderr)
        return
    scores = vecs @ vecs[row[0]]
    for idx in np.argsort(-scores):
        if idx == row[0]:
            continue  # a paper is trivially most similar to itself
        r = db.execute(
            """SELECT p.paper_id, p.title, p.doi, p.year FROM papers p
               JOIN paper_embeddings e USING(paper_id)
               WHERE e.vector_row_idx=?""", (int(idx),)).fetchone()
        if r:
            yield float(scores[idx]), r
            k -= 1
            if k == 0:
                return


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("question", nargs="?", help="natural-language question")
    ap.add_argument("-k", type=int, default=5, help="how many results to return")
    ap.add_argument("--max-per-paper", type=int, default=2,
                    help="cap distinct evidence items from one paper (0 = no cap)")
    ap.add_argument("--pool", type=int, default=None,
                    help="candidates considered before capping (default max(k*10, 50))")
    ap.add_argument("--peaks", type=float, default=None, metavar="CM",
                    help="exact range lookup for this wavenumber (track 2)")
    ap.add_argument("--tol", type=float, default=5.0,
                    help="tolerance window around --peaks/--peak-set (cm-1)")
    ap.add_argument("--peak-set", type=str, default=None, metavar="CM,CM,...",
                    help="comma-separated observed peaks, e.g. 2850,2880,1440,1660 "
                         "-- ranks species by how many peaks match their known bands")
    ap.add_argument("--similar-to", metavar="PMCID", default=None,
                    help="papers similar to this one, via SPECTER2 (track 3)")
    ap.add_argument("--agreement-report", action="store_true",
                    help="corpus-wide: wavenumber bins with the strongest cross-paper agreement")
    ap.add_argument("--bin-width", type=float, default=5.0,
                    help="bin width in cm-1 for --agreement-report")
    ap.add_argument("--min-papers", type=int, default=3,
                    help="minimum independent papers for --agreement-report")
    args = ap.parse_args()

    db = sqlite3.connect(DATA / "papers.db")

    if args.similar_to:
        src = db.execute("SELECT title FROM papers WHERE paper_id=?",
                         (args.similar_to,)).fetchone()
        print(f"\n=== PAPERS SIMILAR TO {args.similar_to} (SPECTER2) ===")
        print(f"    source: {src[0][:70] if src else '?'}\n")
        for score, (pid, title, doi, year) in similar_papers(db, args.similar_to, args.k):
            print(f"  [{score:.3f}] {pid:13} ({year}) {(title or pid)[:58]}")
            print(f"            doi:{doi}")

    # Track 2: structured lookup. A wavenumber is a measurement, not a string --
    # this is a range query, not a similarity search. Embeddings cannot do it:
    # cosine has no notion that 1659 is near 1663.
    if args.peaks is not None:
        lo, hi = args.peaks - args.tol, args.peaks + args.tol
        by_paper = agreement_for_range(db, lo, hi)
        print(f"\n=== PEAK TABLE AGREEMENT for {args.peaks} +/-{args.tol} cm-1 ===")
        print(f"    {len(by_paper)} independent papers report a peak in this window")
        for pid, d in by_paper.items():
            print(f"\n  {pid}  {(d['title'] or pid)[:60]}  doi:{d['doi']}")
            for wl, wh, asg, origin in d["rows"]:
                band = f"{wl:.0f}" + (f"-{wh:.0f}" if wh != wl else "")
                tag = origin
                if origin:
                    ident = lipid_identity(db, origin)
                    if ident:
                        tag = f"{ident['full_name']} ({origin}, {ident['lm_id']})"
                print(f"      {band:>10} cm-1  {asg}" + (f"  [{tag}]" if tag else ""))
        if not by_paper:
            print("  (no papers report a peak in this window)")

    if args.peak_set:
        peaks = [float(x) for x in args.peak_set.split(",")]
        results = match_peak_set(db, peaks, args.tol)
        print(f"\n=== PEAK-SET MATCH for {peaks} +/-{args.tol} cm-1 ===")
        print(f"    ranked by how many observed peaks match each species' known bands\n")
        for r in results[:args.k]:
            ident = lipid_identity(db, r["origin"])
            label = f"{ident['full_name']} ({r['origin']}, {ident['main_class']})" \
                if ident else r["origin"]
            print(f"  [{r['n_matched']}/{len(peaks)}] {label}  "
                  f"({r['n_known']} known bands in corpus)")
            for peak, hits in r["matched"].items():
                pid, title, doi, wl, wh, asg = hits[0]
                band = f"{wl:.0f}" + (f"-{wh:.0f}" if wh != wl else "")
                print(f"      {peak:>7.1f} -> {band:>10} cm-1  {asg}  [{pid}]")
        if not results:
            print("  (no species in corpus has any known band matching these peaks)")

    if args.agreement_report:
        rows = agreement_report(db, args.bin_width, args.min_papers, args.k * 5)
        print(f"\n=== AGREEMENT REPORT: bands with >={args.min_papers} independent "
              f"papers ({args.bin_width:.0f} cm-1 bins) ===")
        for bin_, n_papers, n_rows, assignments in rows:
            asg = assignments[:70] + ("..." if len(assignments) > 70 else "")
            print(f"  {bin_:6.0f} cm-1   {n_papers:2} papers ({n_rows:2} rows)   {asg}")
        if not rows:
            print("  (none at this threshold -- lower --min-papers or widen --bin-width)")

    if not args.question:
        return 0

    # Track 1: semantic retrieval over prose.
    qvec = embed_query(args.question)
    max_per_paper = args.max_per_paper or 10**9
    print(f"\n=== TOP {args.k} CHUNKS for: {args.question!r} "
          f"(max {args.max_per_paper or 'unlimited'}/paper) ===")
    for score, (text, sec, xrefs, spans, title, doi, paper_id, year) in \
            search_chunks(db, qvec, args.k, pool=args.pool,
                          max_per_paper=max_per_paper):
        print(f"\n[{score:.3f}] {(title or paper_id)[:66]} ({year})")
        print(f"        {sec or '(no section)'}")
        print(f"        doi:{doi}")
        body = text if len(text) < 460 else text[:460] + " ..."
        for line in (body[i:i + 92] for i in range(0, len(body), 92)):
            print(f"    {line}")
        dois = cited_dois(db, paper_id, xrefs)
        if dois:
            print(f"        cites: {', '.join(dois[:4])}"
                  + (f" (+{len(dois)-4} more)" if len(dois) > 4 else ""))
    db.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
