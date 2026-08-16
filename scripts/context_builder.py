"""Phase 4: Context Builder — merges all retrieval tracks into a cited evidence block.

This is the layer that makes Phase 5/6 input clean:
- dedupes overlapping evidence across tracks
- ranks by relevance + independent agreement
- attaches citations per claim (paper → sentence → DOI)
- summarizes long passages to a fixed token budget
- outputs a single structured evidence block for the LLM panel

Usage:
    .venv/bin/python scripts/context_builder.py "question" [--peaks 2850,2880,1440,1660] [-k 5]
"""

import argparse
import json
import sqlite3
import sys
import textwrap
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"

BGE_MODEL = "BAAI/bge-base-en-v1.5"
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "

MAX_EVIDENCE_TOKENS = 3500
MAX_PER_PAPER = 2
PEAK_TOL = 5.0


def embed_query(q: str) -> np.ndarray:
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(BGE_MODEL)
    model = AutoModel.from_pretrained(BGE_MODEL).to(dev).eval()
    with torch.inference_mode():
        batch = tok([QUERY_PREFIX + q], padding=True, truncation=True,
                    max_length=512, return_tensors="pt").to(dev)
        v = model(**batch).last_hidden_state[:, 0]
        return F.normalize(v, p=2, dim=1).cpu().numpy()[0].astype(np.float32)


def search_prose(db, qvec, k, pool=None, max_per_paper=MAX_PER_PAPER):
    """Track 1: semantic prose search (from query.py)"""
    if pool is None:
        pool = max(k * 10, 50)
    vecs = np.load(DATA / "chunk_vectors.npy")
    scores = vecs @ qvec

    picked = []
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

    return picked


def cited_dois(db, paper_id: str, xref_json: str) -> list[str]:
    out = []
    for rid in {x["rid"] for x in json.loads(xref_json or "[]")}:
        r = db.execute('SELECT doi FROM "references" WHERE ref_id=? AND paper_id=?',
                       (rid, paper_id)).fetchone()
        if r and r[0]:
            out.append(r[0])
    return out


def get_peak_evidence(db, peaks: list[float], tol: float = PEAK_TOL):
    """Track 2: peak table lookup + agreement counts + lipid identity"""
    from query import agreement_for_range, match_peak_set, lipid_identity

    peak_ranges = [(p - tol, p + tol) for p in peaks]
    all_papers = {}
    peak_to_papers = {}

    for lo, hi in peak_ranges:
        by_paper = agreement_for_range(db, lo, hi)
        for pid, d in by_paper.items():
            if pid not in all_papers:
                all_papers[pid] = d
            for wl, wh, asg, origin in d["rows"]:
                peak_to_papers.setdefault(f"{wl}-{wh}", []).append(pid)

    peak_set_results = match_peak_set(db, peaks, tol)

    return {
        "by_paper": all_papers,
        "peak_to_papers": peak_to_papers,
        "peak_set_matches": peak_set_results,
        "lipid_identity": {r["origin"]: lipid_identity(db, r["origin"])
                          for r in peak_set_results}
    }


def get_paper_similarity(db, paper_id: str, k: int = 5):
    """Track 3: SPECTER2 paper-level similarity"""
    from query import similar_papers
    return list(similar_papers(db, paper_id, k))


def format_peak_evidence(peak_ev: dict) -> list[dict]:
    """Convert peak evidence to unified evidence items"""
    items = []
    evidence_id = 0

    # Peak-set matches (species-level)
    for r in peak_ev["peak_set_matches"]:
        evidence_id += 1
        ident = peak_ev["lipid_identity"].get(r["origin"])
        label = f"{ident['full_name']} ({r['origin']}, {ident['main_class']})" \
            if ident else r["origin"]

        lines = [f"Peak-set match: {label} — {r['n_matched']}/{r['n_known']} observed peaks match"]
        for peak, hits in r["matched"].items():
            pid, title, doi, wl, wh, asg = hits[0]
            band = f"{wl:.0f}" + (f"-{wh:.0f}" if wh != wl else "")
            lines.append(f"  {peak:.1f} cm⁻¹ → {band} cm⁻¹  {asg}  [{pid}]")

        items.append(dict(
            id=evidence_id,
            track="peak_set",
            source="Czamara 2015 peak table",
            paper_id=r["origin"],
            title=label,
            text="\n".join(lines),
            score=r["n_matched"] * 1000 + r["n_known"] + r["n_matched"] / max(r["n_known"], 1),  # prioritize matched count, then known count
            citations=[]
        ))

    # Per-peak agreement (paper-level)
    for key, pids in peak_ev["peak_to_papers"].items():
        if len(pids) < 2:
            continue
        evidence_id += 1
        lines = [f"Cross-paper agreement at {key} cm⁻¹: {len(pids)} independent papers"]
        for pid in pids[:5]:
            d = peak_ev["by_paper"].get(pid, {})
            title = d.get("title", pid)
            lines.append(f"  {pid}: {title}")

        items.append(dict(
            id=evidence_id,
            track="peak_agreement",
            source="peak_tables",
            paper_id=",".join(pids),
            title=f"Agreement: {key} cm⁻¹",
            text="\n".join(lines),
            score=0.5,
            citations=[]
        ))

    return items


def format_prose_evidence(prose_hits: list[dict], db) -> list[dict]:
    """Convert prose hits to unified evidence items"""
    items = []
    for i, h in enumerate(prose_hits, 1):
        dois = cited_dois(db, h["paper_id"], h["xrefs"])
        items.append(dict(
            id=i,
            track="prose",
            source="paper prose",
            paper_id=h["paper_id"],
            title=h["title"],
            text=h["text"],
            section=h["sec"],
            score=h["score"],
            year=h["year"],
            doi=h["doi"],
            citations=dois
        ))
    return items


def format_paper_similarity(sim_hits: list[tuple]) -> list[dict]:
    """Convert paper similarity hits to unified evidence items"""
    items = []
    for i, (score, (pid, title, doi, year)) in enumerate(sim_hits, 1):
        items.append(dict(
            id=i + 100,
            track="paper_similarity",
            source="SPECTER2 citation-graph similarity",
            paper_id=pid,
            title=title,
            text=f"Paper similar to query paper: {title} ({year})",
            score=score,
            year=year,
            doi=doi,
            citations=[]
        ))
    return items


def dedupe_evidence(items: list[dict]) -> list[dict]:
    """Remove near-duplicate evidence items"""
    seen_texts = set()
    unique = []
    for item in items:
        text_sig = item["text"][:200]
        if text_sig not in seen_texts:
            seen_texts.add(text_sig)
            unique.append(item)
    return unique


def rank_evidence(items: list[dict]) -> list[dict]:
    """Rank by: peak_set > peak_agreement > prose > paper_similarity, then by score DESC"""
    track_priority = {"peak_set": 0, "peak_agreement": 1, "prose": 2, "paper_similarity": 3}
    return sorted(items, key=lambda x: (track_priority.get(x["track"], 99), -x["score"]))


def summarize_text(text: str, max_chars: int = 500) -> str:
    """Summarize long passages to token budget"""
    if len(text) <= max_chars:
        return text
    sentences = text.split(". ")
    summary = ""
    for s in sentences:
        if len(summary) + len(s) + 2 > max_chars:
            break
        summary += s + ". "
    return summary.strip() + " ..."


def build_context_block(question: str, peaks: list[float] = None,
                         k: int = 5, paper_id: str = None) -> dict:
    """Main entry: builds the complete context block for the LLM panel"""
    db = sqlite3.connect(DATA / "papers.db")
    qvec = embed_query(question)

    # Track 1: prose retrieval
    prose_hits = search_prose(db, qvec, k)
    prose_items = format_prose_evidence(prose_hits, db)

    # Track 2: peak evidence (if peaks provided)
    peak_items = []
    if peaks:
        peak_ev = get_peak_evidence(db, peaks)
        peak_items = format_peak_evidence(peak_ev)

    # Track 3: paper similarity (if paper_id provided)
    paper_items = []
    if paper_id:
        sim_hits = get_paper_similarity(db, paper_id, k)
        paper_items = format_paper_similarity(sim_hits)

    # Merge, dedupe, rank
    all_items = prose_items + peak_items + paper_items
    all_items = dedupe_evidence(all_items)
    all_items = rank_evidence(all_items)

    # Assign sequential evidence IDs
    for i, item in enumerate(all_items, 1):
        item["evidence_id"] = i

    # Build formatted evidence block for LLM
    evidence_lines = []
    total_chars = 0
    for item in all_items:
        block = f"[{item['evidence_id']}] {item['title']} ({item['paper_id']})\n{item['text']}"
        if total_chars + len(block) > MAX_EVIDENCE_TOKENS * 4:
            block = block[:MAX_EVIDENCE_TOKENS * 4] + " ..."
        evidence_lines.append(block)
        total_chars += len(block)
        if total_chars > MAX_EVIDENCE_TOKENS * 4:
            break

    db.close()

    return dict(
        question=question,
        peaks=peaks,
        evidence_items=all_items,
        evidence_block="\n\n".join(evidence_lines),
        n_evidence=len(all_items)
    )


def main() -> int:
    ap = argparse.ArgumentParser(description="Build grounded context for LLM panel")
    ap.add_argument("question", help="Natural language question")
    ap.add_argument("--peaks", type=str, default=None,
                    help="Comma-separated observed peaks, e.g. 2850,2880,1440,1660")
    ap.add_argument("-k", type=int, default=5, help="Results per track")
    ap.add_argument("--paper-id", type=str, default=None,
                    help="Paper ID for SPECTER2 similarity track")
    ap.add_argument("--json", action="store_true", help="Output raw JSON")
    args = ap.parse_args()

    peaks = [float(x) for x in args.peaks.split(",")] if args.peaks else None

    ctx = build_context_block(args.question, peaks, args.k, args.paper_id)

    if args.json:
        print(json.dumps(ctx, indent=2))
        return 0

    print(f"\n=== CONTEXT BLOCK for: {args.question!r} ===")
    if peaks:
        print(f"Peaks: {peaks} ±{PEAK_TOL} cm⁻¹")
    print(f"Total evidence items: {ctx['n_evidence']}\n")
    print(ctx["evidence_block"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())