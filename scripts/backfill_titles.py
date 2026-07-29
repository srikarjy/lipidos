"""Backfill titles for pdf:-sourced papers from Crossref/arXiv, not PDF metadata.

`parse_pdf.py` reads title from PDF metadata (`title_of()`) and trusts it
unconditionally. Two distinct failure shapes surfaced once the Colab re-embed
made these chunks retrievable (see docs/solutions.md):
  - NULL title (23 papers) -- PDF had no metadata title at all. Crashed
    query.py/answer.py's string slicing the first time one ranked top-k.
  - Junk-but-non-NULL title (9+ more papers) -- PDF metadata *had* a title
    field, it just wasn't the paper's title: "Microsoft Word - ....docx",
    "No Job Name", bare manuscript IDs ("ac500014b 1..5"), even a raw
    "doi:10.1016/..." string. These looked like real strings and would not
    have been caught by a NULL check -- the exact "output looks plausible,
    nothing errors, it's just quietly wrong" failure mode this project's own
    docs (PROGRESS.md problem #1) describe for text extraction generally.

Fix: don't trust PDF-internal title metadata as authoritative for any pdf:
paper. Always attempt the real bibliographic source first, keep the existing
(possibly-junk) title only if that lookup fails:
  - has a DOI  -> Crossref `/works/{doi}` (public, no key).
  - arXiv-only (paper_id like "pdf:arxiv_0704.2669v1", no DOI)
    -> arXiv API `id_list` lookup by the id embedded in the filename.

Usage:
    .venv/bin/python scripts/backfill_titles.py --dry-run
    .venv/bin/python scripts/backfill_titles.py
"""

import argparse
import re
import sqlite3
import time
from pathlib import Path

import requests
from defusedxml import ElementTree

ROOT = Path(__file__).resolve().parent.parent
UA = {"User-Agent": "lipid-raman-rag/0.1 (mailto:srikarjy@bu.edu)"}
NS = {"a": "http://www.w3.org/2005/Atom"}


def crossref_title(doi: str) -> str | None:
    try:
        r = requests.get(f"https://api.crossref.org/works/{doi}",
                         headers=UA, timeout=15)
        r.raise_for_status()
        titles = r.json()["message"].get("title")
        return titles[0] if titles else None
    except Exception:
        return None


def arxiv_title(arxiv_id: str) -> str | None:
    try:
        r = requests.get("http://export.arxiv.org/api/query",
                         params={"id_list": arxiv_id}, headers=UA, timeout=15)
        r.raise_for_status()
        root = ElementTree.fromstring(r.text)
        entry = root.find("a:entry", NS)
        if entry is None:
            return None
        t = entry.find("a:title", NS)
        return " ".join(t.text.split()) if t is not None and t.text else None
    except Exception:
        return None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--db", default=None)
    args = ap.parse_args()

    db_path = Path(args.db) if args.db else ROOT / "data" / "papers.db"
    db = sqlite3.connect(db_path)
    rows = db.execute(
        "SELECT paper_id, doi, title FROM papers WHERE paper_id LIKE 'pdf:%'"
    ).fetchall()

    print(f"{len(rows)} pdf:-sourced papers (re-deriving all titles from "
          f"Crossref/arXiv, not trusting PDF metadata)")
    updates = []
    for paper_id, doi, old_title in rows:
        if doi:
            title = crossref_title(doi)
            time.sleep(0.5)  # polite pool
            source = "crossref"
        else:
            m = re.match(r"pdf:arxiv_(.+)", paper_id)
            arxiv_id = m.group(1) if m else None
            title = arxiv_title(arxiv_id) if arxiv_id else None
            time.sleep(3)  # arXiv asks for >=3s between calls
            source = "arxiv"
        if title and title != old_title:
            tag = "NEW" if old_title is None else "FIX"
            print(f"  {tag}   {paper_id:40} {old_title!r} -> {title[:60]!r}  ({source})")
            updates.append((title, paper_id))
        elif title:
            print(f"  OK    {paper_id:40} unchanged  ({source})")
        else:
            print(f"  MISS  {paper_id:40} old={old_title!r}  ({source}, no title found -- kept as-is)")

    print(f"\n{len(updates)}/{len(rows)} titles updated")

    if args.dry_run:
        print("(dry run -- nothing written)")
        db.close()
        return 0

    db.executemany("UPDATE papers SET title=? WHERE paper_id=?", updates)
    db.commit()
    db.close()
    print(f"-> {db_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
