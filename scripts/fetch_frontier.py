"""Fetch the citation frontier: papers our corpus cites but does not contain.

PMC is a *biomedical* index; Raman spectroscopy is an analytical chemistry field.
Wiley/RSC/Elsevier hold 35% of what our corpus cites and PMC carries none of it --
including the Czamara review, the single most-cited missing paper (44x).

So this is track 2 of ingestion:
    papers.db references  ->  rank by citation count  ->  Unpaywall  ->  OA PDF

Ranking by how often our own corpus cites a paper beats any keyword query: it is
the field telling us what is load-bearing, rather than us guessing.

LEGAL/ETHICAL SCOPE -- do not widen without thinking:
  * Only fetches copies Unpaywall reports as `is_oa` (gold/green/hybrid/bronze).
    Closed papers are recorded as unavailable and skipped, never circumvented.
  * Identifies itself with a real contact email, as Unpaywall requires.
  * Rate-limited. Do not point this at institutional proxy access to bulk-pull
    paywalled PDFs -- that violates publisher terms and gets universities blocked.

Usage:
    python scripts/fetch_frontier.py --top 40          # 40 most-cited missing
    python scripts/fetch_frontier.py --top 40 --dry-run
"""

import argparse
import json
import sqlite3
import sys
import time
from collections import Counter
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
PDF_DIR = DATA / "pdf"
DB_PATH = DATA / "papers.db"

EMAIL = "srikarjy@bu.edu"          # Unpaywall requires an identifying contact
UA = f"lipid-raman-rag/0.1 (mailto:{EMAIL})"
UNPAYWALL = "https://api.unpaywall.org/v2/"
DELAY = 0.3

SCHEMA = """
CREATE TABLE IF NOT EXISTS frontier (
    doi          TEXT PRIMARY KEY,
    cited_by     INTEGER NOT NULL,   -- how many of our papers cite it
    title        TEXT,
    journal      TEXT,
    year         INTEGER,
    is_oa        INTEGER,
    oa_status    TEXT,               -- gold/green/hybrid/bronze/closed
    pdf_url      TEXT,
    local_pdf    TEXT,               -- path once downloaded
    fetch_error  TEXT
);
"""


def frontier(db, top: int) -> list[tuple[str, int]]:
    """DOIs our corpus cites but does not contain, most-cited first."""
    have = {r[0].lower() for r in
            db.execute("SELECT doi FROM papers WHERE doi IS NOT NULL")}
    cited = Counter(r[0].lower() for r in
                    db.execute('SELECT doi FROM "references" WHERE doi IS NOT NULL'))
    out = [(d, n) for d, n in cited.most_common() if d not in have]
    print(f"cited DOIs: {len(cited):,} unique | already held: {len(have):,} | "
          f"frontier: {len(out):,}")
    return out[:top]


def unpaywall(doi: str) -> dict:
    r = requests.get(UNPAYWALL + doi, params={"email": EMAIL},
                     headers={"User-Agent": UA}, timeout=45)
    r.raise_for_status()
    return r.json()


def best_pdf(rec: dict) -> str | None:
    """Prefer a publisher copy, then any repository copy that exposes a PDF."""
    locs = [l for l in (rec.get("oa_locations") or []) if l.get("url_for_pdf")]
    if not locs:
        return None
    locs.sort(key=lambda l: (l.get("host_type") != "publisher",
                             l.get("version") != "publishedVersion"))
    return locs[0]["url_for_pdf"]


def download(url: str, dest: Path) -> None:
    r = requests.get(url, headers={"User-Agent": UA}, timeout=90,
                     allow_redirects=True)
    r.raise_for_status()
    head = r.content[:5]
    if not head.startswith(b"%PDF"):
        # Repositories often serve an HTML landing page from a "pdf" URL. Saving
        # it would produce a file that parses to nothing and looks like a bug later.
        raise ValueError(f"not a PDF (got {head!r}, {len(r.content)}B)")
    dest.write_bytes(r.content)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--top", type=int, default=40)
    ap.add_argument("--dry-run", action="store_true",
                    help="resolve OA status, download nothing")
    args = ap.parse_args()

    PDF_DIR.mkdir(parents=True, exist_ok=True)
    db = sqlite3.connect(DB_PATH)
    db.executescript(SCHEMA)

    targets = frontier(db, args.top)
    print(f"resolving OA status for top {len(targets)} most-cited...\n")

    got = closed = failed = 0
    for i, (doi, n) in enumerate(targets, 1):
        try:
            rec = unpaywall(doi)
        except Exception as e:
            print(f"  [{n:2}x] unpaywall failed {doi}: {e}", file=sys.stderr)
            db.execute("INSERT OR REPLACE INTO frontier (doi,cited_by,fetch_error) "
                       "VALUES (?,?,?)", (doi, n, str(e)))
            failed += 1
            continue

        title = (rec.get("title") or "")[:120]
        row = dict(doi=doi, cited_by=n, title=title,
                   journal=rec.get("journal_name"),
                   year=rec.get("year"), is_oa=int(bool(rec.get("is_oa"))),
                   oa_status=rec.get("oa_status"), pdf_url=best_pdf(rec),
                   local_pdf=None, fetch_error=None)

        if not rec.get("is_oa"):
            closed += 1
            print(f"  [{n:2}x] CLOSED  {title[:56]}")
        elif not row["pdf_url"]:
            print(f"  [{n:2}x] OA but no direct PDF  {title[:44]}")
        elif args.dry_run:
            print(f"  [{n:2}x] {row['oa_status']:6} would fetch  {title[:44]}")
        else:
            dest = PDF_DIR / (doi.replace("/", "_") + ".pdf")
            if dest.exists():
                row["local_pdf"] = str(dest)
                print(f"  [{n:2}x] have    {title[:52]}")
            else:
                try:
                    download(row["pdf_url"], dest)
                    row["local_pdf"] = str(dest)
                    got += 1
                    print(f"  [{n:2}x] {row['oa_status']:6} {dest.stat().st_size//1024:5}KB  {title[:44]}")
                except Exception as e:
                    row["fetch_error"] = str(e)[:150]
                    failed += 1
                    print(f"  [{n:2}x] FAIL    {title[:40]} -- {str(e)[:44]}")
            time.sleep(DELAY)

        db.execute(
            "INSERT OR REPLACE INTO frontier "
            "(doi,cited_by,title,journal,year,is_oa,oa_status,pdf_url,local_pdf,fetch_error) "
            "VALUES (:doi,:cited_by,:title,:journal,:year,:is_oa,:oa_status,:pdf_url,"
            ":local_pdf,:fetch_error)", row)
        db.commit()
        time.sleep(DELAY)

    print(f"\ndownloaded {got} | closed (skipped, not circumvented) {closed} | failed {failed}")
    print(f"PDFs -> {PDF_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
