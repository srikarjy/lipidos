"""Fetch Raman/lipid preprints from arXiv (free API, full PDFs, no paywall).

Scope note, measured 2026-07-15: arXiv holds only ~29 Raman+lipid papers and ~95
Raman+membrane. It is a physics/CS archive and this field mostly is not there --
so this is a small supplementary source, not a fix for the corpus gap. It is
included because the physics-side papers (vibrational calculations, instrument
methods) are exactly what PubMed's biomedical index misses, and they are free.

Usage:
    python scripts/fetch_arxiv.py --limit 40
"""

import argparse
import re
import time
from pathlib import Path

import requests
from defusedxml import ElementTree

ROOT = Path(__file__).resolve().parent.parent
PDF_DIR = ROOT / "data" / "pdf"
API = "http://export.arxiv.org/api/query"
UA = {"User-Agent": "lipid-raman-rag/0.1 (mailto:srikarjy@bu.edu)"}
NS = {"a": "http://www.w3.org/2005/Atom"}

QUERIES = [
    'all:raman AND all:lipid',
    'all:raman AND all:"fatty acid"',
    'abs:raman AND abs:membrane',
    'all:raman AND all:cholesterol',
]


def search(q: str, n: int) -> list[dict]:
    r = requests.get(API, params={"search_query": q, "max_results": n,
                                  "sortBy": "relevance"}, headers=UA, timeout=60)
    r.raise_for_status()
    root = ElementTree.fromstring(r.text)
    out = []
    for e in root.findall("a:entry", NS):
        aid = e.find("a:id", NS).text.rsplit("/", 1)[-1]
        pdf = next((l.get("href") for l in e.findall("a:link", NS)
                    if l.get("title") == "pdf"), None)
        out.append({"id": aid,
                    "title": " ".join(e.find("a:title", NS).text.split()),
                    "pdf": pdf})
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=40)
    args = ap.parse_args()
    PDF_DIR.mkdir(parents=True, exist_ok=True)

    seen, got = {}, 0
    for q in QUERIES:
        try:
            for e in search(q, args.limit):
                seen.setdefault(e["id"], e)
        except Exception as ex:
            print(f"  search failed {q}: {ex}")
        time.sleep(3)          # arXiv asks for >=3s between calls

    print(f"{len(seen)} unique arXiv papers found")
    for e in list(seen.values())[:args.limit]:
        if not e["pdf"]:
            continue
        dest = PDF_DIR / f"arxiv_{e['id'].replace('/', '_')}.pdf"
        if dest.exists():
            continue
        try:
            r = requests.get(e["pdf"], headers=UA, timeout=90)
            r.raise_for_status()
            if not r.content[:5].startswith(b"%PDF"):
                raise ValueError("not a PDF")
            dest.write_bytes(r.content)
            got += 1
            print(f"  {dest.stat().st_size//1024:5}KB  {e['title'][:58]}")
        except Exception as ex:
            print(f"  FAIL {e['id']}: {str(ex)[:50]}")
        time.sleep(3)
    print(f"\ndownloaded {got} arXiv PDFs -> {PDF_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
