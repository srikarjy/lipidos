"""Parse prose from PDF papers into the same chunks table as JATS XML.

This is track 1's second ingestion path. The frontier fetcher lands PDFs from
Wiley/RSC/Elsevier that PMC has no XML for; this turns their body text into
searchable chunks. Tables are NOT handled here -- only ~3-4 of the PDFs carry
peak tables, and those get bespoke extractors (see parse_czamara.py). This file
is prose only.

Extraction: PyMuPDF get_text("blocks"). Fast (whole PDF corpus in ~13s) and,
unlike a naive y-coordinate grouping, column-aware: each block carries its
bounding box, so a two-column body can be read in the right order instead of
interleaving the columns into nonsense.

Layout is detected per page, not assumed: this corpus is a roughly even mix of
one- and two-column papers. Blocks are ordered by (column, y).

Caveats, tagged source='pdf' so retrieval quality can be measured separately:
  * PDF text carries glyph damage a JATS parser never sees -- "375 cm1" (lost
    superscript minus), "υ▷CH2◁" (upsilon for nu, bracket glyphs). We do NOT
    try to repair these; we record that the chunk came from a PDF.
  * No <xref> markup, so PDF chunks have no sentence->citation map. Their
    xref_map_json is empty; claim->DOI resolves only to the paper, not to the
    specific reference inside it.

Usage:
    .venv/bin/python scripts/parse_pdf.py --dry-run
    .venv/bin/python scripts/parse_pdf.py
"""

import argparse
import json
import re
import sqlite3
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
import fitz

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
PDF_DIR = DATA / "pdf"
DB_PATH = DATA / "papers.db"

MIN_CHUNK = 80          # match the JATS parser: drop caption stubs and one-liners
MIN_BLOCK = 40

# Everything from here down is back-matter, not body prose.
STOP = re.compile(r"^\s*(references|bibliography|acknowledg|"
                  r"conflict of interest|competing interests|"
                  r"author contributions|supporting information|"
                  r"supplementary)", re.I)

# Running heads / footers / download stamps that repeat on every page.
JUNK = re.compile(r"wileyonlinelibrary|copyright ©|©\s*\d{4}|"
                  r"downloaded from|all rights reserved|"
                  r"^\s*\d+\s*$|doi:|https?://|see discussions, stats",
                  re.I)


def order_blocks(page) -> list[str]:
    """Body text blocks in reading order, column-aware."""
    blocks = [b for b in page.get_text("blocks") if b[4].strip()]
    if not blocks:
        return []
    mid = page.rect.width / 2
    # Two-column if most blocks sit cleanly left or right of the page centre;
    # single-column if blocks span the centre.
    spanning = sum(1 for b in blocks if b[0] < mid - 20 and b[2] > mid + 20)
    two_col = spanning < len(blocks) * 0.3

    def key(b):
        if two_col:
            side = 0 if (b[0] + b[2]) / 2 < mid else 1
            return (side, round(b[1]))
        return (0, round(b[1]))

    blocks.sort(key=key)
    return [b[4] for b in blocks]


def clean(text: str) -> str:
    """De-hyphenate line breaks, collapse whitespace, drop junk lines."""
    lines = [l for l in text.splitlines() if not JUNK.search(l)]
    text = "\n".join(lines)
    text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)   # hyphenation across a line break
    text = re.sub(r"\s*\n\s*", " ", text)
    return re.sub(r"[ \t]+", " ", text).strip()


def chunks_from_pdf(path: Path) -> list[tuple[str, str]]:
    """-> [(section_hint, chunk_text)]. section_hint is best-effort from PDF."""
    doc = fitz.open(path)
    out = []
    stopped = False
    for page in doc:
        if stopped:
            break
        for block in order_blocks(page):
            head = block.strip().splitlines()[0] if block.strip() else ""
            if STOP.match(head):
                stopped = True          # references reached; ignore the rest
                break
            body = clean(block)
            if len(body) >= MIN_CHUNK:
                out.append(("", body))
    return out


def title_of(doc) -> str | None:
    t = (doc.metadata or {}).get("title")
    return t.strip() if t and len(t.strip()) > 8 else None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    db = sqlite3.connect(DB_PATH)
    # A one-time, additive column. JATS chunks default to 'jats'.
    cols = {r[1] for r in db.execute("PRAGMA table_info(chunks)")}
    if "source" not in cols:
        db.execute("ALTER TABLE chunks ADD COLUMN source TEXT DEFAULT 'jats'")
        db.commit()

    pdfs = sorted(p for p in PDF_DIR.glob("*.pdf") if p.stem != "czamara2014")
    # Czamara is handled exclusively by parse_czamara.py (peak_tables only,
    # paper_id='czamara2015'). eval_queries.py depends on it staying
    # unchunked/unembedded as a trusted reference source, not a retrievable
    # paper -- so its prose is deliberately excluded here.
    total_chunks = 0
    per_paper = []
    for pdf in pdfs:
        try:
            doc = fitz.open(pdf)
        except Exception as e:
            print(f"  UNREADABLE {pdf.name}: {e}", file=sys.stderr)
            continue
        paper_id = "pdf:" + pdf.stem
        chunks = chunks_from_pdf(pdf)
        if not chunks:
            continue
        per_paper.append((pdf.name, len(chunks)))
        total_chunks += len(chunks)

        if args.dry_run:
            continue

        db.execute(
            "INSERT OR REPLACE INTO papers (paper_id,pmid,title,abstract,doi,journal,year)"
            " VALUES (?,NULL,?,NULL,?,NULL,NULL)",
            (paper_id, title_of(doc), _doi_from_name(pdf.stem)))
        db.execute("DELETE FROM chunks WHERE paper_id=?", (paper_id,))
        for section, text in chunks:
            db.execute(
                "INSERT INTO chunks (paper_id,section_path,text,sentence_offsets_json,"
                "xref_map_json,vector_row_idx,source) VALUES (?,?,?,?,?,NULL,'pdf')",
                (paper_id, section, text, json.dumps([[0, len(text)]]), "[]"))
    if not args.dry_run:
        db.commit()

    per_paper.sort(key=lambda x: -x[1])
    print(f"{'DRY RUN: ' if args.dry_run else ''}{len(per_paper)} PDFs -> {total_chunks} prose chunks")
    print(f"  mean {total_chunks/max(len(per_paper),1):.0f} chunks/paper")
    for name, n in per_paper[:5]:
        print(f"    {n:4}  {name}")
    if not args.dry_run:
        print("\n  chunks tagged source='pdf'. Re-run embed.py to vectorise them.")
        print("  NOTE: embed.py must be updated to embed only unvectorised chunks.")
    return 0


def _doi_from_name(stem: str) -> str | None:
    # Frontier PDFs are named like "10.1038_nprot.2016.036"
    if stem.startswith("10.") and "_" in stem:
        return stem.replace("_", "/", 1)
    return None


if __name__ == "__main__":
    raise SystemExit(main())
