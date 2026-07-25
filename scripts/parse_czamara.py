"""Extract Table 2 of Czamara et al. 2015, "Raman spectroscopy of lipids: a review".

    K. Czamara, K. Majzner, M.Z. Pacia, K. Kochan, A. Kaczor, M. Baranska
    J. Raman Spectrosc. 2015;46:4-20.  doi:10.1002/jrs.4607

The single most-cited paper in our corpus's citation frontier (44x), tabulating
35 lipids against their vibrational modes. Unreachable via PubMed/PMC/Europe PMC
(Wiley analytical chemistry journal); its Unpaywall "green OA" record at
Jagiellonian University is metadata-only, so the PDF was acquired manually.

Extraction: pdfplumber, text strategy, text_x_tolerance=1.
------------------------------------------------------------
The tolerance is the whole trick. At the default (3) pdfplumber merges adjacent
columns -- myristic acid reads '2943 2909' in ONE cell, fusing two different
vibrational modes. At 1 every value lands in its own cell AND empty cells are
preserved, which is what matters: the blanks are the chemistry. Saturated fatty
acids have no C=C, so ν(C=C) is empty; collapsing the row left-to-right would
assign a C=C stretch to a saturated lipid.

    MA   [Fatty acids, MA, ·, ·, 2943, 2909, 2869, ·, 2832, ·, ·, ·, 1457, ...]
    POA  [·, POA, ·, 3005, ·, 2924, 2895, ·, 2849, ·, 1655, ·, ·, 1442, ·]

(Do not hand-roll word-coordinate geometry for this. It was tried, took far
longer, and produced the same numbers. pymupdf's find_tables() finds nothing
here; camelot stream also works, at 99.8% reported accuracy.)

Column labels are hardcoded per region rather than parsed: the printed header
stacks the Greek mode letter on a separate visual line from its formula, so
extraction yields fragments like '(=Cs' / 'H )2'. The modes are a fixed,
published list -- reading them off the paper is more reliable than recovering
them from broken glyph runs, and it preserves ν/δ/τ, which every text extractor
in this pipeline strips.

Table 2 spans two pages with different layouts:
    page 10: Group | Acronym | Regions I-III   -- carries lipid names
    page 11: Regions IV-VII                    -- NUMBERS ONLY, no names
Page 11 carries lipid identity only by row order, matching page 10. Both yield
exactly 35 data rows, so they zip by index -- asserted, not assumed.

Validation: cell-for-cell against an independent extraction (Scholar Gateway's
markdown of the same table) agreed 200/200 on the overlapping 20 lipids.

Usage:
    .venv/bin/python scripts/parse_czamara.py --dry-run
    .venv/bin/python scripts/parse_czamara.py
"""

import argparse
import re
import sqlite3
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
import pdfplumber

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
PDF = DATA / "pdf" / "czamara2014.pdf"
DB_PATH = DATA / "papers.db"

PAPER_ID = "czamara2015"
DOI = "10.1002/jrs.4607"
TITLE = "Raman spectroscopy of lipids: a review"
JOURNAL = "Journal of Raman Spectroscopy"
YEAR = "2015"

SETTINGS = {
    "vertical_strategy": "text",
    "horizontal_strategy": "text",
    "text_x_tolerance": 1,          # 3 (the default) fuses adjacent mode columns
    "intersection_x_tolerance": 1,
}

# Column -> mode, by position, as printed in the paper. `None` = a spacer column
# that pdfplumber emits but the table does not use.
PAGE_NAMED = 9
# Verified against the printed header, which extracts as:
#   [Group, Acronym, ·, (=CH), (=CH)as3, (=CH)s3, (=Cs, H)2, (=CH)s2,
#    (C = O), (C = C), (C, H/CH)23, (CH/CH)23, (CH)2]
# idx 6/7 are one mode (νas(=CH2)) split across two columns; idx 11 likewise.
MODES_NAMED = [
    None, None, None,               # Group, Acronym, spacer
    "ν(=CH)", "νas(=CH3)", "νs(=CH3)", "νas(=CH2)", None, "νs(=CH2)",
    "ν(C=O)", "ν(C=C)", None, "β(CH2/CH3)", "α(CH2/CH3)", "β(CH2)",
]
PAGE_ANON = 10
MODES_ANON = [
    None, "τ(CH2)", None, "δ(=CH)", "ν(C-C)", "ν(P-O)", None, None,
    "β(CH)", "ν(C-O-O)skel", None, None, None,
    "νasN+(CH3)3", None, "νsN+(CH3)3", None,
]

NUM = re.compile(r"\d{3,4}")


def data_rows(table) -> list[list]:
    """Rows that are actually table data: >=2 cells holding wavenumbers."""
    out = []
    for r in table:
        n = sum(1 for c in r if c and NUM.fullmatch(str(c).strip().split(",")[0].strip()))
        if n >= 2:
            out.append(r)
    return out


def cells_from(row, modes) -> list[tuple[str, float]]:
    """(mode, wavenumber) for each populated cell. One cell may list several
    bands ('1743, 1729' -- the ester C=O Fermi doublet); emit each."""
    out = []
    for i, c in enumerate(row):
        if not c or i >= len(modes) or modes[i] is None:
            continue
        for m in NUM.finditer(str(c)):
            wn = float(m.group(0))
            if 100 <= wn <= 4000:
                out.append((modes[i], wn))
    return out


def extract() -> tuple[list[str], list[tuple[str, str, float]]]:
    with pdfplumber.open(PDF) as pdf:
        t_named = pdf.pages[PAGE_NAMED].extract_tables(SETTINGS)[0]
        t_anon = pdf.pages[PAGE_ANON].extract_tables(SETTINGS)[0]

    named = data_rows(t_named)
    anon = data_rows(t_anon)

    # On rows where a new group starts, the group label bleeds into the acronym
    # cell ('rols TCA', 'ol ester CPA'). The acronym is the trailing token.
    def acronym(cell) -> str | None:
        toks = re.findall(r"[A-Z]{2,4}", str(cell or ""))
        return toks[-1] if toks else None

    lipids = [acronym(r[1]) for r in named]
    if None in lipids:
        raise SystemExit(f"no acronym found in rows: "
                         f"{[r[1] for r, l in zip(named, lipids) if l is None]}")
    if len(named) != len(anon):
        # Page 11 has no lipid names; zipping mismatched rows would attach
        # cholesterol's bands to a triacylglycerol. Refuse rather than guess.
        raise SystemExit(f"row mismatch: {len(named)} named vs {len(anon)} anonymous "
                         f"-- refusing to zip by index")

    cells = []
    for lipid, row in zip(lipids, named):
        cells += [(lipid, m, wn) for m, wn in cells_from(row, MODES_NAMED)]
    for lipid, row in zip(lipids, anon):
        cells += [(lipid, m, wn) for m, wn in cells_from(row, MODES_ANON)]
    return lipids, cells


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    if not PDF.exists():
        print(f"missing {PDF}")
        return 1

    lipids, cells = extract()
    modes = sorted({c[1] for c in cells})
    print(f"extracted {len(cells)} cells | {len(lipids)} lipids | {len(modes)} modes")
    print(f"  lipids: {' '.join(lipids)}")
    print()
    for m in modes:
        print(f"  {sum(1 for c in cells if c[1] == m):3} x  {m}")

    if args.dry_run:
        print("\n(dry run -- nothing written)")
        return 0

    db = sqlite3.connect(DB_PATH)
    db.execute("INSERT OR REPLACE INTO papers (paper_id,pmid,title,abstract,doi,journal,year)"
               " VALUES (?,NULL,?,NULL,?,?,?)",
               (PAPER_ID, TITLE, DOI, JOURNAL, YEAR))
    db.execute("DELETE FROM peak_tables WHERE paper_id=?", (PAPER_ID,))
    db.executemany(
        "INSERT INTO peak_tables (paper_id,table_label,wavenumber_low,wavenumber_high,"
        "assignment,origin,reference) VALUES (?,?,?,?,?,?,?)",
        [(PAPER_ID, "Table 2", wn, wn, mode, lipid, DOI) for lipid, mode, wn in cells])
    db.commit()
    n = db.execute("SELECT COUNT(*) FROM peak_tables WHERE paper_id=?",
                   (PAPER_ID,)).fetchone()[0]
    print(f"\nwrote {n} peak rows for {PAPER_ID} -> {DB_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
