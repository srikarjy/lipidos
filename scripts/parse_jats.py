"""Parse JATS XML papers into papers.db (no embeddings -- see embed.py).

Three tracks, one database:
  1. prose   -> chunks table      (one row per <p>, with sentence + xref maps)
  2. tables  -> peak_tables table (structured spectral rows, never embedded)
  3. papers  -> papers table      (title/abstract feed SPECTER2 later)

Plus a references table: an <xref rid="B12"> is meaningless without resolving
B12 to a real citation, and claim -> DOI is the point of the whole project.

Parsing is separated from embedding on purpose: parsing is fast and
deterministic, embedding is slow. Splitting them means we can re-parse without
paying for re-embedding, and vice versa.

Usage:
    python scripts/parse_jats.py                 # data/raw/*.xml -> data/papers.db
    python scripts/parse_jats.py --rebuild       # drop and rebuild from scratch
"""

import argparse
import json
import re
import sqlite3
import sys
from pathlib import Path

from defusedxml import ElementTree  # untrusted network XML: entity-expansion defense

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
RAW_DIR = DATA / "raw"
DB_PATH = DATA / "papers.db"

SCHEMA = """
CREATE TABLE IF NOT EXISTS papers (
    paper_id   TEXT PRIMARY KEY,      -- PMCID, e.g. PMC10670390
    pmid       TEXT,
    title      TEXT,
    abstract   TEXT,
    doi        TEXT,
    journal    TEXT,
    year       TEXT
);
CREATE TABLE IF NOT EXISTS chunks (
    chunk_id              INTEGER PRIMARY KEY,
    paper_id              TEXT NOT NULL REFERENCES papers(paper_id),
    section_path          TEXT,       -- "Materials and methods > Raman spectroscopy"
    text                  TEXT NOT NULL,
    sentence_offsets_json TEXT,       -- [[start,end], ...] into text
    xref_map_json         TEXT,       -- [{"rid":"B12","sentence":2,"offset":140}, ...]
    vector_row_idx        INTEGER     -- join key into chunk_vectors.npy
);
CREATE TABLE IF NOT EXISTS peak_tables (
    row_id         INTEGER PRIMARY KEY,
    paper_id       TEXT NOT NULL REFERENCES papers(paper_id),
    table_label    TEXT,
    wavenumber_low  REAL,
    wavenumber_high REAL,
    assignment     TEXT,
    origin         TEXT,
    reference      TEXT
);
CREATE TABLE IF NOT EXISTS "references" (
    ref_id    TEXT NOT NULL,          -- rid, e.g. B12 -- matches xref_map_json
    paper_id  TEXT NOT NULL REFERENCES papers(paper_id),
    citation  TEXT,
    doi       TEXT,
    PRIMARY KEY (ref_id, paper_id)
);
CREATE TABLE IF NOT EXISTS paper_embeddings (
    paper_id       TEXT PRIMARY KEY REFERENCES papers(paper_id),
    vector_row_idx INTEGER
);
CREATE INDEX IF NOT EXISTS idx_chunks_paper ON chunks(paper_id);
CREATE INDEX IF NOT EXISTS idx_peaks_range  ON peak_tables(wavenumber_low, wavenumber_high);
"""

# ---------------------------------------------------------------- text helpers

SUP = str.maketrans("0123456789+-−=()n", "⁰¹²³⁴⁵⁶⁷⁸⁹⁺⁻⁻⁼⁽⁾ⁿ")
SUB = str.maketrans("0123456789+-−=()", "₀₁₂₃₄₅₆₇₈₉₊₋₋₌₍₎")

# Protect these from the sentence splitter. "cm-1." and "et al." and "0.5" all
# contain a period that does not end a sentence; naive splitting on '.' would
# put citations in the wrong sentence, which defeats sentence-level attribution.
ABBREV = r"(?:et al|e\.g|i\.e|cf|vs|approx|ca|Fig|Figs|Ref|Refs|Eq|Tab|No|Dr|Prof|St|Inc|Ltd|etc|resp|min|sec|hr|wt|vol|conc|temp|max|std|dev|Suppl)"


def render(el, drop_citations: bool = False) -> tuple[str, list[dict]]:
    """Flatten a JATS element to text, recording where each citation lands.

    Returns (text, xrefs) where xrefs is [{"rid":..., "offset":...}].
    We cannot use itertext(): it discards positions, and we need the character
    offset of every <xref> to attribute it to a sentence later.

    Superscripts/subscripts are rendered as real unicode, so "cm<sup>-1</sup>"
    becomes "cm-1" rather than the silently-corrupted "cm" that naive .text
    parsing produces (see docs/solutions.md).
    """
    parts: list[str] = []
    xrefs: list[dict] = []
    pos = 0

    def emit(s: str) -> None:
        nonlocal pos
        if s:
            parts.append(s)
            pos += len(s)

    def walk(node) -> None:
        emit(node.text or "")
        for child in node:
            tag = child.tag.split("}")[-1]
            if tag == "xref" and child.get("ref-type") == "bibr":
                for rid in (child.get("rid") or "").split():
                    xrefs.append({"rid": rid, "offset": pos})
                # In table cells the citation marker's digits fuse onto the
                # wavenumber: "850-880" + citation "8" reads as "850-8808".
                # Callers parsing numbers out of cells drop the marker text.
                if not drop_citations:
                    emit("".join(child.itertext()))
            elif tag in ("sup", "sub"):
                # ACS/Nature style renders citations as superscript numbers:
                # <sup><xref ref-type="bibr">34</xref></sup>. Flattening with
                # itertext() here would silently swallow the citation. Recurse
                # instead whenever a real citation is nested inside.
                if child.find(".//xref[@ref-type='bibr']") is not None \
                        or (child.get("ref-type") == "bibr"):
                    walk(child)  # recurse: keeps citation capture, honours drop
                else:
                    inner = "".join(child.itertext())
                    table = SUP if tag == "sup" else SUB
                    mapped = inner.translate(table)
                    # If a char had no unicode form, fall back to explicit
                    # notation rather than dropping it.
                    emit(mapped if mapped != inner or inner.isalpha() else f"^{inner}")
            elif tag in ("italic", "bold", "underline", "sc", "monospace",
                         "named-content", "styled-content"):
                walk(child)
            elif tag in ("disp-formula", "inline-formula", "tex-math", "mml:math"):
                emit(" ")
            else:
                walk(child)
            emit(child.tail or "")

    walk(el)

    # Whitespace-normalise while remapping offsets. Doing the substitution after
    # the walk would change the string length and silently drift every xref
    # offset -- misattributing citations to the wrong sentence with no error.
    raw = "".join(parts)
    out: list[str] = []
    remap = [0] * (len(raw) + 1)
    prev_space = False
    for i, ch in enumerate(raw):
        remap[i] = len(out)
        if ch.isspace():
            if not prev_space:
                out.append(" ")
                prev_space = True
        else:
            out.append(ch)
            prev_space = False
    remap[len(raw)] = len(out)

    text = "".join(out)
    lead = len(text) - len(text.lstrip())
    text = text.strip()
    for x in xrefs:
        x["offset"] = max(0, min(remap[x["offset"]] - lead, len(text)))
    return text, xrefs


def split_sentences(text: str) -> list[tuple[int, int]]:
    """Sentence boundaries as (start, end) offsets.

    Not a naive split on '.': decimals (0.5), abbreviations (et al., Fig.) and
    unit notation all carry periods that do not end sentences. Getting this
    wrong misattributes citations to neighbouring sentences.
    """
    # Mask periods that must not split: decimals, abbreviations, initials.
    # Every substitution is length-preserving (1 char -> 1 char) so that offsets
    # computed against `protected` remain valid against `text`.
    protected = text
    protected = re.sub(r"(\d)\.(\d)", lambda m: f"{m.group(1)}\x00{m.group(2)}", protected)
    protected = re.sub(rf"\b{ABBREV}\.", lambda m: m.group(0).replace(".", "\x00"),
                       protected, flags=re.I)
    protected = re.sub(r"\b([A-Z])\.", lambda m: f"{m.group(1)}\x00", protected)

    spans, start = [], 0
    for m in re.finditer(r"[.!?]+[\)\]\"']*\s+", protected):
        end = m.start() + len(m.group(0).rstrip())
        if end > start:
            spans.append((start, end))
        start = m.end()
    if start < len(text):
        spans.append((start, len(text)))
    return spans or [(0, len(text))]


def sentence_of(offset: int, spans: list[tuple[int, int]]) -> int:
    for i, (s, e) in enumerate(spans):
        if s <= offset <= e:
            return i
    return len(spans) - 1


# ---------------------------------------------------------------- front matter

def _find_text(root, path: str) -> str | None:
    el = root.find(path)
    if el is None:
        return None
    return render(el)[0] or None


def parse_front(root) -> dict:
    meta = root.find(".//front/article-meta")
    if meta is None:
        return {}
    ids = {e.get("pub-id-type"): (e.text or "").strip()
           for e in meta.findall("article-id")}
    abstract_el = meta.find("abstract")
    return {
        "pmid": ids.get("pmid"),
        "pmcid": "PMC" + ids["pmc"] if ids.get("pmc") else None,
        "doi": ids.get("doi"),
        "title": _find_text(meta, "title-group/article-title"),
        "abstract": render(abstract_el)[0] if abstract_el is not None else None,
        "journal": _find_text(root, ".//front/journal-meta/journal-title-group/journal-title")
                   or _find_text(root, ".//front/journal-meta/journal-title"),
        "year": _find_text(meta, "pub-date/year"),
    }


def parse_refs(root, paper_id: str) -> list[tuple]:
    """rid -> citation text + DOI, so xref markers resolve to something real."""
    out = []
    for ref in root.findall(".//back//ref"):
        rid = ref.get("id")
        if not rid:
            continue
        doi = None
        for pid in ref.findall(".//pub-id"):
            if pid.get("pub-id-type") == "doi":
                doi = (pid.text or "").strip()
        out.append((rid, paper_id, render(ref)[0][:600], doi))
    return out


# ---------------------------------------------------------------- prose track

SKIP_SECTIONS = re.compile(
    r"^(references|acknowledg|funding|conflict|competing|author contribution|"
    r"supplementary|data availability|abbreviation|ethic|consent)", re.I)


def walk_sections(sec, trail: list[str]) -> list[tuple[str, str, list[dict]]]:
    """Yield (section_path, paragraph_text, xrefs) for every <p>, depth-first."""
    title_el = sec.find("title")
    title = render(title_el)[0] if title_el is not None else ""
    if title and SKIP_SECTIONS.match(title):
        return []
    path = trail + ([title] if title else [])
    out = []
    for child in sec:
        tag = child.tag.split("}")[-1]
        if tag == "p":
            text, xrefs = render(child)
            if len(text) >= 80:  # drop figure-caption stubs and one-liners
                out.append((" > ".join(path), text, xrefs))
        elif tag == "sec":
            out.extend(walk_sections(child, path))
    return out


# ---------------------------------------------------------------- table track

PEAK_HDR = re.compile(r"wavenumber|raman shift|wave number|band position|peak position|"
                      r"cm\s*[-−⁻]\s*1|frequency", re.I)
ASSIGN_HDR = re.compile(r"assign|vibration|mode|attribution", re.I)
REF_HDR = re.compile(r"^ref|source", re.I)
ORIGIN_HDR = re.compile(r"origin|molecule|compound|biomarker|component|species", re.I)

# "347-353", "1440", "~1663", "1437 - 1442", "3013+/-2"
NUM_RANGE = re.compile(r"(\d{2,4}(?:\.\d+)?)\s*(?:[-–—−]\s*(\d{2,4}(?:\.\d+)?))?")


def cells(row, drop_citations: bool = False) -> list[str]:
    """Cell text, with colspan expanded.

    Peak tables routinely use a spanning header ("Normalized Raman intensity"
    over three group columns). Without expanding colspan the header has fewer
    cells than the data rows, so every column index after the span is silently
    off -- which reads intensity values as assignments.
    """
    out = []
    for c in row:
        if c.tag.split("}")[-1] not in ("td", "th"):
            continue
        text = render(c, drop_citations=drop_citations)[0]
        try:
            span = max(1, int(c.get("colspan", 1)))
        except ValueError:
            span = 1
        out.extend([text] * span)
    return out


def parse_peak_table(tw, paper_id: str) -> list[tuple]:
    """Extract spectral rows, but only from tables that actually are peak tables."""
    rows = tw.findall(".//tr")
    if not rows:
        return []

    header, hdr_idx = None, 0
    for i, r in enumerate(rows[:3]):  # header is near the top, if it exists
        c = cells(r)
        if c and PEAK_HDR.search(" | ".join(c)) and ASSIGN_HDR.search(" | ".join(c)):
            header, hdr_idx = c, i
            break
    if header is None:
        return []  # not a peak assignment table -- leave it alone

    def col(pat, default=None):
        for j, h in enumerate(header):
            if pat.search(h):
                return j
        return default

    wn_c, as_c = col(PEAK_HDR), col(ASSIGN_HDR)
    ref_c, or_c = col(REF_HDR), col(ORIGIN_HDR)
    if wn_c is None or as_c is None:
        return []

    label_el = tw.find(".//label")
    label = render(label_el)[0] if label_el is not None else tw.get("id")

    out = []
    for r in rows[hdr_idx + 1:]:
        c = cells(r)
        wn_cells = cells(r, drop_citations=True)  # digits must not fuse onto bands
        if len(c) <= max(wn_c, as_c) or len(wn_cells) <= wn_c:
            continue
        assignment = c[as_c].strip()
        # An assignment names a vibrational mode ("δ(CH₂)"). A cell with no
        # letters is a measurement that landed in the wrong column -- a garbage
        # row in the evidence store is worse than a missing one, because the
        # RAG panel is meant to be checkable.
        if not assignment or not re.search(r"[A-Za-zα-ωΑ-Ω]", assignment):
            continue
        # One cell can list several bands ("3013, 1663"); emit a row per band so
        # a range query can find each one.
        for m in NUM_RANGE.finditer(wn_cells[wn_c]):
            lo = float(m.group(1))
            hi = float(m.group(2)) if m.group(2) else lo
            # Validate BOTH bounds. Checking only `lo` let "850-8808" through --
            # a citation digit fused onto the band's upper bound.
            if not (100 <= lo <= 4000 and 100 <= hi <= 4000 and hi >= lo):
                continue
            out.append((paper_id, label, lo, hi, assignment,
                        c[or_c].strip() if or_c is not None and len(c) > or_c else None,
                        c[ref_c].strip() if ref_c is not None and len(c) > ref_c else None))
    return out


# ---------------------------------------------------------------- driver

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rebuild", action="store_true", help="drop existing db first")
    ap.add_argument("--raw-dir", type=Path, default=RAW_DIR)
    args = ap.parse_args()

    if args.rebuild:
        DB_PATH.unlink(missing_ok=True)
        # The vector files are keyed to chunk rows by vector_row_idx. Dropping
        # the db orphans them: queries would then silently return nothing (or,
        # worse, match stale rows) instead of failing. Delete them together.
        for f in ("chunk_vectors.npy", "paper_vectors.npy"):
            p = DATA / f
            if p.exists():
                p.unlink()
                print(f"  removed stale {f} (rebuild invalidates vectors -- re-run embed.py)")

    db = sqlite3.connect(DB_PATH)
    db.executescript(SCHEMA)

    files = sorted(args.raw_dir.glob("*.xml"))
    if not files:
        print(f"no XML in {args.raw_dir}", file=sys.stderr)
        return 1

    n_chunks = n_peaks = n_refs = 0
    skipped: list[str] = []

    for f in files:
        try:
            root = ElementTree.parse(f).getroot()
        except Exception as e:
            print(f"  UNPARSEABLE {f.name}: {e}", file=sys.stderr)
            skipped.append(f.name)
            continue

        front = parse_front(root)
        paper_id = front.get("pmcid") or f.stem

        body = root.find(".//body")
        if body is None:
            # PMC has metadata but no licensed full text. Embedding a stub would
            # quietly pollute retrieval, so record it and move on.
            print(f"  NO BODY  {paper_id} ({f.stat().st_size // 1024}KB) -- abstract-only record")
            skipped.append(paper_id)
            continue

        db.execute("INSERT OR REPLACE INTO papers VALUES (?,?,?,?,?,?,?)",
                   (paper_id, front.get("pmid"), front.get("title"),
                    front.get("abstract"), front.get("doi"),
                    front.get("journal"), front.get("year")))

        refs = parse_refs(root, paper_id)
        db.executemany('INSERT OR REPLACE INTO "references" VALUES (?,?,?,?)', refs)
        n_refs += len(refs)

        paras = []
        for sec in body.findall("sec"):
            paras.extend(walk_sections(sec, []))
        for p in body.findall("p"):  # paragraphs before any <sec>
            text, xrefs = render(p)
            if len(text) >= 80:
                paras.append(("", text, xrefs))

        for section_path, text, xrefs in paras:
            spans = split_sentences(text)
            xmap = [{"rid": x["rid"], "offset": x["offset"],
                     "sentence": sentence_of(x["offset"], spans)} for x in xrefs]
            db.execute(
                "INSERT INTO chunks (paper_id, section_path, text, "
                "sentence_offsets_json, xref_map_json, vector_row_idx) "
                "VALUES (?,?,?,?,?,NULL)",
                (paper_id, section_path, text, json.dumps(spans), json.dumps(xmap)))
            n_chunks += 1

        for tw in root.findall(".//table-wrap"):
            rows = parse_peak_table(tw, paper_id)
            db.executemany(
                "INSERT INTO peak_tables (paper_id, table_label, wavenumber_low, "
                "wavenumber_high, assignment, origin, reference) VALUES (?,?,?,?,?,?,?)",
                rows)
            n_peaks += len(rows)

    db.commit()

    n_papers = db.execute("SELECT COUNT(*) FROM papers").fetchone()[0]
    print(f"\nparsed {n_papers}/{len(files)} papers"
          f"{f' ({len(skipped)} skipped: no body/unparseable)' if skipped else ''}")
    print(f"  chunks      {n_chunks}")
    print(f"  peak rows   {n_peaks}")
    print(f"  references  {n_refs}")
    print(f"  -> {DB_PATH}")
    db.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
