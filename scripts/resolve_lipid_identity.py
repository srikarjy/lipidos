"""Phase 3: resolve Czamara acronyms (MA, COA, PC...) to LIPID MAPS records.

Czamara's peak table uses the paper's OWN local abbreviations (Table 1 of
czamara2014.pdf), not any standard nomenclature -- "CHL" isn't a LIPID MAPS
abbreviation, it's this one paper's shorthand. So resolving them means going
through the compound's full name, not the acronym itself.

LIPID MAPS' REST API has no free-text name search (only lm_id, pubchem_cid,
inchi_key, formula, abbrev, abbrev_chains -- confirmed against the live API,
not the docs, which don't list this restriction). The two-hop chain that
works: PubChem name->CID (public, no key), then LIPID MAPS pubchem_cid->record.

SwissLipids was evaluated and deferred: no discoverable per-compound REST
endpoint, only a bulk SPA download -- disproportionate engineering for 33
lipids. See docs/solutions.md.

Results are cached in `lipid_identity` (one row per acronym) so query.py
never makes a network call at query time -- resolution is a one-time offline
step, like embedding.

Usage:
    .venv/bin/python scripts/resolve_lipid_identity.py --dry-run
    .venv/bin/python scripts/resolve_lipid_identity.py
"""

import argparse
import json
import sqlite3
import time
import urllib.parse
import urllib.request

ROOT_ACRONYM_TO_NAME = {
    # Fatty acids
    "MA": "myristic acid", "PA": "palmitic acid", "SA": "stearic acid",
    "ARA": "arachidic acid", "POA": "palmitoleic acid", "OA": "oleic acid",
    "EA": "elaidic acid", "LA": "linoleic acid", "ALA": "alpha-linolenic acid",
    "AA": "arachidonic acid",
    # Triacylglycerols
    "TCA": "tricaproin", "TCY": "tricaprylin", "TCI": "tricaprin",
    "TLU": "trilaurin", "TMA": "trimyristin", "TPA": "tripalmitin",
    "TSA": "tristearin", "TAR": "triarachidin", "TBH": "tribehenin",
    "TPO": "tripalmitolein", "TPE": "tripetroselinin", "TOA": "triolein",
    "TEA": "trielaidin", "TLA": "trilinolein", "TLN": "trilinolenin",
    "TEI": "tri-11-eicosenoin", "TER": "trierucin",
    # Sterols and cholesteryl esters
    "CHL": "cholesterol", "CPA": "cholesteryl palmitate",
    "CSA": "cholesteryl stearate", "COA": "cholesteryl oleate",
    "CLA": "cholesteryl linoleate",
    # Phospholipids
    "PC": "L-alpha-phosphatidylcholine", "PE": "L-alpha-phosphatidylethanolamine",
    "SM": "sphingomyelin",
}

DB_PATH = None  # set in main()

SCHEMA = """
CREATE TABLE IF NOT EXISTS lipid_identity (
    acronym       TEXT PRIMARY KEY,   -- Czamara's local abbreviation, e.g. "COA"
    full_name     TEXT NOT NULL,      -- Table 1 compound name, e.g. "cholesteryl oleate"
    pubchem_cid   INTEGER,
    lm_id         TEXT,               -- LIPID MAPS ID, e.g. "LMST01020003"
    lm_abbrev     TEXT,               -- LIPID MAPS' own abbreviation, e.g. "CE 18:1"
    sys_name      TEXT,
    main_class    TEXT,
    sub_class     TEXT,
    formula       TEXT,
    exactmass     REAL,
    inchi_key     TEXT,
    resolved      INTEGER NOT NULL    -- 0/1: whether both hops succeeded
);
"""


def http_json(url: str) -> dict | list | None:
    try:
        with urllib.request.urlopen(url, timeout=15) as r:
            return json.loads(r.read())
    except Exception:
        return None


def pubchem_cid(name: str) -> int | None:
    url = ("https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/"
           f"{urllib.parse.quote(name)}/cids/JSON")
    data = http_json(url)
    try:
        return data["IdentifierList"]["CID"][0]
    except (TypeError, KeyError, IndexError):
        return None


def lipidmaps_by_cid(cid: int) -> dict | None:
    url = f"https://www.lipidmaps.org/rest/compound/pubchem_cid/{cid}/all/json"
    data = http_json(url)
    if isinstance(data, dict) and data.get("lm_id"):
        return data
    return None


def lipidmaps_by_abbrev(abbrev: str) -> dict | None:
    """Some Czamara acronyms coincide with LIPID MAPS' own class abbreviation
    (PC, PE) even though the PubChem-CID chain misses them -- LIPID MAPS
    stores the class ("PC" = generic diacylglycerophosphocholine) under a
    PubChem CID that PubChem's name search for the spelled-out name doesn't
    reach. Try the acronym directly as a second strategy.
    """
    url = f"https://www.lipidmaps.org/rest/compound/abbrev/{urllib.parse.quote(abbrev)}/all/json"
    data = http_json(url)
    if isinstance(data, list) and data and data[0].get("lm_id"):
        return data[0]
    if isinstance(data, dict) and data.get("lm_id"):
        return data
    return None


def resolve_all(acronym_to_name: dict) -> list[tuple]:
    rows = []
    for acronym, name in acronym_to_name.items():
        cid = pubchem_cid(name)
        time.sleep(0.34)  # PubChem asks for <=5 req/s
        lm = lipidmaps_by_cid(cid) if cid else None
        method = "pubchem_cid"
        if not lm:
            lm = lipidmaps_by_abbrev(acronym)
            method = "abbrev" if lm else None
        if lm:
            rows.append((acronym, name, cid, lm.get("lm_id"), lm.get("abbrev"),
                         lm.get("sys_name"), lm.get("main_class"), lm.get("sub_class"),
                         lm.get("formula"), float(lm["exactmass"]) if lm.get("exactmass") else None,
                         lm.get("inchi_key"), 1))
            print(f"  OK    {acronym:5} {name:30} -> {lm.get('lm_id')}  "
                  f"({method})  {lm.get('main_class')}")
        else:
            rows.append((acronym, name, cid, None, None, None, None, None,
                         None, None, None, 0))
            reason = "no PubChem CID" if not cid else "no LIPID MAPS record via cid or abbrev"
            print(f"  MISS  {acronym:5} {name:30} -> {reason}")
    return rows


def main() -> int:
    global DB_PATH
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--db", default=None)
    args = ap.parse_args()

    from pathlib import Path
    DB_PATH = Path(args.db) if args.db else Path(__file__).resolve().parent.parent / "data" / "papers.db"

    rows = resolve_all(ROOT_ACRONYM_TO_NAME)
    n_ok = sum(r[-1] for r in rows)
    print(f"\nresolved {n_ok}/{len(rows)}")

    if args.dry_run:
        print("(dry run -- nothing written)")
        return 0

    db = sqlite3.connect(DB_PATH)
    db.executescript(SCHEMA)
    db.execute("DELETE FROM lipid_identity")
    db.executemany(
        "INSERT INTO lipid_identity VALUES (?,?,?,?,?,?,?,?,?,?,?,?)", rows)
    db.commit()
    db.close()
    print(f"-> {DB_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
