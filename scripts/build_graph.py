"""Phase 3b: derive a Neo4j knowledge graph from papers.db.

papers.db stays canonical -- this is a rebuildable, derived layer, same
pattern as resolve_lipid_identity.py's cache: SQLite is the source of truth,
this script re-derives the graph from it, and the graph can always be wiped
and rebuilt from scratch rather than migrated in place.

Runs against Neo4j AuraDB Free (cloud), not local Docker. The 8GB M2 Air
already fights for RAM -- MLX inference (scripts/answer.py) runs outside
Docker for the same reason.

Nodes:
  Paper       {doi UNIQUE, paper_id, title, journal, year, in_corpus}
              union of `papers` (in_corpus=true) and `frontier`
              (in_corpus=false) -- frontier triples the citation graph's
              connectivity for free (references resolving to frontier.doi
              outnumber references resolving to in-corpus papers.doi).
  Lipid       {acronym UNIQUE, full_name, lm_id, lm_abbrev, formula,
               exactmass, inchi_key} -- from lipid_identity where resolved=1.
  LipidClass, LipidSubclass  {name UNIQUE} -- from lipid_identity's
              main_class/sub_class.

Relationships:
  (Paper)-[:CITES]->(Paper)                          from `references`, by DOI
  (Paper)-[:REPORTS_PEAK {wavenumber_low,
                           wavenumber_high,
                           assignment}]->(Lipid)      from `peak_tables`
  (Lipid)-[:IN_CLASS]->(LipidClass)
  (Lipid)-[:IN_SUBCLASS]->(LipidSubclass)

Deliberately not built: chunk-level "Paper MENTIONS Lipid" edges (would need
free-text matching against lipid names -- see docs/questions.md).

Usage:
    .venv/bin/python scripts/build_graph.py --dry-run
    .venv/bin/python scripts/build_graph.py
"""

import argparse
import os
import sqlite3
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
BATCH = 500


def fetch_papers(db) -> list[dict]:
    rows = db.execute(
        """SELECT paper_id, doi, title, journal, year FROM papers
           WHERE doi IS NOT NULL AND doi != ''""").fetchall()
    papers = [dict(doi=doi, paper_id=pid, title=title, journal=journal,
                   year=year, in_corpus=True)
              for pid, doi, title, journal, year in rows]
    in_corpus_dois = {p["doi"] for p in papers}

    rows = db.execute(
        """SELECT doi, title, journal, year FROM frontier
           WHERE doi IS NOT NULL AND doi != ''""").fetchall()
    for doi, title, journal, year in rows:
        if doi in in_corpus_dois:
            continue
        papers.append(dict(doi=doi, paper_id=None, title=title,
                            journal=journal, year=year, in_corpus=False))
    return papers


def fetch_citations(db) -> list[dict]:
    """(citing paper's doi) -> (cited doi), restricted to both ends
    resolving to a Paper node this build actually creates (papers or
    frontier) -- a reference to a DOI outside both tables would just be a
    dangling MATCH at insert time, so filter it out here instead."""
    rows = db.execute(
        """SELECT p.doi AS src_doi, r.doi AS dst_doi
           FROM "references" r
           JOIN papers p ON r.paper_id = p.paper_id
           WHERE r.doi IS NOT NULL AND r.doi != ''
                 AND p.doi IS NOT NULL AND p.doi != ''
                 AND (
                     EXISTS (SELECT 1 FROM papers p2 WHERE p2.doi = r.doi)
                     OR EXISTS (SELECT 1 FROM frontier f WHERE f.doi = r.doi)
                 )""").fetchall()
    return [dict(src=src, dst=dst) for src, dst in rows]


def fetch_lipids(db) -> list[dict]:
    rows = db.execute(
        """SELECT acronym, full_name, lm_id, lm_abbrev, main_class, sub_class,
                  formula, exactmass, inchi_key
           FROM lipid_identity WHERE resolved = 1""").fetchall()
    return [dict(acronym=a, full_name=fn, lm_id=lm, lm_abbrev=ab,
                  main_class=mc, sub_class=sc, formula=f, exactmass=em,
                  inchi_key=ik)
            for a, fn, lm, ab, mc, sc, f, em, ik in rows]


def fetch_peak_edges(db) -> list[dict]:
    rows = db.execute(
        """SELECT pa.doi, p.origin, p.wavenumber_low, p.wavenumber_high, p.assignment
           FROM peak_tables p
           JOIN lipid_identity li ON li.acronym = p.origin AND li.resolved = 1
           JOIN papers pa ON pa.paper_id = p.paper_id
           WHERE p.origin IS NOT NULL AND p.origin != ''
                 AND pa.doi IS NOT NULL AND pa.doi != ''""").fetchall()
    return [dict(doi=doi, acronym=o, wavenumber_low=wl, wavenumber_high=wh,
                  assignment=asg) for doi, o, wl, wh, asg in rows]


def load_db(db_path: Path) -> dict:
    db = sqlite3.connect(db_path)
    data = dict(
        papers=fetch_papers(db),
        citations=fetch_citations(db),
        lipids=fetch_lipids(db),
        peak_edges=fetch_peak_edges(db),
    )
    db.close()
    return data


CONSTRAINTS = [
    "CREATE CONSTRAINT paper_doi IF NOT EXISTS FOR (p:Paper) REQUIRE p.doi IS UNIQUE",
    "CREATE CONSTRAINT lipid_acronym IF NOT EXISTS FOR (l:Lipid) REQUIRE l.acronym IS UNIQUE",
    "CREATE CONSTRAINT class_name IF NOT EXISTS FOR (c:LipidClass) REQUIRE c.name IS UNIQUE",
    "CREATE CONSTRAINT subclass_name IF NOT EXISTS FOR (s:LipidSubclass) REQUIRE s.name IS UNIQUE",
]

MERGE_PAPERS = """
UNWIND $rows AS row
MERGE (p:Paper {doi: row.doi})
SET p.paper_id = row.paper_id, p.title = row.title, p.journal = row.journal,
    p.year = row.year, p.in_corpus = row.in_corpus
"""

MERGE_CITES = """
UNWIND $rows AS row
MATCH (src:Paper {doi: row.src}), (dst:Paper {doi: row.dst})
MERGE (src)-[:CITES]->(dst)
"""

MERGE_LIPIDS = """
UNWIND $rows AS row
MERGE (l:Lipid {acronym: row.acronym})
SET l.full_name = row.full_name, l.lm_id = row.lm_id, l.lm_abbrev = row.lm_abbrev,
    l.formula = row.formula, l.exactmass = row.exactmass, l.inchi_key = row.inchi_key
MERGE (c:LipidClass {name: row.main_class})
MERGE (s:LipidSubclass {name: row.sub_class})
MERGE (l)-[:IN_CLASS]->(c)
MERGE (l)-[:IN_SUBCLASS]->(s)
"""

MERGE_PEAK_EDGES = """
UNWIND $rows AS row
MATCH (p:Paper {doi: row.doi}), (l:Lipid {acronym: row.acronym})
MERGE (p)-[:REPORTS_PEAK {wavenumber_low: row.wavenumber_low,
                           wavenumber_high: row.wavenumber_high,
                           assignment: row.assignment}]->(l)
"""


def batched(rows: list, size: int = BATCH):
    for i in range(0, len(rows), size):
        yield rows[i:i + size]


def build(driver, data: dict) -> None:
    with driver.session() as session:
        session.run("MATCH (n) DETACH DELETE n")
        for stmt in CONSTRAINTS:
            session.run(stmt)
        for batch in batched(data["papers"]):
            session.run(MERGE_PAPERS, rows=batch)
        for batch in batched(data["lipids"]):
            session.run(MERGE_LIPIDS, rows=batch)
        for batch in batched(data["citations"]):
            session.run(MERGE_CITES, rows=batch)
        for batch in batched(data["peak_edges"]):
            session.run(MERGE_PEAK_EDGES, rows=batch)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--db", default=None)
    args = ap.parse_args()

    db_path = Path(args.db) if args.db else ROOT / "data" / "papers.db"
    data = load_db(db_path)

    print(f"Paper nodes:      {len(data['papers'])} "
          f"({sum(p['in_corpus'] for p in data['papers'])} in_corpus, "
          f"{sum(not p['in_corpus'] for p in data['papers'])} frontier-only)")
    print(f"CITES edges:      {len(data['citations'])}")
    print(f"Lipid nodes:      {len(data['lipids'])}")
    print(f"REPORTS_PEAK edges: {len(data['peak_edges'])}")

    if args.dry_run:
        print("(dry run -- nothing written to Neo4j)")
        return 0

    load_dotenv(ROOT / ".env")
    uri = os.environ["NEO4J_URI"]
    user = os.environ["NEO4J_USERNAME"]
    password = os.environ["NEO4J_PASSWORD"]

    from neo4j import GraphDatabase
    driver = GraphDatabase.driver(uri, auth=(user, password))
    driver.verify_connectivity()
    build(driver, data)
    driver.close()
    print("-> Neo4j AuraDB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
