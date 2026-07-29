"""Phase 3b: query the Neo4j knowledge graph built by build_graph.py.

Two traversals that are awkward or impossible to phrase in SQL against
papers.db directly -- everything else (general visual exploration of how
papers/lipids/classes relate) is better done in the Aura console's own
Neo4j Browser than in a custom CLI.

Usage:
    .venv/bin/python scripts/graph_query.py --lipid-neighbors COA
    .venv/bin/python scripts/graph_query.py --citation-path 10.1000/xyz 10.1000/abc
"""

import argparse
import os

from dotenv import load_dotenv
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

LIPID_NEIGHBORS = """
MATCH (l:Lipid {acronym: $acronym})-[:IN_CLASS]->(c:LipidClass)
MATCH (l)-[:IN_SUBCLASS]->(s:LipidSubclass)
OPTIONAL MATCH (p:Paper)-[:REPORTS_PEAK]->(l)
WITH l, c, s, collect(DISTINCT p.title) AS papers
OPTIONAL MATCH (l)<-[:REPORTS_PEAK]-(shared:Paper)-[:REPORTS_PEAK]->(other:Lipid)
WHERE other <> l
WITH l, c, s, papers, other, count(DISTINCT shared) AS shared_papers
WHERE shared_papers >= 1
RETURN l.full_name AS full_name, c.name AS main_class, s.name AS sub_class,
       papers, collect({acronym: other.acronym, full_name: other.full_name,
                         shared_papers: shared_papers}) AS related_lipids
"""

CITATION_PATH = """
MATCH (src:Paper {doi: $src}), (dst:Paper {doi: $dst}),
      path = shortestPath((src)-[:CITES*..6]->(dst))
RETURN [n IN nodes(path) | {doi: n.doi, title: n.title, year: n.year}] AS chain
"""


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--lipid-neighbors", metavar="ACRONYM", default=None,
                     help="class/subclass + papers reporting it + lipids sharing peak reports")
    ap.add_argument("--citation-path", nargs=2, metavar=("FROM_DOI", "TO_DOI"),
                     default=None, help="shortest citation path between two papers")
    args = ap.parse_args()

    if not args.lipid_neighbors and not args.citation_path:
        ap.error("pass --lipid-neighbors or --citation-path")

    load_dotenv(ROOT / ".env")
    from neo4j import GraphDatabase
    driver = GraphDatabase.driver(
        os.environ["NEO4J_URI"],
        auth=(os.environ["NEO4J_USERNAME"], os.environ["NEO4J_PASSWORD"]))

    with driver.session() as session:
        if args.lipid_neighbors:
            rec = session.run(LIPID_NEIGHBORS, acronym=args.lipid_neighbors).single()
            if not rec:
                print(f"  {args.lipid_neighbors} not found in graph "
                      f"(unresolved acronym or no shared peak reports)")
            else:
                print(f"\n=== {rec['full_name']} ({args.lipid_neighbors}) ===")
                print(f"    class:    {rec['main_class']}")
                print(f"    subclass: {rec['sub_class']}")
                print(f"    reported by {len(rec['papers'])} paper(s):")
                for t in rec["papers"]:
                    print(f"      - {t[:70]}")
                print(f"    shares peak-report papers with:")
                for r in sorted(rec["related_lipids"],
                                 key=lambda r: -r["shared_papers"]):
                    print(f"      - {r['full_name']} ({r['acronym']}): "
                          f"{r['shared_papers']} shared paper(s)")

        if args.citation_path:
            src, dst = args.citation_path
            rec = session.run(CITATION_PATH, src=src, dst=dst).single()
            if not rec:
                print(f"  no citation path found from {src} to {dst} (within 6 hops)")
            else:
                print(f"\n=== CITATION PATH {src} -> {dst} ===")
                for n in rec["chain"]:
                    print(f"  {n['doi']}  ({n['year']})  {(n['title'] or '')[:70]}")

    driver.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
