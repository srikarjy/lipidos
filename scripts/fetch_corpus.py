"""Fetch raw JATS XML for Raman/lipid papers from PubMed Central.

Three steps, kept separate on purpose:
  1. esearch (db=pubmed) -> PMIDs. Selection happens HERE, not in PMC.
  2. elink (pubmed->pmc)  -> PMCIDs
  3. efetch (db=pmc)      -> raw JATS XML, written to disk verbatim

Why select in PubMed and not PMC: PMC searches the entire full text, so any
paper merely mentioning "lipid" in its reference list matches -- 36,822 hits vs
PubMed's 4,316. PubMed searches title/abstract/MeSH on curated indexing. A loose
corpus is the documented failure this project exists to avoid.

Why raw XML and not the PubMed MCP tool's full_text: that field silently drops
table bodies, inline citations, superscripts and Greek symbols. The peak
assignment tables are the highest-value content in these papers. See
docs/solutions.md.

Nothing is parsed here. Raw XML is ground truth on disk, so a parser bug shipped
today stays recoverable tomorrow without re-fetching or re-hammering NCBI.

Usage:
    python scripts/fetch_corpus.py --limit 30
    python scripts/fetch_corpus.py                # whole corpus (~1256)
"""

import argparse
import json
import re
import sys
import time
from pathlib import Path

import requests

EUTILS = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "data" / "raw"

# The corpus definition lives in version control, not in shell history, so the
# exact query that produced the corpus is always recoverable.
#
# v2 (2026-07-15). v1 was "(Raman spectroscopy) AND (lipid OR ...)" -- 1256 papers
# that *use* Raman on biological samples but do not tabulate band assignments.
# Measured on a 30-paper sample: only 3 papers had peak tables, 84 of 99 rows came
# from one paper, and 2850 cm-1 (the CH2 symmetric stretch -- the most fundamental
# lipid band there is) returned nothing at all.
#
# v2 targeted papers that must report assignments to make their point:
#   - explicit assignment / reference-spectra work
#   - reviews (reviews tabulate; that is what they are for)
#   - pure lipids and model membranes (measuring DPPC means assigning its bands)
# v2's "reviews tabulate" assumption was wrong: measured on the fetched v2 corpus,
# review-only papers (matched via `Review[Publication Type]` alone, no substantive
# term) contributed peak tables at 3% (3/104) vs 11% for every other group -- and
# sampling their titles showed why: "Gut Microbiota-Derived Metabolites... in Pigs",
# "Meibomian gland dysfunction", "Dietary intakes of flavan-3-ols and cardiovascular
# health". Any review qualified by mentioning "Raman" and "lipid" ANYWHERE (title,
# abstract or MeSH), so broad biology/nutrition reviews that name-drop Raman once
# got in. See docs/solutions.md for the full diagnostic.
LIPIDS = ("(lipid OR phospholipid OR triacylglycerol OR cholesterol OR fatty acid "
          "OR sphingomyelin OR ceramide)")
ASSIGNMENT = ("(band assignment OR peak assignment OR vibrational assignment "
              "OR tentative assignment OR reference spectra OR spectral library "
              "OR marker band OR characteristic band)")
MODEL_LIPIDS = ("(model membrane OR lipid bilayer OR pure lipid OR lipid standard "
                "OR DPPC OR POPC OR liposome)")
# v3: a review only qualifies if it is actually ABOUT Raman spectroscopy of lipids --
# both terms in the title, not just present somewhere in the record. PubMed's
# [Title] tag does not stem plurals, so singular and plural are listed explicitly.
LIPIDS_TITLE = ('(lipid[Title] OR lipids[Title] OR phospholipid[Title] OR '
                 'phospholipids[Title] OR triacylglycerol[Title] OR '
                 'triacylglycerols[Title] OR cholesterol[Title] OR '
                 '"fatty acid"[Title] OR "fatty acids"[Title] OR '
                 'sphingomyelin[Title] OR ceramide[Title] OR ceramides[Title])')
REVIEW_ON_TOPIC = f"(Review[Publication Type] AND Raman[Title] AND {LIPIDS_TITLE})"
QUERY = (f"(Raman) AND {LIPIDS} AND "
         f"({ASSIGNMENT} OR {MODEL_LIPIDS} OR {REVIEW_ON_TOPIC}) "
         'AND "pubmed pmc"[sb]')

HEADERS = {"User-Agent": "lipid-raman-rag/0.1 (srikarjy@bu.edu)"}
DELAY = 0.4  # NCBI allows ~3 req/sec unkeyed; stay under it.


def _request(method: str, path: str, kw: dict, tries: int = 4) -> requests.Response:
    """Call NCBI with retry. It drops connections and 502s intermittently."""
    for attempt in range(1, tries + 1):
        try:
            r = requests.request(method, f"{EUTILS}/{path}", headers=HEADERS,
                                 timeout=60, **kw)
            r.raise_for_status()
            return r
        except requests.RequestException as e:
            if attempt == tries:
                raise
            wait = 2 ** attempt
            print(f"  retry {attempt}/{tries} after {e.__class__.__name__} "
                  f"(waiting {wait}s)", file=sys.stderr)
            time.sleep(wait)


def _get(path: str, params: dict) -> requests.Response:
    return _request("GET", path, {"params": params})


def _post(path: str, data: dict) -> requests.Response:
    return _request("POST", path, {"data": data})


def esearch(query: str, limit: int | None) -> list[str]:
    """Which papers match? Returns PMIDs, most relevant first."""
    r = _get("esearch.fcgi", {"db": "pubmed", "term": query,
                              "retmax": limit or 10000, "sort": "relevance",
                              "retmode": "json"})
    res = r.json()["esearchresult"]
    print(f"PubMed matched {res['count']} papers with PMC full text; "
          f"taking {len(res['idlist'])} (by relevance)")
    return res["idlist"]


def resolve_pmc(pmids: list[str]) -> tuple[dict[str, str], dict[str, dict]]:
    """PMID -> PMCID, plus per-paper metadata, from one efetch.

    Not using elink: it 502s and drops connections intermittently. Fetching the
    PubMed records directly is reliable and returns the PMC id in each record's
    ArticleIdList anyway -- along with title/journal/DOI, which we want regardless.
    """
    mapping: dict[str, str] = {}
    meta: dict[str, dict] = {}
    for i in range(0, len(pmids), 100):
        batch = pmids[i:i + 100]
        xml = _post("efetch.fcgi", {"db": "pubmed", "id": ",".join(batch),
                                    "retmode": "xml"}).text
        for art in re.findall(r"<PubmedArticle>.*?</PubmedArticle>", xml, re.S):
            pm = re.search(r"<PMID[^>]*>(\d+)</PMID>", art)
            pmc = re.search(r'<ArticleId IdType="pmc">(PMC\d+)</ArticleId>', art)
            if not (pm and pmc):
                continue  # no PMC copy -> nothing to fetch, drop it
            pmid = pm.group(1)
            mapping[pmid] = pmc.group(1).removeprefix("PMC")
            meta[pmid] = {
                "pmcid": pmc.group(1),
                "title": _text(re.search(r"<ArticleTitle>(.*?)</ArticleTitle>", art, re.S)),
                "journal": _text(re.search(r"<Title>(.*?)</Title>", art, re.S)),
                "doi": _text(re.search(r'<ArticleId IdType="doi">(.*?)</ArticleId>', art, re.S)),
                "year": _text(re.search(r"<PubDate>.*?<Year>(\d{4})</Year>", art, re.S)),
            }
        time.sleep(DELAY)
    print(f"resolved {len(mapping)}/{len(pmids)} PMIDs to PMC ids")
    return mapping, meta


def _text(m: re.Match | None) -> str | None:
    """Strip inline markup from a matched XML field."""
    return re.sub(r"<[^>]+>", "", m.group(1)).strip() if m else None


def efetch(pmc_id: str) -> str:
    """One paper's raw JATS XML."""
    return _get("efetch.fcgi", {"db": "pmc", "id": pmc_id,
                                "rettype": "xml"}).text


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None,
                    help="fetch only the first N papers")
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    pmids = esearch(QUERY, args.limit)
    if not pmids:
        print("no papers matched -- check QUERY", file=sys.stderr)
        return 1

    pmid_to_pmc, meta = resolve_pmc(pmids)

    fetched, skipped, failed = 0, 0, []
    for i, (pmid, pmc_id) in enumerate(pmid_to_pmc.items(), 1):
        dest = OUT_DIR / f"PMC{pmc_id}.xml"
        if dest.exists():
            skipped += 1
            continue
        try:
            dest.write_text(efetch(pmc_id), encoding="utf-8")
            fetched += 1
        except Exception as e:
            # Fail loud per paper; one bad response must not kill the run.
            failed.append({"pmid": pmid, "pmc_id": pmc_id, "error": str(e)})
            print(f"  FAILED PMC{pmc_id}: {e}", file=sys.stderr)
        if i % 10 == 0:
            print(f"  {i}/{len(pmid_to_pmc)} (fetched {fetched}, skipped {skipped})")
        time.sleep(DELAY)

    # What the corpus actually is, recorded next to the corpus itself.
    manifest = {
        "query": QUERY,
        "fetched_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "sort": "relevance",
        "papers": meta,
        "counts": {"matched": len(pmids), "linked": len(pmid_to_pmc),
                   "fetched": fetched, "skipped": skipped, "failed": len(failed)},
        "failures": failed,
    }
    (OUT_DIR.parent / "manifest.json").write_text(json.dumps(manifest, indent=2))

    print(f"\ndone: {fetched} fetched, {skipped} already present, {len(failed)} failed")
    print(f"raw XML  -> {OUT_DIR}")
    print(f"manifest -> {OUT_DIR.parent / 'manifest.json'}")
    return 1 if failed and not fetched else 0


if __name__ == "__main__":
    raise SystemExit(main())
