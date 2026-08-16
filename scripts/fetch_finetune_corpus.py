"""Fetch a broad PubMed abstract corpus for fine-tuning -- NOT the retrieval corpus.

This is separate, on purpose, from `fetch_corpus.py` / `data/papers.db`. That
corpus is small and high-precision by design (217->281 papers, JATS full text,
every claim traceable to a DOI) -- exactly the "selection, not scale" fix that
solved this project's real retrieval problem (see docs/solutions.md, 2026-07-15
"Selection, not scale is the bottleneck"). This script does the opposite on
purpose: bulk abstracts, no full-text parsing, no citation graph, sized for
domain-adaptation fine-tuning rather than grounded retrieval. Mixing the two
would blur the "every claim resolves to a DOI in the curated corpus" guarantee
that the retrieval corpus exists to provide -- see docs/solutions.md for the
full reasoning.

Two pools, combined and deduped by PMID (see docs/PROGRESS.md for the query
rationale):
  Pool A -- bulk lipid biochemistry/lipidomics/membrane biophysics (should
            clear 100K alone).
  Pool B -- spectroscopic method vocabulary (Raman/IR/vibrational), not
            lipid-restricted on purpose: this literature constantly discusses
            Raman and IR together (shared peak-assignment/selection-rule
            language), and restricting to "Raman AND lipid" alone is exactly
            the ~4K-hit scarcity that made the retrieval corpus need PubMed
            title/MeSH selection in the first place.

NCBI esearch hard-caps retstart+retmax at 9,999 per query (confirmed live,
2026-08-14) -- neither pool's raw query returns fewer than that, so both are
paged by calendar month (each month's hit count is comfortably under the cap)
working backwards from the most recent month, until the pool's target count is
reached. This intentionally does not exhaust either pool's full multi-decade
history -- domain-adaptation fluency doesn't need it, and it would cost a lot
of NCBI traffic for no measured benefit.

Usage:
    .venv/bin/python scripts/fetch_finetune_corpus.py
    .venv/bin/python scripts/fetch_finetune_corpus.py --target-a 5000 --target-b 1000  # smoke test
"""

import argparse
import calendar
import datetime
import json
import re
import sys
import time
from pathlib import Path

import requests

EUTILS = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "data" / "finetune_corpus"
HEADERS = {"User-Agent": "lipid-raman-rag/0.1 (srikarjy@bu.edu)"}
DELAY = 0.4  # NCBI allows ~3 req/sec unkeyed; stay under it.

POOL_A_QUERY = (
    '("lipids"[MeSH] OR "lipidomics"[MeSH] OR "lipid bilayers"[MeSH] OR '
    '"membrane fluidity"[MeSH] OR "unsaturated fatty acids"[MeSH] OR '
    '"phospholipids"[MeSH]) AND hasabstract[text] AND English[Language]'
)
POOL_B_QUERY = (
    '("spectrum analysis, Raman"[MeSH] OR '
    '"spectroscopy, fourier transform infrared"[MeSH] OR '
    '"vibrational spectroscopy"[tiab]) AND hasabstract[text] AND English[Language]'
)


def _request(method: str, path: str, kw: dict, tries: int = 4) -> requests.Response:
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


def month_windows(start: datetime.date):
    """(mindate, maxdate) strings, one calendar month at a time, walking
    backwards from `start` towards 1900. Each window is comfortably under
    NCBI's 9,999-record esearch cap for these queries (measured: ~3-4K/month
    for Pool A in recent years)."""
    y, m = start.year, start.month
    while y >= 1900:
        last_day = calendar.monthrange(y, m)[1]
        yield f"{y}/{m:02d}/01", f"{y}/{m:02d}/{last_day:02d}"
        m -= 1
        if m == 0:
            m, y = 12, y - 1


def esearch_month(query: str, mindate: str, maxdate: str) -> list[str]:
    r = _request("GET", "esearch.fcgi", {"params": {
        "db": "pubmed", "term": query, "retmax": 9999, "retmode": "json",
        "mindate": mindate, "maxdate": maxdate, "datetype": "pdat",
    }})
    res = r.json()["esearchresult"]
    if "ERROR" in res:
        raise RuntimeError(f"esearch error for {mindate}..{maxdate}: {res['ERROR']}")
    return res["idlist"]


def collect_pmids(query: str, target: int, label: str) -> list[str]:
    """Walk backwards month by month until `target` unique PMIDs collected."""
    seen: list[str] = []
    seen_set: set[str] = set()
    today = datetime.date.today()
    for mindate, maxdate in month_windows(today):
        ids = esearch_month(query, mindate, maxdate)
        new = [i for i in ids if i not in seen_set]
        seen.extend(new)
        seen_set.update(new)
        time.sleep(DELAY)
        if len(seen) % 5000 < len(new):  # periodic progress, not every month
            print(f"  {label}: {len(seen)}/{target} PMIDs "
                  f"(through {mindate[:7]})")
        if len(seen) >= target:
            break
    print(f"  {label}: collected {len(seen)} PMIDs "
          f"(back to {mindate[:7] if seen else 'n/a'})")
    return seen


def _text(m: re.Match | None) -> str:
    return re.sub(r"<[^>]+>", "", m.group(1)).strip() if m else ""


def efetch_abstracts(pmids: list[str]) -> dict[str, dict]:
    """PMID -> {title, abstract, journal, year}. Regex-parsed like
    fetch_corpus.py's resolve_pmc -- consistent, already-validated approach
    for this codebase rather than pulling in a new XML dependency."""
    out: dict[str, dict] = {}
    for i in range(0, len(pmids), 200):
        batch = pmids[i:i + 200]
        xml = _request("POST", "efetch.fcgi", {"data": {
            "db": "pubmed", "id": ",".join(batch), "retmode": "xml",
        }}).text
        for art in re.findall(r"<PubmedArticle>.*?</PubmedArticle>", xml, re.S):
            pm = re.search(r"<PMID[^>]*>(\d+)</PMID>", art)
            if not pm:
                continue
            pmid = pm.group(1)
            abstract_parts = re.findall(r"<AbstractText[^>]*>(.*?)</AbstractText>",
                                         art, re.S)
            abstract = " ".join(re.sub(r"<[^>]+>", "", p).strip()
                                 for p in abstract_parts).strip()
            if not abstract:
                continue  # hasabstract[text] should guarantee this, but don't trust it silently
            out[pmid] = {
                "pmid": pmid,
                "title": _text(re.search(r"<ArticleTitle>(.*?)</ArticleTitle>", art, re.S)),
                "abstract": abstract,
                "journal": _text(re.search(r"<Title>(.*?)</Title>", art, re.S)),
                "year": _text(re.search(r"<PubDate>.*?<Year>(\d{4})</Year>", art, re.S)),
            }
        time.sleep(DELAY)
        if (i // 200) % 10 == 0:
            print(f"  efetch: {min(i + 200, len(pmids))}/{len(pmids)}")
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--target-a", type=int, default=90000)
    ap.add_argument("--target-b", type=int, default=16000)
    ap.add_argument("--max-b-fraction", type=float, default=0.20,
                     help="cap pool B's share of the final combined corpus")
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Pool A ({args.target_a} target): {POOL_A_QUERY}")
    pmids_a = collect_pmids(POOL_A_QUERY, args.target_a, "Pool A")
    print(f"\nPool B ({args.target_b} target): {POOL_B_QUERY}")
    pmids_b = collect_pmids(POOL_B_QUERY, args.target_b, "Pool B")

    set_a, set_b = set(pmids_a), set(pmids_b)
    overlap = set_a & set_b

    # Cap pool B's *unique* contribution if it would skew the mix past
    # max_b_fraction of the combined corpus, per PROGRESS.md's "sanity-check
    # the mix, cap Pool B rather than let it skew the balance" instruction.
    b_only = [p for p in pmids_b if p not in set_a]
    n_a = len(set_a)
    max_b_only = int(n_a * args.max_b_fraction / (1 - args.max_b_fraction))
    if len(b_only) > max_b_only:
        print(f"\nPool B unique contribution {len(b_only)} exceeds "
              f"{args.max_b_fraction:.0%} cap ({max_b_only}); trimming "
              f"(keeping most-recent-first order).")
        b_only = b_only[:max_b_only]

    combined_pmids = list(set_a) + b_only
    print(f"\nPool A: {len(set_a)}  Pool B: {len(set_b)}  overlap: {len(overlap)}")
    print(f"Combined (deduped, B capped): {len(combined_pmids)}")

    print("\nFetching abstracts...")
    records = efetch_abstracts(combined_pmids)
    print(f"Fetched {len(records)}/{len(combined_pmids)} "
          f"({len(combined_pmids) - len(records)} had no abstract text or failed)")

    out_path = OUT_DIR / "abstracts.jsonl"
    with out_path.open("w") as f:
        for pmid in combined_pmids:
            rec = records.get(pmid)
            if rec is None:
                continue
            rec["pool"] = "A" if pmid in set_a else "B"
            f.write(json.dumps(rec) + "\n")

    manifest = {
        "pool_a_query": POOL_A_QUERY,
        "pool_b_query": POOL_B_QUERY,
        "fetched_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "pool_a_pmids_collected": len(set_a),
        "pool_b_pmids_collected": len(set_b),
        "overlap_pmids": len(overlap),
        "pool_b_capped_to": len(b_only),
        "combined_pmids": len(combined_pmids),
        "abstracts_fetched": len(records),
    }
    (OUT_DIR / "manifest.json").write_text(json.dumps(manifest, indent=2))

    print(f"\nabstracts -> {out_path}")
    print(f"manifest  -> {OUT_DIR / 'manifest.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
