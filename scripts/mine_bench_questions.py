"""Mine candidate eval questions from papers' own stated research goals.

A paper's Introduction routinely states its own motivation in a sentence like
"we sought to distinguish X from Y" or "this study aimed to quantify Z". That
sentence is a real question, written by a real domain scientist -- and it
comes with a free ground truth: the paper that wrote it should be a top hit
when we search for it.

This does NOT produce a finished eval set. It produces ranked candidates for
a human to accept or reject -- most goal-statement sentences are too vague or
too specific to make a good query. Accepted candidates get a `known_paper_id`
so eval_queries.py can measure recall@k, not just eyeball a score.

Usage:
    .venv/bin/python scripts/mine_bench_questions.py             # print candidates
    .venv/bin/python scripts/mine_bench_questions.py --out FILE  # write JSON
"""

import argparse
import json
import re
import sqlite3
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DB_PATH = ROOT / "data" / "papers.db"

# Strongest intent markers first -- a sentence matching "aimed to" is a much
# more reliable goal statement than one merely containing "investigate".
PATTERNS = [
    (3, re.compile(r"\b(?:we |this study |the (?:present |current )?(?:study|work|paper) )?"
                    r"(?:aim(?:ed|s)? to|sought to|set out to)\b", re.I)),
    (3, re.compile(r"\b(?:to )?(?:distinguish|differentiate|discriminate) between\b", re.I)),
    (2, re.compile(r"\bin order to (?:distinguish|differentiate|discriminate|quantify|"
                    r"determine|assess|evaluate|characterize|elucidate)\b", re.I)),
    (2, re.compile(r"\bdetermine whether\b", re.I)),
    (2, re.compile(r"\b(?:the )?(?:goal|objective|purpose) of (?:this|the) (?:study|work|paper) "
                    r"(?:was|is) to\b", re.I)),
    (1, re.compile(r"\bwe (?:investigated|quantified|assessed|evaluated|characterized|examined)\b", re.I)),
]

MIN_LEN, MAX_LEN = 60, 320


def sentences_of(text: str, spans: list[list[int]]) -> list[str]:
    return [text[s:e].strip() for s, e in spans]


def score_sentence(s: str) -> int:
    best = 0
    for weight, pat in PATTERNS:
        if pat.search(s):
            best = max(best, weight)
    return best


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path)
    ap.add_argument("--per-paper", type=int, default=1, help="max candidates per paper")
    ap.add_argument("--limit", type=int, default=40)
    a = ap.parse_args()

    db = sqlite3.connect(DB_PATH)
    rows = db.execute(
        "SELECT c.paper_id, p.title, p.doi, c.text, c.sentence_offsets_json "
        "FROM chunks c JOIN papers p ON p.paper_id = c.paper_id "
        "WHERE c.section_path LIKE '%introdu%'"
    ).fetchall()

    candidates = []
    for paper_id, title, doi, text, spans_json in rows:
        spans = json.loads(spans_json)
        for s in sentences_of(text, spans):
            if not (MIN_LEN <= len(s) <= MAX_LEN):
                continue
            score = score_sentence(s)
            if score == 0:
                continue
            candidates.append(dict(score=score, paper_id=paper_id, title=title,
                                    doi=doi, sentence=s))

    candidates.sort(key=lambda c: -c["score"])

    seen_papers: dict[str, int] = {}
    kept = []
    for c in candidates:
        n = seen_papers.get(c["paper_id"], 0)
        if n >= a.per_paper:
            continue
        seen_papers[c["paper_id"]] = n + 1
        kept.append(c)
        if len(kept) >= a.limit:
            break

    if a.out:
        a.out.write_text(json.dumps(kept, indent=2))
        print(f"wrote {len(kept)} candidates -> {a.out}")
    else:
        for c in kept:
            print(f"[{c['score']}] {c['paper_id']}  {c['title'][:60]}")
            print(f"    {c['sentence']}\n")
        print(f"{len(kept)} candidates from {len(seen_papers)} papers "
              f"({len(rows)} intro chunks scanned)")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
