# LipidOS

A grounded literature + spectral retrieval system for Raman spectroscopy of
lipids. Every claim it surfaces resolves to a DOI — no answer is generated
without pointing back to the specific evidence it came from.

Sole engineer: architecture, data pipeline, evaluation. Started 2026-07-15.

## Why this exists

A prior attempt at this project — "download a model, embed 50-60k papers" —
couldn't be reproduced, debugged, or handed off, because none of the
reasoning behind its decisions was ever written down. This project's
discipline exists specifically to not repeat that: every decision is logged
with its reasoning in [`docs/solutions.md`](docs/solutions.md), every number
in [`docs/PROGRESS.md`](docs/PROGRESS.md) was measured, not estimated, and
raw source data is stored separately from parsed output so re-parsing never
requires re-fetching.

## What it does

Three independent retrieval tracks over the Raman/lipid literature, backed
by a SQLite store (`data/papers.db`, gitignored — rebuild with the scripts
below):

1. **Prose track** — JATS XML → section-aware paragraph chunks → BGE
   (`bge-base-en-v1.5`) embeddings → cosine search. Each chunk carries
   sentence offsets and a map of which inline citation falls in which
   sentence, so a retrieved claim resolves claim → sentence → reference → DOI.
2. **Peak table track** — spectral assignment tables (wavenumber →
   vibrational mode → source) parsed into structured rows and queried by
   wavenumber **range**, never embedded — cosine similarity has no notion
   that 1659 cm⁻¹ is adjacent to 1663.
3. **Paper track** — SPECTER2 document embeddings for paper-level similarity
   (citation-graph-aware, not text similarity).

Plus **peak-set matching**: given several observed peaks, ranks candidate
lipid species by how many match that species' known bands — the actual
bench question ("I measured 2850, 2880, 1440, 1660 — what is this?"), not a
single-band lookup.

Plus a **lipid identity layer**: resolves the Czamara review's local
acronyms (`COA`, `PC`...) to real LIPID MAPS records (classification,
formula, exact mass) via a PubChem-name → CID → LIPID MAPS chain, cached
offline so query time never hits the network.

Plus a **base-model answer layer**: Phi-3.5 Mini (4-bit, via `mlx-lm`,
local inference on Apple Silicon) generates an answer from retrieved
evidence, printed in a separate panel that is never merged with the raw
evidence — every generated sentence must cite which evidence item it came
from, checked automatically against the evidence actually retrieved.

Plus a **fine-tuned model**: Phi-3.5 Mini, QLoRA-adapted (Unsloth) for
domain fluency on 107,665 lipid/Raman-spectroscopy abstracts — held-out
perplexity 4.911 → 3.955 (19.5% lower), citation-grounding verified to
survive the fine-tune. Public on the Hub:
[adapter](https://huggingface.co/srikarjy025/lipidos-phi3-domain-adapt) ·
[merged standalone model](https://huggingface.co/srikarjy025/lipidos-phi3-domain-adapt-merged).
Wired end-to-end with the retrieval + citation-checking pipeline via
[`scripts/build_evidence.py`](scripts/build_evidence.py) +
[`scripts/generate_finetuned_answer.py`](scripts/generate_finetuned_answer.py).

## Status

Build phases from [`lipid-raman-rag-blueprint.md`](lipid-raman-rag-blueprint.md):

| Phase | Status |
|---|---|
| 1 — Paper ingestion | Done — 217 papers, 3 tracks, documented model+chunking |
| 2 — Paper QA (base model, no Raman yet) | Done — retrieval, citation grounding, and Phi-3.5 Mini answer layer all validated |
| 3 — Lipid knowledge layer | Done (LIPID MAPS) — 29/35 Czamara acronyms resolved; SwissLipids evaluated and deferred (no queryable API) |
| 4 — Context Builder | Done — `context_builder.py` merges prose + peak + paper-similarity tracks into deduped, cited evidence |
| 5 — Raman integration (peaks → CNN/PCA → context) | Interface + demo done (`raman_integration.py`); real PCA/CNN peak extraction stays external, as designed |
| 6 — Fine-tuning | Done — QLoRA (Unsloth) domain-adaptation fine-tune of Phi-3.5 Mini on 64,000 lipid/Raman abstracts; held-out perplexity 4.911 → 3.955 (19.5% lower); wired end-to-end with Context Builder + citation-checking via `build_evidence.py` + `generate_finetuned_answer.py` |

Full details, measured numbers, and the reasoning behind every non-obvious
decision: [`docs/PROGRESS.md`](docs/PROGRESS.md) (status) and
[`docs/solutions.md`](docs/solutions.md) (decision log).

## Setup

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

Raw paper XML/PDFs and the built database live under `data/` (gitignored —
not checked in). Rebuild the pipeline:

```bash
.venv/bin/python scripts/parse_jats.py           # data/raw/*.xml -> papers.db
.venv/bin/python scripts/parse_czamara.py        # Czamara review peak table
.venv/bin/python scripts/resolve_lipid_identity.py  # acronyms -> LIPID MAPS
.venv/bin/python scripts/embed.py                # prose + paper embeddings
```

## Usage

```bash
# Semantic search over prose, with citations
.venv/bin/python scripts/query.py "how does the 1440 band shift in unsaturated lipids?"

# Single-wavenumber peak lookup, cross-paper agreement
.venv/bin/python scripts/query.py --peaks 1655 --tol 5

# Peak-set matching: which lipid has bands at all of these positions?
.venv/bin/python scripts/query.py --peak-set "2846,3009,1670,1739"

# Papers similar to a given one (SPECTER2, citation-graph-aware)
.venv/bin/python scripts/query.py --similar-to PMC10670390

# Two-panel answer: raw evidence + Phi-3.5 Mini's cited interpretation
.venv/bin/python scripts/answer.py "how is the 1655/1440 ratio used to quantify unsaturation?"

# Evaluation: mined real bench questions + constructed + reference tiers
.venv/bin/python scripts/eval_queries.py
```

## Project structure

```
scripts/
  fetch_corpus.py, fetch_frontier.py, fetch_arxiv.py   ingestion
  filter_corpus.py                                     corpus selection
  parse_jats.py, parse_czamara.py, parse_pdf.py         parsing (raw -> papers.db)
  embed.py                                              prose + paper embeddings
  resolve_lipid_identity.py                             acronym -> LIPID MAPS
  query.py                                              3-track retrieval + peak-set matching
  answer.py                                             Phase 2: base-model answer layer
  eval_queries.py                                       evaluation harness
  mine_bench_questions.py                                mines real eval questions from corpus papers
docs/
  PROGRESS.md      status, measured numbers, hard problems solved
  solutions.md     decision log: question, decision, why
  questions.md     open, unresolved questions
lipid-raman-rag-blueprint.md   architecture / plan of record (6-phase build)
```
