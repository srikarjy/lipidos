# Lipid Raman Research Assistant — Progress

Status doc. Every number here was measured, not estimated. Decisions and their
reasoning live in `solutions.md`; open problems in `questions.md`.

---

**Project/Role:** Grounded literature + spectral retrieval system for Raman
spectroscopy of lipids. Sole engineer — architecture, data pipeline, evaluation.

**Dates:** 2026-07-15 → present. Phase 1 (ingestion) complete; Phase 2 (paper QA)
complete — retrieval, citation grounding, and the base-model answer layer
(Phi-3.5 Mini via MLX) all validated. Phase 3 (LIPID MAPS knowledge layer)
complete, SwissLipids deferred (no queryable API found). Phase 4 (Context
Builder) not started.

**What I built:** A three-track retrieval system over the Raman/lipid literature
where every claim resolves to a DOI.

1. **Prose track** — JATS XML → section-aware paragraph chunks → BGE embeddings →
   cosine search. Each chunk carries sentence offsets and a map of which inline
   citation falls in which sentence, so a retrieved claim resolves to
   claim → sentence → reference → DOI.
2. **Peak table track** — spectral assignment tables parsed into structured rows
   and queried by wavenumber **range**, never embedded. Cosine similarity has no
   notion that 1659 cm⁻¹ is adjacent to 1663, but does treat "1663" and "1553" as
   similar; wavenumber lookup is a range query, not a similarity problem.
3. **Paper track** — SPECTER2 document embeddings for paper-level similarity.

Plus two ingestion paths: PubMed-selected PMC full text (pipeline), and a
citation-frontier fetcher that ranks papers by how often the corpus cites them,
resolves OA status via Unpaywall, and fetches only legally open copies.

**Technologies:** Python · SQLite · NumPy · PyTorch (MPS) · HuggingFace
Transformers · BGE (bge-base-en-v1.5) · SPECTER2 + proximity adapter · defusedxml ·
PyMuPDF · pdfplumber · NCBI E-utilities · Unpaywall · Crossref · arXiv API ·
Phi-3.5 Mini 4-bit via mlx-lm (Apple Silicon local inference) · Unsloth +
PEFT/QLoRA (Hugging Face Jobs, A10G) for domain-adaptation fine-tuning

**Scale:**

| | |
|---|---|
| documents ingested | 217 JATS XML (post v4b filter, min_coocc=2) + 62 PDF (unparsed) |
| papers parsed | 217 |
| prose chunks, embedded | 7,833 |
| peak table rows | not re-measured since v3 (was 1,058; peak-table papers now exempt from filtering) |
| references resolved | **11,307 of 12,774 (88.5%)** carry a DOI parsed directly from JATS `<pub-id pub-id-type="doi">`, across 219 JATS-sourced papers (the 61 PDF-sourced papers have no parsed reference list — see `docs/solutions.md`) |
| inline citations mapped to sentences | not re-verified since v2 |
| vectors | 12,563 × 768 (chunks) + 281 × 768 (papers) |
| search latency | **83.7 ms p50 / 135.4 ms mean** (full `search_chunks`, 35 real queries × 20 reps = 700 runs, k=5, 281 papers / 12,563 chunks) — see `scripts/bench_latency.py` |
| fine-tuning corpus (separate from retrieval corpus, see below) | **107,665 abstracts** (Pool A 92,384 / Pool B 15,281 unique) — `data/finetune_corpus/`, see `docs/solutions.md` |
| QLoRA domain-adaptation fine-tune | Phi-3.5 Mini, held-out perplexity **4.911 → 3.955 (19.5% lower)**, 2,000-example held-out set, trained on 64,000 of 105,665 examples (60.6% of one epoch, step-capped to a fixed compute budget) — see `docs/solutions.md` |
| citation-check with fine-tuned model | **Verified, 0/3 hallucinated evidence IDs** (2 real domain questions + 1 deliberate out-of-domain probe, real retrieved evidence, ported `answer.py`'s exact system prompt + `check_citations` check onto the fine-tuned model via `scripts/verify_finetuned_citations.py`) — see `docs/solutions.md` |

**Metrics:**

- **Corpus redefinition: 7× more peak data.** v1 ("Raman AND lipid") returned
  papers that *use* Raman; v2 targets papers that must *tabulate* assignments
  (reviews, pure-lipid/model-membrane studies, explicit assignment work).
  Peak rows 99 → 699. Single-paper concentration 85% → 12%. Papers contributing
  tables 3 → 38. **2850 cm⁻¹ — the CH₂ symmetric stretch, the most fundamental
  lipid band — went from 0 rows to 17.**
- **Citation capture +27%** (2,639 → 3,360) after fixing citations nested in
  `<sup>`; **3,360/3,360 offset integrity** verified (every citation lands inside
  the sentence it is attributed to).
- **Czamara Table 2 extracted and independently validated**: 35 lipids × 17
  vibrational modes = 409 rows. **15/15 chemistry checks** (saturated lipids have
  no C=C; unsaturated show C=C at 1653–1657 with =CH at 3002–3005; triacylglycerols
  carry ester C=O at 1727–1749, doublet preserved). **200/200 cell agreement**
  against an independent extraction of the same table.
- **Retrieval: 14/14** curated domain queries return relevant top hits
  (mean top-1 cosine 0.784).
- **Scaling measured, not guessed**: at 281 papers / 12,563 chunks (38.6 MB),
  the brute-force cosine matmul + argsort itself is 1.8 ms p50 (200 reps) and
  stays exact. The end-to-end `search_chunks` call a real query actually pays
  — matmul, argsort, then one SQLite row-fetch per pooled candidate (pool=50)
  plus the per-paper cap/merge — is **83.7 ms p50 / 135.4 ms mean** (700 runs:
  35 real `eval_queries.py` questions × 20 reps, k=5; `scripts/bench_latency.py`).
  Per-candidate row fetches, not the cosine step, now dominate; the prior
  "3.1 ms" figure measured the matmul alone on a larger (48k-chunk) corpus and
  didn't include DB I/O — this is the first time the number reflects what a
  query actually costs end to end. The memory cliff is at ~2M vectors (≈52k
  papers → 6.1 GB → 18.9 s/query from swapping) — which is the scale of a
  prior failed attempt at this project, and a likely cause of it.
- **QLoRA domain-adaptation, measured not estimated: 19.5% lower held-out
  perplexity** (4.911 base → 3.955 fine-tuned, same 2,000-example held-out
  set for both). Phi-3.5 Mini + Unsloth QLoRA on 64,000 of 105,665 corpus
  abstracts. The fine-tuned model still answers only through the existing
  grounded-retrieval + citation-checking pipeline — this changes domain
  fluency, not the citation-verification requirement.

**Hard technical problems:**

1. **Silent corruption is the dominant failure mode in this domain.** Every text
   extraction tool tried destroys exactly the characters spectroscopy depends on,
   and none of them error. The PubMed MCP tool's `full_text` drops all table
   bodies, all inline citations (flattened to `[]`), superscripts (`1662 cm⁻¹` →
   `1662 cm`) and Greek symbols — 56 KB vs 350 KB of raw JATS on the same paper.
   A full-text academic search tool strips ν/δ from mode labels and renders
   `ν(C-C)` as `(C  C)`, ambiguous against C=C. My own parser shipped six such
   bugs. **The output always looks plausible.** Countermeasure: raw-first storage
   (parse is separable from fetch — the corpus was re-parsed six times without
   re-downloading), plus integrity assertions and independent cross-validation
   rather than eyeballing.

2. **PubMed cannot see this field.** Raman spectroscopy is analytical chemistry;
   PubMed indexes biomedicine. `"Raman spectroscopy of lipids"[Title]` → **0 hits**.
   The corpus cites 23,360 unique DOIs and holds 117 of them; 35% of that frontier
   is Wiley/RSC/Elsevier, which PMC does not carry. Solved by ranking the frontier
   by citation count and fetching legal OA copies — and by discovering that
   `is_oa: true` is not a fetch guarantee (the most-cited missing paper's "green OA"
   record was metadata-only, with no file attached).

3. **In a PDF table, the empty cells are the chemistry.** The flat text stream of
   Czamara Table 2 reads `MA 2943 2909 2869 2832 1457 1433 1419` — seven numbers
   for ten columns, with no indication which three are blank. Those blanks encode
   that myristic acid is saturated and therefore has no C=C. Collapsing the row
   left-to-right assigns a C=C stretch to a saturated fatty acid. Solved with
   `pdfplumber(text_x_tolerance=1)`; the default (3) fuses adjacent mode columns.
   The table also spans two pages with different layouts — the continuation page
   carries lipid identity only by row order — so the pages are zipped by index
   under an assertion that both yield exactly 35 rows.

4. **Selection, not scale, is the bottleneck.** Searching PMC full text returns
   36,822 hits for Raman+lipid because a paper mentioning "lipid" once in its
   bibliography matches; PubMed's title/abstract/MeSH index returns 4,316. Select
   in PubMed, fetch from PMC.

5. **Retrieval "I don't know" gap: −0.007 → +0.002. Closed out for now.**
   v3 (review dilution) didn't move it. Diagnosed the gap-driving papers
   (ovarian-cancer exosome/SERS, cold-plasma food science) and found a real
   signal: papers where "Raman" and a lipid term never co-occur in the same
   chunk contribute peak tables at 3% vs 16% for 10+ co-occurring chunks.
   Dropped zero-co-occurrence papers, then pushed to `min_coocc=2` (with a
   guard exempting any paper that already contributes `peak_tables` rows —
   table data lives outside chunk text, so the naive filter would have cut
   PMC10670390, an 84-peak-row paper). 328→217 papers across both rounds.
   Gap stuck at +0.002 both times. Root-caused: the ceiling score is pinned
   by a *legitimate* paper (SRS microscopy of lipid unsaturation in ovarian
   cancer cells) that shares surface vocabulary with an unrelated
   out-of-domain query — an embedding-model limitation, not corpus
   contamination. Further filtering would mean cutting good data to game a
   metric. Corpus-side fix is done; closing this further needs a query-side
   domain gate, not more corpus surgery. Deferred, not blocking.

**What I personally did:** Set the architecture and the constraints it had to
satisfy; drove corpus quality as the primary axis when retrieval "looked fine";
supplied the Czamara review that no API could reach; directed tool selection
(surfacing that a full-text academic search tool covers the Wiley literature
PubMed cannot, which produced the independent extraction that caught a column
off-by-one silently mislabelling every unsaturated lipid's C=C as a carbonyl);
and rejected hand-rolled geometry in favour of a library, which proved both faster
and more correct.

---

## Next

Ordered by what unblocks the most.

1. ~~**Evaluation set from real bench questions.**~~ **Done (2026-07-24).**
   `scripts/mine_bench_questions.py` mines papers' own Introduction goal
   statements ("we aimed to distinguish X from Y"); 15 curated + 4 questions
   grounded in the Czamara review added to `eval_queries.py`, each mined query
   carrying the source `known_paper_id` as ground truth. **Recall@10: 15/15
   (100%)** — every source paper was retrieved by its own question, mean top-1
   score 0.891. See `solutions.md` for the full method and why this check
   matters beyond the constructed 16.
2. ~~**Tighten the corpus.**~~ **Done (v3, 2026-07-23).** `Review[Publication
   Type]` in the v2 query gave any review a pass if "Raman" and a lipid term
   appeared anywhere in the record; measured on v2, review-only papers
   contributed peak tables at 3% vs 11% for every other group. Fixed by
   requiring `Raman[Title] AND lipid-terms[Title]` for the review branch.
   426→328 papers, 16,286→12,181 chunks, peak-table-paper density 9.2%→11.0%.
   **Prediction falsified:** separation gap unchanged at −0.007 — corpus
   dilution was not the cause of the "can't say I don't know" problem. See
   `solutions.md` for the full diagnostic and the corrected (previously
   unmeasured) 61% figure.
3. ~~**Diversity-aware retrieval.**~~ **Done (2026-07-23).** `search_chunks` in
   `query.py` now over-fetches a pool, caps at 2 evidence items per paper, and
   merges chunks adjacent in source-document order (consecutive `chunk_id`)
   into one item. Mean distinct papers in top-5: 2.7 → 4.14. Free fix — top-1
   scores and the separation gap are untouched, since capping only affects
   positions 2-5. `eval_queries.py` now measures this directly.
4. ~~**Agreement counts, not confidence scores.**~~ **Done (2026-07-24).**
   `query.py --peaks W` now groups by paper and reports "N independent papers"
   (`COUNT(DISTINCT paper_id)`, not row count — a 35-lipid review table doesn't
   count as 35 sources). `--agreement-report` adds a corpus-wide leaderboard,
   pure SQL bin/group/having. Measured leaders: 1005 cm⁻¹ (phenylalanine, 12
   papers), 1440 (CH₂ deformation, 9), 1130 & 1660 (8) — chemically sane.
   Assignment text deliberately left unnormalised (see `solutions.md`); known
   caveat that fixed-width binning can split one physical band across two bins.
5. ~~**Peak-set matching.**~~ **Done, position-only (2026-07-24).** `query.py
   --peak-set "2846,3009,1670,1739"` groups `peak_tables` by species (`origin`)
   and ranks by how many observed peaks match that species' known bands.
   Verified against cholesteryl oleate's own Czamara bands: exact 4/4 match,
   unambiguous top hit. Intensity ratios (I₂₈₈₀/I₂₈₅₀ chain order, I₁₆₆₀/I₁₄₄₀
   unsaturation) deliberately deferred — no literature-sourced classification
   threshold to score against without fabricating one. See `solutions.md`.
6. ~~**Prose from the 61 unparsed PDFs.**~~ **Done (2026-07-27).** `parse_pdf.py`
   run for real: 61 PDFs → 4,632 prose chunks, tagged `source='pdf'` (Czamara
   excluded — see `solutions.md`, its review text stays table-only per the
   invariant `eval_queries.py` depends on). Corpus: 220→281 papers, 7,931→12,563
   chunks. Not yet embedded — new chunks have `vector_row_idx=NULL` and are
   invisible to retrieval until item 8 runs. Once embedded, measure whether
   `source='pdf'` chunks retrieve worse than `source='jats'` (glyph damage:
   `375 cm1`, `υ▷CH2◁`).
7. ~~**Czamara Table 1 → LIPID MAPS.**~~ **Done (2026-07-24).** `scripts/resolve_lipid_identity.py`
   resolves 29/35 acronyms to LIPID MAPS records (LM_ID, classification, formula),
   cached in `lipid_identity`. `query.py --peaks`/`--peak-set` now show resolved
   names. 6 unresolved are genuine LIPID MAPS coverage gaps, not a bug. See
   `solutions.md` for the PubChem-CID + abbrev-fallback resolution chain, and why
   SwissLipids was evaluated and deferred (no queryable per-compound API).
8. ~~**Embedding on Colab Pro.**~~ **Done (2026-07-28).** All 12,563 chunks
   (BGE) + 281 papers (SPECTER2) re-embedded on an A100 via
   `colab/embed_chunks.ipynb` (writes `scripts/embed.py` verbatim into the
   Colab runtime so the logic run there can't drift from the repo). SPECTER2
   proximity-adapter warning during the run turned out not to matter — mean
   pairwise cosine 0.879/std 0.027 at n=281 lines up with the already-falsified
   adapter hypothesis from 2026-07-15 (see `solutions.md`: corpus topical
   homogeneity, not a model artifact, was already proven to explain this).
   Surfaced two real, previously-invisible data-quality bugs in the 61 PDF-
   sourced papers once their chunks became retrievable for the first time —
   23 `NULL` titles and 13 more with junk PDF-metadata titles ("Microsoft
   Word - ....docx", "No Job Name", bare manuscript IDs) masquerading as real
   ones. Fixed via `scripts/backfill_titles.py` (Crossref/arXiv lookup,
   PDF-internal title never trusted as authoritative again), plus a targeted
   local re-embed of the 1,122 chunks whose BGE input had stale junk-title
   text baked in, plus defensive fixes in `query.py`/`answer.py` so a missing
   title can never crash retrieval again. See `solutions.md` for the full
   incident writeup.
9. ~~**Knowledge graph layer (Neo4j AuraDB).**~~ **Done (2026-07-28).**
   `scripts/build_graph.py` derives a graph from `papers.db` (Paper, Lipid,
   LipidClass, LipidSubclass nodes; CITES, REPORTS_PEAK, IN_CLASS,
   IN_SUBCLASS edges) — SQLite stays canonical, the graph is rebuilt from it,
   never migrated. Runs against AuraDB Free (cloud), not local Docker, to
   avoid the same RAM contention that keeps MLX inference outside Docker.
   418 Paper nodes (255 in_corpus + 163 frontier-only, added for free
   citation reach), 867 CITES edges, 29 Lipid nodes, 345 REPORTS_PEAK edges.
   `scripts/graph_query.py` adds two traversals SQL handles poorly:
   `--lipid-neighbors` (class/subclass + lipids sharing peak-report papers)
   and `--citation-path` (shortest citation chain between two DOIs), both
   verified against live data. General exploration happens in Aura's own
   Neo4j Browser. See `solutions.md` for why this doesn't contradict the
   earlier pgvector/FAISS rejection (visualization need, not a performance
   need) and `questions.md` for the deferred chunk-level MENTIONS edges.
10. ~~**Fine-tuning corpus (broad, separate from the retrieval corpus).**~~
    **Done (2026-08-14).** `scripts/fetch_finetune_corpus.py` pulled two
    PubMed E-utilities pools by MeSH term — Pool A (bulk lipid biochemistry/
    lipidomics/membrane biophysics) and Pool B (Raman/IR/vibrational
    spectroscopy vocabulary, not lipid-restricted) — paged by calendar month
    to work around NCBI's 9,999-record esearch cap. **107,665 unique
    abstracts** (Pool A 92,384 / Pool B 15,281 unique, 85.8%/14.2% split),
    stored as flat JSONL in `data/finetune_corpus/`, deliberately never
    merged into `papers.db`. See `solutions.md` for the full query text and
    why keeping this corpus separate from the retrieval corpus matters.
11. ~~**QLoRA fine-tune, Phi-3.5 Mini, domain adaptation.**~~ **Done
    (2026-08-15).** `scripts/train_domain_adapt.py`, next-token/causal-LM
    objective on raw title+abstract text (not an instruction format — no
    specific downstream task shape, the goal is domain fluency; see the
    script's docstring for the explicit reasoning this task required).
    Unsloth `FastLanguageModel` QLoRA (4-bit, rank 16) on a Hugging Face Jobs
    A10G. **Held-out perplexity 4.911 → 3.955 (19.5% lower)**, same
    2,000-example held-out set used for both measurements. Trained on 64,000
    of 105,665 examples (60.6% of one epoch) — step-capped to a fixed compute
    budget, not a full epoch; see `solutions.md` for why one epoch was never
    the target here. Adapter pushed to
    `srikarjy025/lipidos-phi3-domain-adapt` (private). This model still only
    answers through the existing grounded-retrieval + citation-checking
    pipeline — fine-tuning changed domain fluency, not the citation
    requirement. See `solutions.md` for the full incident writeup: four
    GPU-capacity preemptions (`exit code 143`, no Python traceback) across
    the run, a wrong assumption about `/data` bucket-mount persistence that
    cost one fully-wasted 2.5-hour run, and the Hub-repo-backed checkpoint
    resume design that fixed it.
12. ~~**Verify citation-checking with the fine-tuned model in place.**~~
    **Done (2026-08-15).** `answer.py`'s citation check runs against an
    MLX-quantized model, a different weight format from the fine-tune's
    PEFT/transformers adapter — not directly interchangeable.
    `scripts/verify_finetuned_citations.py` ports `answer.py`'s exact system
    prompt and `check_citations` regex onto the actual fine-tuned model
    (base + adapter via `transformers`/`peft`), tested against real retrieved
    evidence for 2 domain questions plus 1 deliberate out-of-domain probe.
    **Result: 0/3 hallucinated evidence IDs** — the out-of-domain question
    correctly got "the evidence does not answer this" rather than a made-up
    citation. Caveat: `check_citations` only verifies cited evidence numbers
    exist, not that the cited evidence's content actually supports the
    claim — a pre-existing scope limit of the check itself, unchanged by
    fine-tuning either way.
13. ~~**Wire Context Builder + the fine-tuned model into one real pipeline.**~~
    **Done (2026-08-15).** Blueprint Phases 4 (`context_builder.py`) and 5
    (`raman_integration.py`) already existed but only ever printed merged
    evidence — neither generated an interpretation, and neither was ever
    connected to `answer.py` (grepped: zero references) or to any fine-tuned
    model. Also found: the blueprint's own intended Phase 6 model
    (`data/finetuned_phi3_mlx/`) has only a training config on disk, no
    actual adapter weights — it never finished/saved, so tonight's QLoRA
    adapter is the only complete, working fine-tuned artifact in this
    project. New split-script pipeline: `scripts/build_evidence.py` (local,
    Context Builder's 3-track merged/deduped/cited evidence, fast, no GPU)
    → `scripts/generate_finetuned_answer.py` (remote GPU, loads the
    fine-tuned model, generates, runs the same `check_citations` as
    `answer.py`). Split because the fine-tuned model (~7.6 GB bf16) doesn't
    comfortably fit this project's 8 GB M2 development machine. **Verified
    working end-to-end**: real question → 5 merged evidence items → cited
    interpretation referencing evidence #2/#3/#5, zero hallucinated IDs.

## Deliberately not built

- **Vector database (pgvector/FAISS/Chroma).** They solve approximate search at a
  scale we cannot reach. Brute force is 83.7 ms p50 end-to-end (1.8 ms for the
  matmul itself) and *exact* at the full corpus; adopting ANN would trade
  exactness for speed we do not need, and pgvector-in-Docker
  costs RAM against the model on an 8 GB machine. FAISS becomes correct at ~2M
  vectors — 40× beyond the entire reachable literature.
- **Blended confidence scores.** Cosine 0.80 is not 80% confidence. Combining a
  retrieval score, a database hit-count and a classifier probability yields a number
  with no units that looks authoritative. Surface each signal separately.
- **AI-generated-text detection.** No reliable detector exists, and false positives
  land hardest on non-native-English scientific writing — a large share of this
  literature. Verifiable provenance facts are shipped instead: retraction/erratum
  status (all 426 papers checked, zero flagged), publication type, MEDLINE indexing,
  each with the date checked.
- **General-purpose PDF table extraction.** Layout varies per paper — a parser tuned
  to one J. Raman Spectrosc. table fails on the next paper in the same journal
  (two-column body). Triage says only 3–4 of 62 PDFs carry peak tables. JATS is the
  pipeline; PDF is the scalpel.
