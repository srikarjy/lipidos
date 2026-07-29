# Decision Log

Decisions made, with the reasoning behind them. When a question in `questions.md` gets resolved, it moves here.

---

### 2026-07-15 — Corpus source: PMC JATS XML via efetch, NOT the PubMed MCP tool's `full_text`
**Question:** where do paper full texts come from for Phase 1 ingestion?
**Decision:** fetch raw JATS XML from NCBI efetch (`https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi?db=pmc&id={id}&rettype=xml`). Use the PubMed MCP tool for *search and metadata* only, not for full text.
**Why:** the MCP tool's `full_text` field is lossy in precisely the dimensions this project depends on. Measured on PMC10670390 ([DOI](https://doi.org/10.3390/cells12222589)):

| | MCP `full_text` | Raw JATS XML |
|---|---|---|
| Size | 56 KB | 350 KB |
| Table rows | 0 | 106 |
| Inline citations | 0 (all rendered `[]`) | 451 `<xref>` |
| Superscripts | 0 | 68 |

Specifically it drops: **table bodies** (captions survive, rows do not — the peak assignment tables are the single highest-value content in these papers); **all inline citation markers**, flattened to empty `[]`, which destroys claim→source tracing in a project whose whole thesis is claim→source tracing; **superscripts**, so "1662 cm⁻¹" becomes "1662 cm"; and **Greek symbols**, so a table legend reads "= stretching" instead of "ν = stretching".

This is the same class of silent-truncation bug as naive `.text` XML parsing — the text still looks plausible, so nothing errors, it's just quietly wrong. Building Phase 1 on it would have produced a corpus with no tables and no citations, and we'd likely have blamed retrieval quality.

**Consequence:** we need our own JATS parser. Parse untrusted network XML with `defusedxml`, not stdlib ElementTree (entity-expansion defense).

---

### 2026-07-15 — Evidence unit: paper-authored peak assignment tables
**Question:** what shape does a RAG-panel evidence item take?
**Decision:** adopt the schema the papers already use. Table 2 of PMC10670390 parses to:

```
Wavenumber (cm−1) | Assignment | Origin | Reference
347–353           | vskel(C-C) | NAA    | [112]
473–475           | v(S-S)     | NSE    | [125]
```

**Why:** this *is* the blueprint's `peak → assignment → source → citation` panel format, already produced by domain authors. We don't invent an evidence format; the field's own convention validates the design. Each row carries its own `Reference`, so a claim resolves past the containing paper to the primary study that established it.

**Note:** wavenumbers are reported as **ranges** (347–353), not points — authors already encode tolerance. Relevant to the conditions problem parked in `questions.md`.

---

### 2026-07-24 — Eval set: mine real bench questions from papers' own stated aims, not just constructed ones
**Question:** `scripts/eval_queries.py`'s 16 queries were all hand-constructed (my best guess at what a scientist asks). PROGRESS.md flagged this as the remaining unmeasured input — plausible isn't measured.
**Decision:** three-tier eval set (`scripts/eval_queries.py`), sourced per tier:

1. **Mined** (`scripts/mine_bench_questions.py`) — regex over every corpus paper's Introduction chunks for goal-statement language ("we aimed to," "sought to," "in order to distinguish," etc.), ranked by how strong the intent marker is, capped at 1 candidate/paper. Scanned 1,036 intro chunks across 189 papers, surfaced 40 candidates; hand-picked 15 that were spectroscopically specific (rejected ones like "this study aims to bridge this knowledge gap" — no testable content). Each carries a `known_paper_id`: the paper that wrote the sentence is the ground truth, so these are judged by **recall@10** (was the source paper actually retrieved), not eyeballed score.
2. **Reference** — 4 discrimination questions pulled directly from stated facts in the Czamara 2015 review (`data/marker/czamara2014/czamara2014.md`), e.g. the exact bands separating cholesteryl palmitate from cholesteryl stearate. Judged by `expect` content, same as the original constructed tier, since Czamara isn't itself chunked/embedded (only its peak table is parsed) so there's no paper_id to check recall against.
3. **Constructed** — the original 16, kept as-is, still eyeballed by `expect`.

**Result:** mined-question recall@10 = **15/15 (100%)** — every paper whose own Introduction posed a specific spectroscopic question had that paper surface in its own top-10 results, at a mean top-1 score of 0.891 (vs 0.831 for the constructed tier). Separation gap unchanged at +0.002 (see the −0.007→+0.002 fix in PROGRESS.md; this didn't move it further, as expected — the gap's remaining cause is an embedding-model ceiling, not corpus/query coverage).
**Why this matters:** the constructed tier could have been unconsciously shaped to match what the corpus already retrieves well (confirmation bias in query design). Ground-truthing to a paper that wrote its own question, independent of my retrieval intuitions, is a check the constructed tier structurally cannot provide.

---

### 2026-07-24 — Peak-set matching: position-only, no fabricated ratio thresholds
**Question:** the real bench query is "I have peaks at 2850, 2880, 1440, 1660 — what lipid is this?" Single-band `--peaks` lookup doesn't help someone who already knows what 1440 is; PROGRESS.md flagged band ratios (I₂₈₈₀/I₂₈₅₀ chain order, I₁₆₆₀/I₁₄₄₀ unsaturation) as the actual analytical readout.
**Decision:** implemented `match_peak_set()` / `--peak-set` in `scripts/query.py` as **position-only** matching: groups `peak_tables` rows by species (`origin`, mostly the Czamara review's 35-lipid table), ranks species by how many of the *observed* peaks fall within tolerance of that species' known bands. Deliberately did not implement intensity-ratio scoring in this pass — the corpus states ratios are indices of chain order/unsaturation but gives no numeric classification threshold ("ratio > X means Y"), and fabricating one would violate the project's own "no blended confidence scores, surface each signal separately" rule (see PROGRESS.md "Deliberately not built").
**Verified:** querying `2846,3009,1670,1739` (cholesteryl oleate's own bands per Czamara) returns COA as the sole 4/4 match; dropping to `2846,2866,1670,1739` (2866 is CHL-backbone CH₂ stretch, mentioned in Czamara's prose but not captured in the parsed table row) correctly drops to a 3/4 tie among COA/CPA/TEA — the miss is a real data-completeness gap (prose vs. table extraction), not a matching bug.
**Consequence:** intensity-ratio scoring is deferred, not abandoned — needs either a literature-sourced threshold or explicit "value only, no verdict" framing before it's added.

---

### 2026-07-24 — Phase 2 complete: Phi-3.5 Mini base model as the LLM panel, via MLX
**Question:** blueprint Phase 2 is "question → retriever → base model → answer with citations, validate grounding survives generation before adding complexity." Retrieval + citation grounding were already validated (recall@10 work above); the model layer itself hadn't been wired in.
**Decision:** `scripts/answer.py`. Retrieval reuses `search_chunks` from `query.py` unchanged. Model is `mlx-community/Phi-3.5-mini-instruct-4bit` via `mlx-lm`, run locally on the M2 Air outside Docker (per [[user-profile]]'s memory constraint). Two panels printed, never merged, per the blueprint's "Output design" requirement: RAG panel (raw evidence, unchanged from `query.py`) and LLM panel (generated answer), with a citation checker (`check_citations`) that flags any `evidence #N` the model cites that doesn't exist in the retrieved set — catches hallucinated pointers, not just hallucinated facts.
**New dependency conflict, fixed:** `pip install mlx-lm` silently bumped `transformers` 4.57.6 → 5.14.1, which is exactly the fragility the requirements.txt header already warns about (`adapters~=4.57.6`). Re-pinned to 4.57.6; smoke-tested that `mlx_lm.generate` still works against it despite mlx-lm's own declared `transformers>=5.0.0` requirement (pip's conflict warning is overly strict here). Documented in `requirements.txt` directly so this doesn't get silently re-broken by a future `pip install -U`.
**Real bug found and fixed:** mlx-lm's `TokenizerWrapper.eos_token_ids` for this model only contained `{32000}` (`<|endoftext|>`), but Phi-3.5's own chat template ends each turn with `<|end|>` (`tokenizer.eos_token_id` = 32007) — a different token. Generation never saw its real stop token, ran past the first turn, and hallucinated a fake follow-up `<|end|><|assistant|>` exchange that repeated until `max_tokens`. Fixed with `tokenizer.eos_token_ids.add(tokenizer.eos_token_id)` before generating. This is a base-model quirk in the mlx-lm conversion of this specific checkpoint, not a project bug, but it would have silently produced garbage multi-turn output in every answer without the fix.
**Verified:** two manual queries. One with no supporting evidence in the retrieved set — model explicitly said "the question cannot be answered" rather than guessing, citing evidence [2] as the closest-but-insufficient match. One with directly supporting evidence — cited [2] and [3] correctly, zero hallucinated evidence IDs in either case.
**Consequence:** Phase 2 is done. Phase 3 (LIPID MAPS/SwissLipids knowledge layer) is next per the blueprint's ordering, though `--peak-set` (built ahead of schedule) already covers part of Phase 5's intent without a trained CNN/PCA classifier.

---

### 2026-07-24 — Phase 3: LIPID MAPS only, SwissLipids deferred (no per-compound REST endpoint found)
**Question:** blueprint Phase 3 is "retriever + LIPID MAPS + SwissLipids → merged evidence." This closes PROGRESS.md item 7 too: Czamara's peak table only carries local acronyms (MA, COA, PC...), 22% of PMC peak rows carry any origin at all, and there was no lipid identity to normalize until now.
**Decision — LIPID MAPS chain:** LIPID MAPS' REST API has no free-text name search (confirmed against the live API, not just docs — `input_item=name` returns "This input item does not exist"; valid inputs are `lm_id, pubchem_cid, inchi_key, formula, abbrev, abbrev_chains, regno`). Resolution chain: PubChem `compound/name/{name}/cids` (public, no key) → LIPID MAPS `compound/pubchem_cid/{cid}/all`. Falls back to LIPID MAPS `compound/abbrev/{acronym}` directly when the acronym happens to match LIPID MAPS' own class abbreviation (fixed PC, PE — their PubChem-CID chain misses even though the compound exists in LIPID MAPS, apparently indexed under a different preferred CID). `scripts/resolve_lipid_identity.py` runs this once offline against the 33 acronyms in Czamara Table 1 and caches results in a new `lipid_identity` table — `query.py` never makes a network call at query time.
**Result: 29/35 resolved.** Remaining 6 (TCA, TCY, TPE, TEA, TEI — specific TAG standards not individually catalogued in LIPID MAPS; SM — a generic class name, not a single compound) confirmed as real coverage gaps: tried a third fallback (InChIKey from PubChem) for the TAGs and it also returned empty. Not a matching bug.
**Decision — SwissLipids deferred:** investigated live (not from memory/docs) — `/api` 404s, and the only concrete data-access pattern findable in the site's own JS bundle is a bulk SPA file download (`file.php?cas=download_files...`), not a queryable per-compound endpoint. Integrating it properly would mean downloading and locally joining against SwissLipids' full species dump — disproportionate engineering for a 33-lipid problem. Revisit only if LIPID MAPS coverage becomes the actual bottleneck.
**Wired into `query.py`:** `--peaks` and `--peak-set` now show `full_name (acronym, LM_ID/class)` instead of the bare acronym wherever `lipid_identity` has a resolved row, falling back to the raw acronym otherwise (e.g. TEA, one of the 6 misses, still prints as "TEA"). Verified: `--peak-set "2846,3009,1670,1739"` now prints "cholesteryl oleate (COA, Sterols [ST01])" as the sole 4/4 match.

---

### 2026-07-15 — Repo reset
Prior git history and the Phase 1 PubMedQA/LIPID MAPS pipeline were deleted at user request; repo re-initialized and restarted from `lipid-raman-rag-blueprint.md`. Old history is bundled at
`/private/tmp/claude-501/-Users-srikarjy-resume-projects-LipidOS/ed31585e-671e-4928-baee-c432b3b07373/scratchpad/lipidos-backup/old-history.bundle`
(restore: `git clone old-history.bundle recovered/`). **This is session scratchpad — not permanent.** Copy it somewhere durable if the old Phase 1 work matters.

---

### 2026-07-15 — Three tracks, one SQLite db + flat numpy vectors
**Question:** how are prose, tables and papers stored and searched?
**Decision:** `data/papers.db` (papers, chunks, peak_tables, references, paper_embeddings) plus `chunk_vectors.npy` (1109, 768) and `paper_vectors.npy` (28, 768). `vector_row_idx` joins a SQLite row to its numpy row.
**Why:** SQLite holds everything relational — text, metadata, foreign keys, and *range* queries. The `.npy` files hold only floats, because cosine similarity is a numpy operation and SQLite has no fast vector search. No Docker, no pgvector: 1,109 chunks does not justify infrastructure we'd pay for in RAM against the model on an 8GB machine.

### 2026-07-15 — Peak tables are never embedded
**Decision:** spectral tables go to `peak_tables` and are queried by wavenumber range (`WHERE high >= ? AND low <= ?`), not by similarity.
**Why:** embedding models tokenize numbers into meaningless fragments, and cosine similarity has no notion that 1659 is *near* 1663 — while happily thinking "1663" and "1553" look alike. A wavenumber lookup is a range query. This is the blueprint's own principle ("structured lookup — no embedding needed, no hallucination risk"), applied to tables found inside papers rather than inside LIPID MAPS. The rows are already ranges (347–353), so the data asks for it.

### 2026-07-15 — BGE for chunks, SPECTER2 for papers
**Decision:** `bge-base-en-v1.5` (CLS-pooled, L2-normalised) for chunk retrieval; `allenai/specter2_base` for paper-level embeddings. Driven through `transformers` directly rather than `sentence-transformers` — both are BERT-family CLS models, so a heavy dependency buys nothing.
**Why:** they do different jobs. BGE is trained on query→passage pairs, which *is* chunk retrieval. SPECTER2 is trained on citation graphs for whole-document similarity from title+abstract. Using SPECTER2 for passages is a task mismatch; using BGE for paper similarity throws away citation-graph signal. The blueprint's "domain-tuned, e.g. SPECTER2-style" instinct was right for track 3 and wrong for track 1.
**Still open:** no bake-off has been run. MedCPT (NCBI, query→passage *and* biomedical) is the untested third option. Revisit with a real question set.

### 2026-07-15 — Parser bugs found and fixed (all silent-corruption class)
Each of these produced plausible-looking output while being wrong — no exception, no warning:
1. **xref offsets drifted.** Offsets were computed on raw text, then whitespace-normalised afterwards, changing string length. Fixed with a length-preserving remap. Verified: 3360/3360 citations land inside their assigned sentence.
2. **Citations nested in `<sup>` were swallowed.** ACS/Nature render citations as superscript numbers (`<sup><xref>34</xref></sup>`); the `<sup>` branch flattened them with `itertext()`. **592 citations across 9 papers** were lost. Fixed by recursing when a `bibr` xref is nested. Total captured: 2639 → 3360.
3. **`colspan` broke column mapping.** A spanning header ("Normalized Raman intensity" over 3 group columns) gave the header fewer cells than data rows, so "assignment" indexed an intensity column and stored `59.1 ± 31.4` as a vibrational assignment. Fixed by expanding colspan; plus a guard rejecting letterless assignments.
4. **`--rebuild` orphaned the vector files.** Dropping the db wiped `vector_row_idx` but left the `.npy` files, so queries returned silently empty. `--rebuild` now deletes the vectors too.

### 2026-07-15 — Sentence splitting is not `split('.')`
**Decision:** guarded splitter protecting decimals (`0.5`), abbreviations (`et al.`, `Fig.`, `e.g.`) and initials.
**Why:** the spec said split on '.'. This corpus is saturated with decimals and abbreviations; naive splitting misattributes citations to neighbouring sentences, which silently breaks the sentence-level attribution the output design depends on.

### 2026-07-15 — Environment pinned to a venv
**Decision:** project `.venv` + pinned `requirements.txt`. Anaconda base had `protobuf 4.25.9`; `transformers` needs `>=5.27` for its BERT import.
**Why:** mutating the base conda env is how the previous attempt became unreproducible. Pinning is the whole point of Phase 1.

---

### 2026-07-15 — Corpus v2: target papers that *tabulate*, not papers that *use*
**Question:** v1's corpus (1,256 papers, "Raman spectroscopy AND lipid") produced almost no peak tables. Why, and what replaces it?
**Diagnosis, measured on a 30-paper sample:** only 3/28 papers had any peak table, 84 of 99 rows came from a single paper, and **2850 cm⁻¹ — the CH₂ symmetric stretch, the most fundamental lipid band there is — returned zero rows.** The corpus was full of papers that *use* Raman on biological samples (SRS microscopy, cancer diagnostics, COVID serum). Those papers cite assignments; they don't tabulate them.
**Decision:** v2 targets papers that must report assignments to make their point — explicit assignment/reference-spectra work, `Review[Publication Type]` (reviews tabulate; that is what reviews are for), and pure-lipid/model-membrane studies (measuring DPPC means assigning its bands). 510 papers with PMC full text.
**Result:**

| | v1 (30 papers) | v2 (523 papers) |
|---|---|---|
| papers parsed | 28 | 425 |
| chunks | 1,109 | 16,286 |
| peak rows | 99 | 699 |
| references | 2,600 | 30,462 |
| papers w/ peak tables | 3 | 38 |
| top paper's share of peak rows | 84/99 (85%) | 84/699 (12%) |
| 2850 cm⁻¹ lookup | **0 rows** | real assignments |

**Why it matters:** the query, not the model, was the bottleneck. Both retrieval tracks were correct all along and starved of data.

### 2026-07-15 — SPECTER2 proximity adapter: tried, made no difference
**Question:** paper-level similarity showed mean pairwise cosine 0.916 with std 0.026 — every paper "similar" to every other, ranking within noise. Was that `specter2_base` lacking its task adapter?
**Decision:** loaded the `allenai/specter2` proximity adapter (via the `adapters` library). **Hypothesis was wrong.** With adapter: mean 0.901, std 0.026 — statistically identical.
**Conclusion:** the compression is not a model artifact. The corpus is genuinely homogeneous — 28 papers all about Raman spectroscopy of lipids really are all mutually similar, and SPECTER2 was reporting that faithfully. Adapter kept anyway (it is the correct usage of SPECTER2), but it bought nothing here. **Do not attribute a data property to a model without testing it.**
**Caution:** `adapters` silently upgraded transformers 4.51.3 -> 4.57.6, invalidating the pin. requirements.txt updated to match.

### 2026-07-15 — Two more silent-corruption bugs in table parsing
5. **Citations fused onto wavenumbers.** The `<sup>` fix (which correctly recovered 592 prose citations) backfired in tables: a cell reading `850–880⁸` (band 850–880, citation 8) rendered as `850-8808`, producing a wavenumber of 8808 cm⁻¹. Fixed by rendering the wavenumber column with citations dropped — the peak tables carry a separate Reference column anyway.
6. **Range guard only validated the low bound.** `100 <= lo <= 4000` let `850-8808` through because `hi` was never checked. Now validates both bounds and `hi >= lo`. All 699 rows verified within 100–3600 cm⁻¹.

### 2026-07-15 — Track 3 compression is not corpus size either
Follow-up to the adapter test. Hypothesis was that 28 papers was simply too few for SPECTER2 to discriminate. Re-measured at n=425:

| corpus | mean | std | min | max |
|---|---|---|---|---|
| v1, n=28  | 0.901 | 0.026 | 0.822 | 0.975 |
| v2, n=425 | 0.872 | 0.030 | 0.737 | 0.981 |

15x more papers moved the mean 0.03 and the std 0.004. **Both hypotheses for track 3's compression are now falsified** — it is neither the missing adapter nor the sample size. It is topical narrowness: every paper in this corpus is about Raman spectroscopy of lipids, so they genuinely *are* all similar, and SPECTER2 reports that faithfully.

**Consequence:** track 3 ("papers similar to this one") has limited value on a single-topic corpus — it is answering "which of these Raman-lipid papers is most Raman-lipid-ish". It would earn its keep against a broad corpus, or for deduplication. Not a bug; a mismatch between the tool and a deliberately narrow corpus. Worth revisiting whether track 3 pays for itself at all.

---

### 2026-07-23 — Corpus v3: reviews no longer get a free pass
**Question:** PROGRESS.md claimed "61% of PMC papers mention Raman or lipids ≤1 time," blamed on the `Review[Publication Type]` clause, and predicted tightening it would fix the retrieval separation-gap problem. Neither half of that claim had a `solutions.md` entry — it was never actually measured.

**Correction:** the 61% figure does not reproduce under any measurement. Full chunk-text mention count gives 3% of papers at ≤1 mention; title+abstract gives 14%. That number should not have been in PROGRESS.md and is removed.

**What *is* real, measured on the fetched v2 corpus (426 papers):** splitting papers by how they qualified for the query —

| group | n | peak-table contribution |
|---|---|---|
| matched ASSIGNMENT/MODEL_LIPIDS terms | 87 | 11% |
| Review-only (no substantive term match) | 104 | 3% |
| matched via MeSH/indexing only, non-review | 235 | 11% |

Review-only papers contribute peak tables at a third the rate of every other group. Sampling their titles explains why: `Review[Publication Type]` in v2's query gave any review a pass regardless of topic, so long as "Raman" and a lipid term appeared *anywhere* in the record (title, abstract, or MeSH) — "Gut Microbiota-Derived Metabolites... in Pigs," "Meibomian gland dysfunction," "Dietary intakes of flavan-3-ols and cardiovascular health" all qualified this way. The MeSH-only non-review group, by contrast, performs identically to the tightly-matched group and was not the problem.

**Decision:** require reviews to be genuinely *about* Raman spectroscopy of lipids — `Raman[Title] AND {lipid-terms}[Title]` — not just mention both words somewhere:
```
(ASSIGNMENT OR MODEL_LIPIDS OR (Review[Publication Type] AND Raman[Title] AND LIPIDS_TITLE))
```
PubMed's `[Title]` field tag does not stem plurals (`lipid[Title]` and `lipids[Title]` return very different counts), so singular/plural forms are listed explicitly — an early version of this query silently dropped a legitimately-titled paper ("Raman analysis of lipids in cells") because only the singular form was checked.

**Measured effect:** esearch count 511 → 414. Fetched and re-parsed: 426 → 328 papers, 16,286 → 12,181 chunks, 1,108 → 1,058 peak rows, peak-table-contributing papers 39/426 (9.2%) → 36/328 (11.0%) — a real but modest concentration improvement.

**Prediction falsified:** PROGRESS.md predicted this would move the retrieval separation gap positive. Re-ran `eval_queries.py` after rebuild: gap is unchanged at exactly **−0.007**, with the same worst-real-query (`disc-lipid-protein`, 0.647) and same best-nonsense-query (`ood-crispr`, 0.654) as before, hitting the same papers. The ovarian-cancer exosome review driving the OOD false-positive apparently didn't qualify via the Review clause at all, so this fix never touched it. Rules out corpus dilution as the separation-gap cause; strengthens the corpus-contamination hypothesis in `questions.md`.

**Housekeeping:** `--rebuild` in `parse_jats.py` drops the entire db file, which also wipes the `frontier` table (citation-frontier tracking, not owned by that script) — not documented anywhere. Had to recreate its schema and restore 201 rows from a pre-rebuild backup. Worth fixing so `--rebuild` scopes to its own tables only. v3 confirmed good; v2 backups deleted.

---

### 2026-07-23 — Corpus v4: post-fetch co-occurrence filter (`scripts/filter_corpus.py`)
**Question:** v3 fixed the review-only dilution class but did not move the separation gap. What is the remaining contamination?

**Diagnosis:** picked the two papers driving the gap (`disc-lipid-protein`'s top hit, `ood-crispr`'s top hit) and read them. Both are legitimate query matches that are not actually about Raman spectroscopy of lipids:
- An ovarian-cancer exosome/SERS paper matched via "lipid bilayer"/"liposome," describing exosome membrane biology in passing — 0/74 chunks ever mention Raman and a lipid term together.
- A cold-plasma food-science paper (bacterial inactivation) used Raman as an analytical method on an unrelated system; "lipid" appears once, unrelated to the Raman measurement. Not caught by the v3 review fix — neither paper is a review.

**Measured generalization** across all 328 v3 papers, bucketed by how many chunks contain BOTH "Raman" and a lipid term together (not just present somewhere in the paper independently):

| co-occurring chunks | n papers | peak-table contribution |
|---|---|---|
| 0 | 74 | 3% |
| 1-2 | 77 | 10% |
| 3-9 | 145 | 14% |
| 10+ | 32 | 16% |

Monotonic, ~5x spread. PubMed's esearch has no full-text proximity operator, so this cannot be expressed as a fetch-time query — it is a post-fetch filter on parsed chunks.

**Decision:** drop the 74 zero-co-occurrence papers (`scripts/filter_corpus.py`, `--min-coocc 1` default). Sampled titles confirm genuine contamination: nanoparticle coatings, algae metamorphosis, TiO₂ nanotubes, hydrogel wound dressings — Raman used as a generic characterization tool, lipid mentioned once unrelated to it.

**Result:** 328 → 254 papers, 12,181 → 9,114 chunks. Re-ran `eval_queries.py`: separation gap **−0.007 → +0.002**. Positive, but thin — this is not a solid floor, it is one paper away from flipping back negative. The worst real query (`disc-lipid-protein`, 0.647) did not move at all across either filtering round; its ceiling looks like a genuine retrieval-quality problem (protein/lipid CH₂ deformation band overlap — flagged in `eval_queries.py`'s own `expect` field as the hard case), not corpus contamination. Contamination was one real contributor to the gap, not the whole story.

v3 backup kept at `data/papers_v3_backup.db` / `*_v3_backup.npy` pending further validation of v4.

**Follow-up — pushed the filter further, hit a real ceiling.** Added a guard: never
drop a paper that already contributes `peak_tables` rows, since peak data lives
outside chunk text and a table-heavy, terse-prose paper (e.g. PMC10670390, 84 rows
from the paper that originally justified raw-JATS-over-MCP-tool) would otherwise be
wrongly cut at `min_coocc>=3`. With that guard, tried `min_coocc=2`: 254→217 papers,
9,114→7,833 chunks. **Gap unchanged at +0.002** — identical to `min_coocc=1`.

Traced why: `ood-crispr`'s ceiling score (0.645) is pinned by `PMC9564215`
("Ovarian cancer cell fate regulation by the dynamics between saturated and
unsaturated fatty acids") — a paper using single-cell SRS microscopy to study lipid
unsaturation, i.e. legitimately on-topic, arguably a good hit for the `bio-ld` query.
Its co-occurrence count is 2, right at the `min_coocc=2` boundary. Cutting it to
close the gap further would mean deliberately removing a relevant paper to game a
metric, not fixing contamination.

**Conclusion: corpus filtering is done for this problem.** The gap moved
−0.007 → +0.002 through two real, defensible fixes (review dilution, co-occurrence
contamination). What remains is not a data-quality issue — it's the embedding model
conflating topical surface-overlap ("cancer," "cell") with true relevance for an
unrelated out-of-domain query, paired with a genuinely hard in-domain floor
(`disc-lipid-protein`, protein/lipid Raman band overlap) that never moved across
either filtering round. Fixing this needs a different mechanism (e.g. a query-side
domain gate), not more corpus surgery. Deferred; not blocking other work.

---

### 2026-07-23 — Diversity-aware retrieval (`search_chunks` in `query.py`)
**Question:** plain top-k clusters on whichever paper the query matches best. Measured on v2: top-5 averaged 2.7 distinct papers; one query returned 5 chunks from a single paper, 4 of them adjacent paragraphs of the same passage — one opinion shown five times, not five pieces of evidence.

**Decision:** over-fetch a pool (`max(k*10, 50)` candidates), walk it in score order, cap at `max_per_paper` (default 2) evidence items per paper, and merge chunks that are adjacent in the source document (consecutive `chunk_id`, same paper) into a single item instead of counting them separately or showing near-duplicate neighbours. `chunk_id` is a global auto-increment assigned in parse order, which is document order within a paper, so adjacency is a plain `chunk_id` distance-1 check — no need for section/offset math.

**Why cap 2, not 1:** a paper legitimately can have two independent, non-adjacent pieces of relevant evidence (e.g. a methods paragraph and a results paragraph). Capping at 1 would throw that away; capping at 2 stops single-paper domination while still allowing it.

**Result:** re-ran `eval_queries.py` (now imports `search_chunks` from `query.py` so the eval measures actual query-time behaviour, not a separate raw top-k calculation). Mean distinct papers in top-5: **2.7 → 4.14**. Top-1 scores and the separation gap are unchanged (the cap only affects which chunks fill positions 2-5, never which chunk is best) — diversity was a free fix, not a quality/diversity tradeoff.

**Consequence for the next roadmap item (agreement counts):** this is the prerequisite it was blocked on — counting "N independent papers assign X to Y" requires results to actually span N papers first.

---

### 2026-07-24 — Agreement counts over `peak_tables` (`query.py`: `agreement_for_range`, `agreement_report`)
**Question:** replace/augment the flat `--peaks` row dump with an actual "N independent papers agree" count, per the blueprint's stated preference for countable agreement over blended confidence scores.

**Decision:** count distinct `paper_id`, not rows. A review like Czamara 2015 tabulates one vibrational mode across 35 lipid species in a single table — that is one paper's internal breadth, not 35 independent observations, so a naive `COUNT(*)` would let one well-tabulated review dominate an agreement count. Grouping by paper first and counting distinct papers is the actual claim being made ("N independent sources agree").

Assignment text is left as free text, not normalised into a canonical label — every paper phrases it differently (Greek letters vs spelled-out names, different mode notations, different levels of specificity). Attempting to merge "δ(CH₂)" / "CH₂ bending modes" / "Symmetric deformation (Scissor)" into one canonical label is a real NLP problem parked for later; showing the raw variants is more honest than silently forcing agreement that hasn't been verified. It also surfaces a genuinely useful side effect: querying 1440 cm⁻¹ shows some hits are Isoleucine or a lignin ring vibration, not lipid CH₂ at all — the region is not lipid-exclusive, which the raw text makes visible and a forced canonical label would have hidden.

**Two entry points added to `query.py`:**
- `--peaks W --tol T` now reports `agreement_for_range`: papers grouped, one block per paper, all its rows in that window shown together.
- `--agreement-report [--min-papers N] [--bin-width W]`: corpus-wide leaderboard via pure SQL — bin by row midpoint (`ROUND(((low+high)/2.0)/bin_width)*bin_width`), `GROUP BY` bin, `HAVING COUNT(DISTINCT paper_id) >= min_papers`.

**Measured, `--min-papers 4`, 5 cm⁻¹ bins on the current (v4b, 217-paper) corpus:** 1005 cm⁻¹ (phenylalanine ring breathing) leads at 12 papers; 1440 (CH₂ deformation) at 9; 1130 and 1660 at 8. All three are textbook Raman markers — the ranking is chemically sane, not an artifact.

**Known coarseness:** fixed-width binning can split one physical band across two adjacent bins near an edge (1440 and 1445 both surfaced as separate 6-9-paper bins in one run — plausibly the same band, split by rounding). Not fixed; narrowing/widening `--bin-width` is the current lever, same unresolved tension as the tolerance-window problem already parked in `questions.md`.

---

### 2026-07-27 — PDF prose ingestion (`parse_pdf.py`), run for real
**Question:** `parse_pdf.py` existed but had never been run against the DB (no `source` column present) — 61 of the 62 raw PDFs had no prose in the corpus at all, only whatever peak-table data a handful had via bespoke extractors.

**Bug found before running:** the file's own comment said Czamara should be skipped ("skip its prose... skip the file we know is table-only noise") but no skip code existed. `eval_queries.py` (line 20) independently documents the invariant that Czamara "isn't chunked/embedded, only its peak table data" — its review prose is a trusted reference source (`parse_czamara.py`, `paper_id='czamara2015'`), not meant to be a retrievable paper alongside everything else. Running the script as-is would have created a second paper (`pdf:czamara2014`) and silently broken that invariant. Fixed: `parse_pdf.py` now explicitly filters `czamara2014.pdf` out of its glob, with a comment pointing at why.

**Result:** 61 PDFs → 4,632 prose chunks, tagged `source='pdf'`. Corpus: 220→281 papers, 7,931→12,563 chunks. DB snapshotted to `data/papers_pre_pdf_ingest_safety.db` first.

**Re-embedding deferred, not done today.** `embed.py`'s output note in `parse_pdf.py` ("must be updated to embed only unvectorised chunks") is stale — the actual `embed_chunks` code already re-embeds every chunk unconditionally every run, so no code change was needed there. Chose to hold off running it locally (~15-20 min extrapolated from 27 min/16k chunks on M2 MPS) and instead batch it with the Colab Pro A100 move (`PROGRESS.md` item 8), rather than eating the local wait twice. Until that re-embed runs, the new `source='pdf'` chunks have `vector_row_idx=NULL` and are invisible to retrieval — `chunk_vectors.npy` still reflects the pre-ingestion 7,931-chunk state.

---

### 2026-07-28 — Knowledge graph layer: Neo4j AuraDB (cloud), not local Docker
**Question:** the user wants to *see* how lipids, classes, papers, and citations relate — a visualization/exploration need, not a retrieval-performance need. Does this contradict the earlier decision (`PROGRESS.md` "Deliberately not built") to reject pgvector/FAISS as unnecessary infra?

**Decision:** no — different need, different verdict. The vector-DB rejection was about *approximate search at a scale we don't have*; brute-force cosine is exact and fast enough, so ANN would only cost RAM. A graph database's value here isn't query speed, it's that Cypher traversal (and Neo4j Browser's visualization) makes relationship structure legible in a way SQL joins over `papers.db` don't, even at this small scale. Built it as a **derived, rebuildable layer** — `papers.db` stays canonical, `scripts/build_graph.py` re-derives the graph from it (same shape as `resolve_lipid_identity.py`'s cache: wipe-and-rebuild, not migrate).

**Where it runs:** Neo4j AuraDB Free (cloud), not local Docker — the 8GB M2 Air already runs MLX inference outside Docker specifically to avoid RAM contention (see `PROGRESS.md`); adding a local Neo4j instance would recreate that exact problem for no benefit, since the free cloud tier costs nothing but a network hop.

**Graph model, validated against the live DB before building:**
- `peak_tables` has 546 rows carrying a species `origin` tag (104 distinct origins), but only 29 resolve to a full `lipid_identity` record (the Czamara 35-species table) — those 29 became `Lipid` nodes; unresolved origins don't get a node (would need a second identity-resolution pass, out of scope here).
- `references` has 11,307 rows with a DOI, but only 867 resolve to a DOI this graph actually has a `Paper` node for (in-corpus `papers.doi` or tracked `frontier.doi`) — the rest point outside both tables and would be dangling edges, so `fetch_citations` filters to resolvable pairs before ever touching Neo4j, rather than sending unresolvable rows and letting `MERGE`'s `MATCH` silently no-op them.
- Including `frontier` (papers tracked by the citation-frontier fetcher but not ingested) as lightweight `Paper` nodes alongside `papers` roughly triples the citation graph's reach for free — 163 frontier-only papers added, vs 255 in-corpus.

**Deliberately not built:** chunk-level `(Paper)-[:MENTIONS]->(Lipid)` edges. Would need free-text matching lipid names against prose chunks — a real NLP task, not proportionate to what was asked. Parked in `questions.md`.

---

### 2026-07-28 — Colab re-embed surfaced junk PDF titles; parse_pdf.py's title metadata was never trustworthy

**Question:** `embed.py` was run on Colab's A100 (`colab/embed_chunks.ipynb`) to finally vectorize the 12,563-chunk corpus, including the 4,632 `source='pdf'` chunks from PDF ingestion that had sat with `vector_row_idx=NULL` since 2026-07-27. Did it work?

**First failure, immediately visible:** `query.py`/`answer.py` crashed (`TypeError: 'NoneType' object is not subscriptable`) the first time a PDF-sourced chunk ranked in a top-k result — a paper title was `NULL`. 23 of 281 papers had this: `parse_pdf.py`'s `title_of()` reads PDF metadata unconditionally and many raw PDFs simply have no title field. This was invisible until today because those chunks had no vector and were unreachable by any query before the Colab run.

**Second failure, not caught by the first fix:** patched the crash (fall back to `paper_id`/`doi` in three print sites), then wrote `scripts/backfill_titles.py` scoped to `title IS NULL` and resolved all 23 via Crossref (5 with a DOI) or arXiv's `id_list` API (18 arXiv-only, extracting the id from the `pdf:arxiv_...` paper_id). Verification query still showed a garbage title: `'ac5b04468 1..9'` for an ACS paper. Not `NULL` — this is a second, worse failure shape: `title_of()` trusts *whatever string is in the PDF's metadata title field*, and for many publishers that isn't the paper title at all. Found 13 more across the 61 `pdf:` papers: `"Microsoft Word - Nature_Protocols_Final_BW.docx"`, `"No Job Name"`, bare manuscript IDs (`"ac500014b 1..5"`, `"lsa201538 1..10"`, `"pone.0005189 1..12"`), even a raw `"doi:10.1016/j.colsurfb.2004.12.021"` string used as a title. None of these are `NULL`, all look like plausible strings at a glance — exactly the "output looks plausible, nothing errors, it's just quietly wrong" failure class `PROGRESS.md` problem #1 already names for text extraction generally, just showing up in a metadata field instead of body text this time.

**Fix:** rewrote `backfill_titles.py` to stop trusting PDF-internal title metadata as authoritative at all. It now targets every `pdf:`-sourced paper (not just `NULL` ones) and always attempts Crossref (has DOI) or arXiv (no DOI) first, keeping the existing PDF-metadata title only as a last resort if the network lookup fails. Result: 13/61 titles corrected, one genuine miss kept as-is (`arxiv_0606030v1` — not a valid arXiv ID format, likely a mis-parsed filename from the original `fetch_arxiv.py`/manual-acquisition step, no lookup possible).

**Third-order consequence, the one that would have been silent:** `embed.py` prepends title to the chunk text before embedding (`f"{title}\n{section}\n\n{text}"`) — a bare paragraph is unretrievable without knowing which paper it's from. The title fix ran *after* the Colab embed, so 1,122 chunks belonging to those 13 papers had already been vectorized with the junk title baked into the BGE input. Re-embedding the full 12,563-chunk corpus again for a 1,122-chunk fix would have meant another full Colab round-trip; instead, patched just those 1,122 rows in `chunk_vectors.npy` in place (same `vector_row_idx` positions, re-encoded with the corrected title), locally on M2 MPS — small enough to not need the A100. Verified: `chunk_vectors.npy` shape unchanged, retrieval on two independent test queries returns clean top hits with no crash and correct titles.

**Consequence for the pipeline order:** title correctness must be settled *before* embedding, not after — `backfill_titles.py` should run right after `parse_pdf.py` and before `embed.py` for any future PDF ingestion, or this same stale-embedding problem recurs. Not yet enforced in code (no ordering check), just documented here.

**Separately, unrelated to titles:** the Colab run logged `WARNING:adapters.model_mixin:There are adapters available but none are activated for the forward pass` during SPECTER2 paper-embedding, raising a concern that the proximity adapter silently wasn't active. Checked against this file's own 2026-07-15 entry ("SPECTER2 proximity adapter: tried, made no difference") — already proven with n=28 and n=425 that adapter-vs-no-adapter is statistically identical for this corpus (topical homogeneity, not a model artifact). Today's n=281 measurement (mean 0.879, std 0.027) lines up with those prior numbers, so even if the adapter was inactive this run, it already wouldn't have mattered. No re-run needed — did not repeat work already falsified.
