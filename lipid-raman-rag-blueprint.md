# Lipid Raman Research Assistant — Project blueprint

## Problem statement

Two separate pain points in the current workflow:

1. **Ungrounded literature search.** General-purpose LLMs (e.g. ChatGPT) return summaries on lipid/Raman topics with no way to verify what's real vs. invented. No citation trail, no way to check the split between accurate and hallucinated content.
2. **Manual Raman peak identification.** Peaks from spectra (processed via PCA/deep learning) are matched to molecules by hand, cross-referencing literature and databases — repetitive, slow, and not easily reproducible or shareable.

A prior attempt to solve problem 1 (downloading a model, embedding ~50-60k papers) produced a system that "doesn't respond properly," and the build process itself wasn't documented — it can't be reproduced, debugged, or handed off. That failure mode is the thing this blueprint is designed to avoid repeating.

## Design principle

**Don't ask a model to recall facts. Ask it to reason over facts it was just given, with a visible source for each one.**

This applies to both problems:
- Literature grounding = retrieval + citation, not generation from memory.
- Peak identification = structured lookup + retrieved evidence, not free-form guessing.

Fine-tuning is reserved for the one part that's genuinely reasoning (combining peak data with retrieved evidence into an interpretation) — not for storing facts, since models are unreliable at faithfully recalling precise structured data (IDs, InChI strings, exact wavenumbers) from trained weights.

## Architecture overview

```
Paper corpus                         Lipid databases
(curated Raman/lipid papers)         (LIPID MAPS, PubChem, SwissLipids)
      │                                     │
Chunk + embed                        Structured lookup
(domain embedding model)             (InChI → structure/class/refs)
      │                                     │
      └──────────────┬──────────────────────┘
                      ▼
              Retrieval layer
        (top-k papers + database facts)
                      ▼
              Context Builder
   • dedupe overlapping evidence
   • rank by confidence/relevance
   • attach citations per claim
   • summarize long passages
                      ▼
              Grounded context
      (facts + citations, no free generation)
                      │
                      │        Raman peak data
                      │        (from existing PCA/DL pipeline)
                      │              │
                      └──────┬───────┘
                             ▼
                    Fine-tuned Gemma
        (reasons over peaks + retrieved facts)
                             ▼
              Cited interpretation
      (claim → evidence → paper → sentence → DOI)
```

### Component responsibilities

| Component | Job | Notes |
|---|---|---|
| Paper corpus | Source of literature evidence | Curated subset (Raman + lipids), not the full 50-60k — smaller corpus = better retrieval precision |
| Embedding/vector search | Semantic retrieval over papers | Use a domain-tuned embedding model (e.g. SPECTER2-style), not a generic one |
| Lipid databases | Source of structured facts | Direct exact-match lookup (InChI → class/structure/references) — no embedding needed, no hallucination risk |
| Context Builder | Turns raw retrieval into structured evidence | Dedupe, rank, cite, summarize — this is what keeps the LLM from drowning in 30 raw chunks |
| Fine-tuned Gemma | Reasoning only | Combines peak data + grounded context into an interpretation; does not store facts in weights |
| Output | Traceable answer | Every claim traces to evidence → paper → sentence → DOI |

## Deferred / not in scope for v1

These were discussed and are worth revisiting later, but adding them now risks recreating an unmaintainable system (the exact failure this project is meant to fix):

- **Evidence graph** (lipid/peak/paper/experiment nodes with typed edges like `supports`, `contradicts`, `validated_by`) — powerful but a standalone engineering project; treat as a v2+ idea once the linear pipeline is proven.
- **Cross-component confidence scoring** (blending retriever/database/classifier/LLM confidence into one number) — risks implying more rigor than it has, since the scores aren't on comparable scales. Prefer surfacing each component's own signal separately (e.g. "3 of 3 databases agree," "peak classifier flagged this as ambiguous") rather than one blended score.

## Build phases

**Phase 1 — Paper ingestion**
PDF → chunk → embed → vector DB. Document the exact embedding model and chunking method used (this is the fix for "I don't know what I did").

**Phase 2 — Paper QA (no Raman yet)**
Question → retriever → Gemma (base, not fine-tuned) → answer with citations. Validate that retrieval quality and citation grounding actually work before adding complexity.

**Phase 3 — Lipid knowledge layer**
Question → retriever + LIPID MAPS + SwissLipids → merged evidence. Adds the structured-lookup track alongside the paper track.

**Phase 4 — Context Builder**
Merge retrieved papers + structured facts + citations + confidence; dedupe. This is the layer that makes phase 5's fine-tuning input clean.

**Phase 5 — Raman integration**
Peaks → existing CNN/PCA model → predicted candidates → grounded context → Gemma (still base model at this point).

**Phase 6 — Fine-tuning**
Train Gemma on (peaks + retrieved evidence + database facts) → expert interpretation pairs. This is the right point to fine-tune, because by now the model has real grounded context to learn to reason over — it's learning a reasoning pattern, not memorizing facts.

## Output design — two-panel separation (critical requirement)

**Problem this solves:** if retrieved evidence and the fine-tuned model's interpretation are merged into one piece of text, there's no way to tell which sentence is a sourced fact and which is model-generated reasoning. This recreates the original "can't tell what's grounded" problem one layer deeper.

**Fix: never merge them. Show two separate panels, always.**

- **RAG panel** — raw retrieved evidence only. Peak → assignment → source → citation. Nothing generated here; every line is either directly retrieved or it doesn't appear. This panel is checkable in a binary way: does the cited paper/database actually say that?
- **LLM panel** — the fine-tuned model's interpretation, shown separately, never stitched into the same text as the RAG panel.

**The one remaining discipline this requires:** the LLM panel must reference which specific evidence items it's drawing on (e.g. "based on evidence #2 and #4") rather than asserting conclusions with no pointer back to the RAG panel. This is a much lighter requirement than trying to merge-and-reconcile the two into one narrative — it's just "point to what you were given," enforced through the phase 6 training data format (every target interpretation in the training set should reference evidence items by ID, so the model learns to do this by default).

No consistency-checker or reconciliation step is needed, because the two outputs are never combined into one response in the first place.

## Where the data comes from

**Spectral reference data (wavenumber → assignment):**
- Czamara et al., "Raman spectroscopy of lipids: a review" (2015) — Raman spectra and detailed peak analysis for 35 lipids (fatty acids, triacylglycerols, cholesterol, cholesteryl esters, phospholipids). Close to a ready-made reference table for this project's exact scope.
- "Open Raman spectral library for biomolecule identification" (2025) — an open spectral library covering lipids and saccharides, built specifically for spectrum-matching/identification tasks.
- Structural-property papers (e.g. Royal Society Open Science ratiometric fatty acid study) — document specific peak-combination-to-property relationships (e.g. peaks at 3013, 1663, 1264 cm⁻¹ correlating with fatty acid unsaturation), useful as reasoning examples.
- General open Raman databases (RRUFF, Raman Open Database) exist but are mineral/crystallography-focused, not lipid-specific — useful as infrastructure examples, not as a direct data source here. The proprietary Wiley KnowItAll library covers biomolecules broadly but isn't open access.

**Literature corpus (for RAG):**
- PubMed / PMC — full-text, citable papers; there's a PubMed tool available in this environment to pull real papers directly rather than bulk-scraping.
- bioRxiv — preprints not yet indexed in PubMed.
- Seed with known review papers (Czamara review, RSC Analyst lipid/Raman review) — these cite dozens of primary studies that already reason peak-by-peak, which is valuable target-side data for phase 6.

**Structured lipid/molecule facts:**
- **LIPID MAPS** — confirmed as a free, no-authentication REST API (`https://www.lipidmaps.org/rest`), callable directly from Python or any scripting language. No corpus-building or embedding needed for this piece — it's a direct lookup, not a retrieval problem.
  - **LMSD (Structure Database)**: 50,000+ curated lipid structures, each with an LM_ID, InChIKey, molfile, SMILES, systematic name, and standard lipid classification. This is the primary source for the InChI → structure/class lookup.
  - **LMISSD (In-Silico Structure Database)**: 1M+ computationally-expanded structures, useful as a fallback when a specific measured lipid isn't in the curated LMSD but its chain composition can still be matched.
  - Request pattern: `/rest/{context}/{input_item}/{input_value}/{output_item}/{output_format}` (e.g. query by `lm_id`, `pubchem_cid`, or InChIKey; output as JSON or table).
  - This is the fastest concrete win in the whole pipeline — no data collection required, just a client wired to the existing API.
- SwissLipids — lipid classification, cross-referenced structure data.
- PubChem — general compound structure/property lookup.

**Phase 6 target-side data (expert interpretations), in priority order:**
1. Reasoning sentences already written in paper results/discussion sections (real, grounded, no extra labor beyond extraction).
2. Any existing annotated spectra from his own lab (notebooks, past reports) — best fit if it exists, since it's already domain-matched and expert-verified.
3. Synthetic draft interpretations generated from retrieved evidence, but only after human review confirms correctness — never used unreviewed.

## Immediate next step

Since the original corpus/embeddings can't be recovered, Phase 1 starts from scratch:
1. Re-gather or re-organize the paper set (curated subset, not the full 50-60k).
2. Pick and document a domain-appropriate embedding model.
3. Chunk at paragraph/section level, not whole-PDF.
4. Store in a vector DB with a written record of every choice made, so this is reproducible and debuggable going forward.
