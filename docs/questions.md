# Open Questions

Parked problems and undecided calls. When one gets resolved, write the reasoning into `solutions.md` and delete it here.

---

### PMC alone is the wrong corpus — the Raman methods literature isn't in it
PubMed/PMC is a **biomedical** index. The Raman spectroscopy methods literature lives in analytical chemistry journals (Wiley/RSC/Elsevier) that it indexes poorly or not at all. Measured 2026-07-15:

- `"Raman spectroscopy of lipids"[Title]` → **0 hits in PubMed**. The Czamara et al. 2015 review — the blueprint's "close to a ready-made reference table for this project's exact scope" — is not in PubMed, let alone PMC.
- *J Raman Spectrosc* → **8** Raman+lipid papers total in PubMed. For the field's flagship journal that means it's barely indexed.
- *Anal Chem* → 215 total, only **41** in PMC (19%).

A PMC-only corpus gets biology papers that happen to use Raman, and misses the spectroscopy papers where peak assignment tables are densest. It would look healthy by paper count while missing the reference data the tool exists to serve.

Options not yet evaluated: Europe PMC (broader, indexes preprints, some non-PMC content); Crossref + Unpaywall (DOI → any OA copy, publisher-agnostic); bioRxiv; institutional access via BU for the paywalled reviews; manual acquisition of the handful of high-value review tables (Czamara being one).
**Matters because:** this decides whether Phase 1 ingestion is one fetcher or several, and whether the highest-value tables are reachable programmatically at all. Blocks corpus build.

### Peak positions depend on experimental conditions — deferred to Phase 5
A wavenumber is a measurement, not an identifier. Peak position and relative intensity shift with laser wavelength (532/785/1064 nm), sample state (dried film vs solution vs cell vs tissue), temperature and lipid phase (gel vs liquid-crystalline changes chain packing, which moves the CH₂ and C–C skeletal bands), and with baseline-correction and normalization choices made in software before anyone reads a number off the plot.

Consequences to work out when we get there:
- An evidence item probably can't be `peak → assignment → citation`. It likely needs the measurement conditions attached on both sides, or the RAG panel states context-free facts that are only true under a setup it never mentions.
- Matching needs a tolerance window, not equality. Czamara's 1663 vs an observed 1659 may be the same assignment under different conditions, or a different species — the lookup can't distinguish without conditions. One global tolerance probably won't do; it likely varies with band width and how resolved the region is.
- Papers may not report conditions consistently. If many say "a band at 1663 cm⁻¹" with no laser/phase/prep, the conditions field is mostly null and we must decide what the RAG panel does with a citation it can't fully qualify — flag it as unqualified, or drop it.
- Ingestion boundary is unsettled: vendor formats (Renishaw `.wdf`, Bruker OPUS, `.spc`, JCAMP-DX, ad-hoc CSV) disagree on whether raw or processed traces are stored and whether acquisition metadata survives export. Blueprint assumes peaks arrive already extracted from the existing PCA/DL pipeline, which would make formats that pipeline's problem — needs confirming.

**Decision:** deferred. Build the RAG track (Phases 1–2) first; revisit at Phase 5 (Raman integration).
**Matters because:** it may change the evidence schema, so it should be settled before the Context Builder's format hardens in Phase 4.

### Corpus expansion should follow the citation graph, not more keyword queries
Our 28 papers cite 2,600 references, 2,329 with DOIs. The most-cited are the field's foundational papers (Freudiger label-free imaging, 7/28; Kong et al. Raman for cancer, 6/28; Haka breast cancer, 5/28). **The Czamara review (10.1002/jrs.4607) is cited by 4 of 28** — our corpus points straight at the paper we established is unreachable via PubMed/PMC/Europe PMC but *is* green OA at ruj.uj.edu.pl.
**Matters because:** snowball sampling from the reference graph finds what the field considers load-bearing, which a keyword query cannot. It also gives a ranked, evidence-based target list for the Crossref/Unpaywall track — instead of guessing which paywalled papers are worth chasing.

### Retrieval quality is unmeasured
The end-to-end query returns plausible results (0.80 cosine on a cholesterol question, correctly surfacing the 1655/1443 unsaturation ratio). Plausible is not measured. There is no question set, no relevance judgement, no baseline.
**Matters because:** "document the embedding model" was a Phase 1 goal, and documenting an unmeasured choice records a guess. Blocks the BGE/MedCPT/SPECTER2 bake-off, and blocks knowing whether Phase 2 (paper QA) is even worth starting.
