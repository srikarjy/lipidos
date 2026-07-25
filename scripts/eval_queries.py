"""A first evaluation set: what a Raman/lipid scientist actually asks.

These are NOT "what is the band at 1440" -- a working scientist knows that.
These are the questions that cost real time: discriminating similar species,
converting band ratios into physical quantities, and methodological choices
that determine whether a measurement works at all.

Grouped by the kind of work each supports. `expect` records what a good answer
must contain, so a hit can be judged rather than eyeballed.

Three tiers, by provenance (see docs/solutions.md):
  - CONSTRUCTED: the original 16, written by guessing what a scientist asks.
    Judged only by top-1 cosine score -- there is no ground-truth paper.
  - MINED: pulled from corpus papers' own Introduction goal statements
    (scripts/mine_bench_questions.py). Carries `known_paper_id` -- the paper
    that wrote the sentence should itself surface in the results, so these
    are judged by recall (was that paper actually retrieved), not just score.
  - REFERENCE: discrimination facts stated directly in the Czamara 2015
    review (data/marker/czamara2014/czamara2014.md), a trusted domain source.
    No ground-truth paper (Czamara isn't chunked/embedded, only its peak
    table is parsed) -- judged by whether `expect` content shows up in the
    top hits, same as CONSTRUCTED.

Usage:
    .venv/bin/python scripts/eval_queries.py          # run all, report coverage
    .venv/bin/python scripts/eval_queries.py -v       # show top chunk per query
"""

QUERIES = [
    # --- ratios as physical readouts: the core analytical move ---
    dict(id="ratio-order", group="ratio",
         q="what does the I2880/I2850 intensity ratio measure about lipid chain packing?",
         expect="lateral chain order / packing / interchain coupling"),
    dict(id="ratio-unsat", group="ratio",
         q="how is the ratio of the 1655 and 1440 bands used to quantify lipid unsaturation?",
         expect="C=C vs CH2 deformation ratio as unsaturation index"),
    dict(id="ratio-gauche", group="ratio",
         q="trans gauche conformation from the C-C skeletal stretching region 1060-1130",
         expect="1064/1128 all-trans vs 1080 gauche disorder"),

    # --- discrimination: the thing that is genuinely hard ---
    dict(id="disc-chol-ester", group="discriminate",
         q="how to distinguish free cholesterol from cholesteryl esters in Raman spectra",
         expect="1670 ester C=O, 702 sterol ring, 1440 shifts"),
    dict(id="disc-sm-pc", group="discriminate",
         q="spectral differences between sphingomyelin and phosphatidylcholine",
         expect="amide bands in SM, distinct headgroup/backbone features"),
    dict(id="disc-lipid-protein", group="discriminate",
         q="separating overlapping lipid and protein contributions near 1440 and 1450",
         expect="CH2 deformation overlap, amide I vs C=C at 1655"),
    dict(id="disc-sat-unsat", group="discriminate",
         q="saturated versus unsaturated fatty acid Raman marker bands",
         expect="3013 =C-H, 1655 C=C, 1265 cis =C-H rock"),

    # --- method: determines whether the measurement works at all ---
    dict(id="meth-laser", group="method",
         q="which excitation wavelength avoids fluorescence background in lipid tissue Raman",
         expect="785/1064 NIR reduces autofluorescence vs 532"),
    dict(id="meth-baseline", group="method",
         q="baseline correction and fluorescence removal for lipid Raman spectra",
         expect="polynomial/SNIP/asymmetric least squares, Savitzky-Golay"),
    dict(id="meth-srs", group="method",
         q="when to use SRS or CARS instead of spontaneous Raman for lipid droplets",
         expect="speed/sensitivity tradeoff, CH2 2850 contrast"),

    # --- physical state: where the conditions problem bites ---
    dict(id="phase-gel", group="phase",
         q="Raman markers of gel to liquid crystalline phase transition in lipid bilayers",
         expect="CH2 stretch region changes, chain order, temperature"),
    dict(id="phase-temp", group="phase",
         q="how do lipid CH2 stretching bands shift with temperature",
         expect="2850/2880 intensity and position vs chain melting"),

    # --- biological application ---
    dict(id="bio-ld", group="biology",
         q="lipid droplet composition and unsaturation in cancer cells measured by Raman",
         expect="LD unsaturation, 1655/1440, cell line differences"),
    dict(id="bio-perox", group="biology",
         q="Raman detection of lipid peroxidation products",
         expect="conjugated diene, C=C loss, aldehyde markers"),

    # --- out of domain: MUST score low or the floor is broken ---
    dict(id="ood-bread", group="OUT-OF-DOMAIN",
         q="how do I bake sourdough bread at home", expect="(should not match)"),
    dict(id="ood-crispr", group="OUT-OF-DOMAIN",
         q="CRISPR gene editing of the BRCA1 promoter", expect="(should not match)"),

    # --- MINED: papers' own stated aims, ground-truthed to the paper itself ---
    dict(id="mined-astaxanthin", group="mined", known_paper_id="PMC10374751",
         q="tracking endothelial cell uptake of free versus liposome-encapsulated astaxanthin using Raman microscopy",
         expect="AXT uptake, Raman vs fluorescence imaging"),
    dict(id="mined-soybean", group="mined", known_paper_id="PMC10436576",
         q="predicting protein and lipid content of a single soybean seed from Raman hyperspectral imaging",
         expect="hyperspectral imaging, chemometric prediction model"),
    dict(id="mined-pfas", group="mined", known_paper_id="PMC12337092",
         q="how do PFOA and PFBS differ in their effect on model lipid membrane biophysics",
         expect="perfluoroalkyl substances, membrane order/packing changes"),
    dict(id="mined-cars-celegans", group="mined", known_paper_id="PMC2940797",
         q="using CARS microscopy as a label-free method to quantify fat storage in C. elegans",
         expect="CARS vs label-based (Nile red/Oil Red O) fat quantitation"),
    dict(id="mined-cars-steatosis", group="mined", known_paper_id="PMC3511365",
         q="can coherent anti-Stokes Raman scattering microscopy diagnose hepatic microvesicular steatosis label-free",
         expect="CARS clinical diagnosis, lipid droplet imaging in liver tissue"),
    dict(id="mined-ibuprofen", group="mined", known_paper_id="PMC4886502",
         q="using surface-enhanced vibrational spectroscopy to chemically differentiate ibuprofen analogs in lipid bilayers",
         expect="SERS/SEIRA, drug-membrane interaction, analog discrimination"),
    dict(id="mined-sers-chol-aunp", group="mined", known_paper_id="PMC6714513",
         q="effect of gold nanoparticle size on SERS characterization of cholesterol-modified lipid membranes",
         expect="SERS enhancement, AuNP size, cholesterol-lipid systems"),
    dict(id="mined-evoo", group="mined", known_paper_id="PMC7073519",
         q="non-targeted authentication of extra virgin olive oil using vibrational spectroscopy and pattern recognition",
         expect="EVOO authentication, adulteration screening, chemometrics"),
    dict(id="mined-srs-ins", group="mined", known_paper_id="PMC9072573",
         q="hyperspectral stimulated Raman scattering microscopy of lipid bilayer dynamics during infrared neural stimulation",
         expect="hsSRS, INS, lipid bilayer vibrational signatures"),
    dict(id="mined-bmad", group="mined", known_paper_id="PMC9727239",
         q="does Raman microspectroscopy reveal unsaturation heterogeneity within individual lipid droplets and distinguish labile from stable bone marrow adipocyte subtypes",
         expect="single lipid droplet unsaturation, BMAd subtype discrimination"),
    dict(id="mined-platinum", group="mined", known_paper_id="PMC10679116",
         q="FT-Raman spectral marker to determine platinum sensitivity versus resistance in ovarian cancer patients",
         expect="multivariate/ML classification, platinum-resistance biomarker"),
    dict(id="mined-cancer-cluster", group="mined", known_paper_id="PMC6414003",
         q="discriminating different cancer types by clustering Raman spectra with a superparamagnetic clustering method",
         expect="unsupervised clustering, biochemical composition differences across cancer types"),
    dict(id="mined-oral-lesion", group="mined", known_paper_id="PMC7913942",
         q="can Raman spectroscopy in the fingerprint region discriminate malignant, dysplastic, and normal oral epithelial cells by nucleic acid, protein, and lipid profile",
         expect="OSCC vs dysplastic vs normal, fingerprint region discrimination"),
    dict(id="mined-cervix", group="mined", known_paper_id="PMC9308703",
         q="in vivo Raman spectroscopy biochemical fingerprint of cervical remodeling during labor",
         expect="cervical ripening, collagen/lipid changes, latent vs active labor"),
    dict(id="mined-leukemia", group="mined", known_paper_id="PMC9789313",
         q="identifying spectral differences in blood serum of leukemic versus non-leukemic children using a Raman spectral model",
         expect="serum Raman, pediatric leukemia classification"),

    # --- REFERENCE: discrimination facts stated in the Czamara 2015 review ---
    dict(id="ref-cpa-vs-csa", group="reference",
         q="how to distinguish cholesteryl palmitate from cholesteryl stearate by Raman spectrum",
         expect="near-identical CHL backbone bands (702/430), CSA shows extra 1468/1428 shoulders vs CPA's single 1298 PA-chain band"),
    dict(id="ref-pe-vs-pc", group="reference",
         q="which Raman band distinguishes phosphatidylethanolamine from phosphatidylcholine",
         expect="760 cm-1 ethanolamine band unique to PE; choline N+(CH3)3 bands at 719/876 unique to PC"),
    dict(id="ref-sm-vs-pc-ch", group="reference",
         q="how does sphingomyelin differ from phosphatidylcholine in the CH stretching region 2800-3100",
         expect="SM: higher relative 2882 intensity, no 3007 unsaturated =CH band, vs PC"),
    dict(id="ref-ester-vs-free-chol", group="reference",
         q="which Raman band indicates ester C=O in cholesteryl esters versus free cholesterol",
         expect="~1740 cm-1 C=O band present in esters, absent in free cholesterol"),
]

if __name__ == "__main__":
    import argparse, sqlite3, warnings
    warnings.filterwarnings("ignore")
    import numpy as np, torch, torch.nn.functional as F
    from pathlib import Path
    from transformers import AutoModel, AutoTokenizer

    ap = argparse.ArgumentParser()
    ap.add_argument("-v", "--verbose", action="store_true")
    a = ap.parse_args()

    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from query import search_chunks

    DATA = Path(__file__).resolve().parent.parent / "data"
    tok = AutoTokenizer.from_pretrained("BAAI/bge-base-en-v1.5")
    model = AutoModel.from_pretrained("BAAI/bge-base-en-v1.5").eval()
    db = sqlite3.connect(DATA / "papers.db")

    def enc(q):
        with torch.inference_mode():
            b = tok(["Represent this sentence for searching relevant passages: " + q],
                    return_tensors="pt", truncation=True, max_length=512)
            return F.normalize(model(**b).last_hidden_state[:, 0], p=2, dim=1)[0].numpy().astype(np.float32)

    print(f"{'id':22} {'group':10} {'top':>6} {'papers/5':>9} {'recall@10':>10}  top hit")
    print("-" * 110)
    in_dom, ood = [], []
    mined_hits = mined_total = 0
    for item in QUERIES:
        k_check = 10 if "known_paper_id" in item else 5
        hits = list(search_chunks(db, enc(item["q"]), k=k_check))  # diversity-aware: cap 2/paper
        top_score, (text, sec, xrefs, spans, title, doi, paper_id, year) = hits[0]
        ndist = len({row[6] for _, row in hits})

        recall_str = ""
        if "known_paper_id" in item:
            mined_total += 1
            found_rank = next((i for i, (_, row) in enumerate(hits)
                                if row[6] == item["known_paper_id"]), None)
            if found_rank is not None:
                mined_hits += 1
                recall_str = f"hit@{found_rank + 1}"
            else:
                recall_str = "MISS"

        (ood if item["group"] == "OUT-OF-DOMAIN" else in_dom).append(top_score)
        print(f"{item['id']:22} {item['group']:10} {top_score:6.3f} {ndist:>9} {recall_str:>10}  {title[:40]}")
        if a.verbose:
            print(f"{'':22} -> {text[:150]}...")
    print("-" * 110)
    print(f"in-domain  top-1 score: mean {np.mean(in_dom):.3f}  min {np.min(in_dom):.3f}")
    print(f"OUT-OF-DOM top-1 score: mean {np.mean(ood):.3f}  max {np.max(ood):.3f}")
    gap = np.min(in_dom) - np.max(ood)
    print(f"\nseparation gap (worst real query - best nonsense query): {gap:+.3f}")
    print("a positive gap means a score floor could reject out-of-domain queries.")
    print("a negative gap means NO threshold can separate them -- nonsense outranks real questions.")
    if mined_total:
        print(f"\nmined-question recall@10 (source paper actually retrieved): "
              f"{mined_hits}/{mined_total} ({mined_hits/mined_total:.0%})")
