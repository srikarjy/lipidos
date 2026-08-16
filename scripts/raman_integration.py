"""Phase 5: Raman Integration — interface to existing PCA/CNN pipeline.

This module provides the bridge from a Raman spectrum → peak extraction
(PCA/CNN) → candidate lipids → grounded context (Context Builder) → LLM.

Since the PCA/CNN pipeline is external (not in this repo), this defines
the interface and provides a mock/demo implementation that shows how
the integration works end-to-end.

Pipeline:
    Raw spectrum → PCA/CNN (external) → peak list (cm⁻¹) →
    peak_set_match → Context Builder → LLM panel

Usage:
    # With real peaks from your PCA/CNN pipeline:
    .venv/bin/python scripts/raman_integration.py --peaks 2850,2880,1440,1660

    # With a simulated spectrum (demo):
    .venv/bin/python scripts/raman_integration.py --demo
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Optional

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

from context_builder import build_context_block
from query import match_peak_set, lipid_identity
import sqlite3

DATA = ROOT / "data"
DB_PATH = DATA / "papers.db"

DEMO_SPECTRA = {
    "cholesteryl_oleate": {
        "description": "Cholesteryl oleate (COA) - mono-unsaturated sterol ester",
        "peaks": [702, 1005, 1064, 1130, 1265, 1303, 1440, 1659, 1740, 2850, 2880, 2930, 3009],
        "expected_top": "COA"
    },
    "triolein": {
        "description": "Triolein (TOA) - triacylglycerol with 3 oleic acid chains",
        "peaks": [702, 1005, 1064, 1130, 1265, 1303, 1440, 1655, 1740, 2850, 2880, 2930, 3009],
        "expected_top": "TOA"
    },
    "palmitic_acid": {
        "description": "Palmitic acid (PA) - saturated C16 fatty acid",
        "peaks": [702, 1064, 1130, 1298, 1440, 2850, 2880, 2930],
        "expected_top": "PA"
    },
    "linoleic_acid": {
        "description": "Linoleic acid (LA) - di-unsaturated C18 fatty acid",
        "peaks": [702, 1005, 1064, 1130, 1265, 1303, 1440, 1654, 2850, 2880, 2930, 3009],
        "expected_top": "LA"
    },
    "phosphatidylcholine": {
        "description": "PC (phosphatidylcholine) - zwitterionic phospholipid",
        "peaks": [702, 719, 876, 1005, 1064, 1090, 1130, 1265, 1303, 1440, 1657, 1740, 2850, 2880, 2930],
        "expected_top": "PC"
    },
    "mixed_unsaturated": {
        "description": "Mixed unsaturated lipid (simulated real sample)",
        "peaks": [702, 1005, 1064, 1130, 1265, 1303, 1440, 1655, 1740, 2850, 2880, 2930, 3009],
        "expected_top": "COA"
    }
}


def load_peaks_from_file(path: str) -> list[float]:
    """Load peak list from a file (one peak per line, or comma-separated)"""
    with open(path) as f:
        content = f.read().strip()
    if "\n" in content:
        return [float(x.strip()) for x in content.split("\n") if x.strip()]
    return [float(x.strip()) for x in content.split(",") if x.strip()]


def run_peak_matching(db: sqlite3.Connection, peaks: list[float], tol: float = 5.0):
    """Run peak-set matching against the database"""
    results = match_peak_set(db, peaks, tol)
    return results


def format_peak_match_results(results: list[dict], peaks: list[float], tol: float, db) -> str:
    """Format peak matching results for display"""
    if not results:
        return "  No species matched any of the input peaks."

    lines = [f"Input peaks: {peaks} ±{tol} cm⁻¹", f"Matched {len(results)} candidate species:\n"]
    for i, r in enumerate(results[:10], 1):
        ident = lipid_identity(db, r["origin"])
        label = f"{ident['full_name']} ({r['origin']}, {ident['main_class']})" \
            if ident else r["origin"]
        lines.append(f"  {i}. {label}")
        lines.append(f"     {r['n_matched']}/{len(peaks)} peaks match ({r['n_known']} known bands)")
        for peak, hits in r["matched"].items():
            pid, title, doi, wl, wh, asg = hits[0]
            band = f"{wl:.0f}" + (f"-{wh:.0f}" if wh != wl else "")
            lines.append(f"     {peak:.1f} → {band} cm⁻¹  {asg}  [{pid}]")
        lines.append("")
    if len(results) > 10:
        lines.append(f"  ... and {len(results) - 10} more candidates")
    return "\n".join(lines)


def run_full_pipeline(question: str, peaks: list[float], k: int = 5,
                       tol: float = 5.0, demo_name: str = None) -> dict:
    """Run the full Raman integration pipeline:
    1. Peak matching (candidate lipid identification)
    2. Context building (grounded evidence from literature)
    3. Returns structured output for LLM panel
    """
    db = sqlite3.connect(DB_PATH)

    # Step 1: Peak matching
    peak_results = match_peak_set(db, peaks, tol)

    # Step 2: Build grounded context
    ctx = build_context_block(question, peaks, k)

    db.close()

    return {
        "demo_name": demo_name,
        "input_peaks": peaks,
        "tolerance": tol,
        "peak_match_results": peak_results,
        "context_block": ctx
    }


def print_pipeline_output(output: dict):
    """Print the full pipeline output in a readable format"""
    print("=" * 70)
    print("RAMAN INTEGRATION PIPELINE OUTPUT")
    print("=" * 70)

    if output["demo_name"]:
        print(f"\nDemo spectrum: {output['demo_name']}")

    print(f"\nInput peaks: {output['input_peaks']} ±{output['tolerance']} cm⁻¹")

    # Peak matching results
    print("\n" + "-" * 70)
    print("STEP 1: PEAK-SET MATCHING (Candidate Lipid Identification)")
    print("-" * 70)
    print(format_peak_match_results(
        output["peak_match_results"],
        output["input_peaks"],
        output["tolerance"]
    ))

    # Context block
    ctx = output["context_block"]
    print("\n" + "-" * 70)
    print("STEP 2: GROUNDED CONTEXT (for LLM panel)")
    print("-" * 70)
    print(f"Question: {ctx['question']}")
    print(f"Evidence items: {ctx['n_evidence']}")
    print("\n" + ctx["evidence_block"][:3000] + ("..." if len(ctx["evidence_block"]) > 3000 else ""))


def demo_pipeline():
    """Run demo with predefined spectra"""
    print("Running Raman integration pipeline demos...\n")

    db = sqlite3.connect(DB_PATH)

    for name, spec in DEMO_SPECTRA.items():
        print(f"\n{'='*70}")
        print(f"DEMO: {name} - {spec['description']}")
        print(f"{'='*70}")

        question = f"Identify the lipid species from these Raman peaks: {spec['peaks']}"
        output = run_full_pipeline(question, spec["peaks"], k=5, demo_name=name)

        # Show abbreviated results
        peak_results = output["peak_match_results"]
        if peak_results:
            top = peak_results[0]
            ident = lipid_identity(db, top["origin"])
            label = f"{ident['full_name']} ({top['origin']})" if ident else top["origin"]
            print(f"\nTop candidate: {label} ({top['n_matched']}/{len(spec['peaks'])} peaks matched)")
            if spec["expected_top"] in label or top["origin"] == spec["expected_top"]:
                print("  ✓ Matches expected!")
            else:
                print(f"  ✗ Expected: {spec['expected_top']}")

        # Show first few evidence items
        ctx = output["context_block"]
        print(f"Context items for LLM: {ctx['n_evidence']}")
        print(f"First evidence: {ctx['evidence_block'][:200]}...")

    db.close()


def main() -> int:
    ap = argparse.ArgumentParser(description="Raman integration pipeline")
    ap.add_argument("--peaks", type=str, default=None,
                    help="Comma-separated peaks (cm⁻¹), e.g. 2850,2880,1440,1660")
    ap.add_argument("--peaks-file", type=str, default=None,
                    help="Path to file with peaks (one per line or comma-separated)")
    ap.add_argument("--question", type=str,
                    default="What lipid species do these Raman peaks correspond to?",
                    help="Question for the LLM panel")
    ap.add_argument("-k", type=int, default=5, help="Evidence items per track")
    ap.add_argument("--tol", type=float, default=5.0, help="Tolerance in cm⁻¹")
    ap.add_argument("--demo", action="store_true",
                    help="Run all demo spectra")
    ap.add_argument("--json", action="store_true", help="Output JSON")
    args = ap.parse_args()

    if args.demo:
        demo_pipeline()
        return 0

    # Load peaks
    if args.peaks_file:
        peaks = load_peaks_from_file(args.peaks_file)
    elif args.peaks:
        peaks = [float(x.strip()) for x in args.peaks.split(",") if x.strip()]
    else:
        print("Error: provide --peaks or --peaks-file or use --demo")
        return 1

    output = run_full_pipeline(args.question, peaks, args.k, args.tol)

    if args.json:
        # Convert to JSON-serializable
        serializable = {
            "input_peaks": output["input_peaks"],
            "tolerance": output["tolerance"],
            "question": output["context_block"]["question"],
            "peak_matches": [
                {
                    "origin": r["origin"],
                    "n_matched": r["n_matched"],
                    "n_known": r["n_known"],
                    "matched_peaks": {str(k): v for k, v in r["matched"].items()}
                }
                for r in output["peak_match_results"][:10]
            ],
            "context_evidence_count": output["context_block"]["n_evidence"],
            "context_block": output["context_block"]["evidence_block"]
        }
        print(json.dumps(serializable, indent=2))
    else:
        print_pipeline_output(output)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())