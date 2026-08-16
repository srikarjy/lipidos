"""Local half of the integrated pipeline: question (+ optional peaks) -> Context
Builder's merged, deduped, cited evidence block -> uploaded for remote generation.

Why split into two scripts (local build, remote generate): Context Builder
needs `papers.db`/`chunk_vectors.npy` (local, ~40 MB) and only BGE (fast,
runs fine on this machine's MPS). Generation needs the fine-tuned model
(3.8B params, ~7.6 GB in bf16) -- too tight for this 8 GB M2 machine, so it
runs on a remote GPU instead (see `generate_finetuned_answer.py`). This
mirrors the already-proven pattern from `verify_finetuned_citations.py`
(2026-08-15): build evidence locally, upload as a small JSON, generate
remotely.

Usage:
    .venv/bin/python scripts/build_evidence.py "how is the 1655/1440 ratio used?"
    .venv/bin/python scripts/build_evidence.py "..." --peaks 2850,2880,1440,1660
"""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

from context_builder import build_context_block  # noqa: E402

EVIDENCE_REPO = "srikarjy025/lipidos-pipeline-evidence"
OUT_PATH = ROOT / "data" / "pipeline_evidence.json"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("question")
    ap.add_argument("--peaks", type=str, default=None,
                     help="comma-separated observed peaks, e.g. 2850,2880,1440,1660")
    ap.add_argument("-k", type=int, default=5)
    ap.add_argument("--no-upload", action="store_true",
                     help="build and save locally only, skip the Hub upload")
    args = ap.parse_args()

    peaks = [float(x) for x in args.peaks.split(",")] if args.peaks else None
    ctx = build_context_block(args.question, peaks, args.k)

    print(f"Built {ctx['n_evidence']} evidence items for: {ctx['question']!r}")
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(ctx, indent=2))
    print(f"Saved -> {OUT_PATH}")

    if not args.no_upload:
        from huggingface_hub import HfApi
        api = HfApi()
        api.upload_file(
            repo_id=EVIDENCE_REPO, repo_type="dataset",
            path_or_fileobj=str(OUT_PATH), path_in_repo="pipeline_evidence.json",
            commit_message=f"evidence for: {args.question[:60]}",
        )
        print(f"Uploaded -> https://huggingface.co/datasets/{EVIDENCE_REPO}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
