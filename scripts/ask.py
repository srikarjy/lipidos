"""One-command entry point to the full pipeline: question -> Context Builder
(local) -> fine-tuned model generation (remote GPU) -> printed RAG + LLM
panels. Wraps `build_evidence.py` and `generate_finetuned_answer.py` (kept
as separate scripts for the reasons in `build_evidence.py`'s docstring --
this just drives both from one command instead of two).

Requires the `hf` CLI authenticated (`hf auth login`) and billing enabled
for HF Jobs -- this launches a real, billed GPU job (a10g-large, ~$1.50/hr;
a single question typically finishes in 2-3 minutes, well under $0.10).

Usage:
    .venv/bin/python scripts/ask.py "how is the 1655/1440 ratio used to quantify unsaturation?"
    .venv/bin/python scripts/ask.py "..." --peaks 2850,2880,1440,1660
"""

import argparse
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from build_evidence import EVIDENCE_REPO, main as build_evidence_main  # noqa: E402

FLAVOR = "a10g-large"
TIMEOUT = "15m"


def run(cmd: list[str], **kw) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, capture_output=True, text=True, **kw)


def hf_token() -> str:
    r = run(["hf", "auth", "token"])
    return r.stdout.strip()


def launch_job() -> str:
    """Returns the job id."""
    tok = hf_token()
    r = run([
        "hf", "jobs", "uv", "run",
        str(ROOT / "generate_finetuned_answer.py"),
        "--flavor", FLAVOR, "--timeout", TIMEOUT,
        "--secrets", f"HF_TOKEN={tok}", "--detach",
    ])
    if r.returncode != 0:
        print(r.stdout, file=sys.stderr)
        print(r.stderr, file=sys.stderr)
        raise SystemExit("job launch failed")
    # `hf jobs uv run --detach` prints a line like: id=<job_id> name=...
    for line in r.stdout.splitlines():
        if line.startswith("id="):
            return line.split()[0].removeprefix("id=")
    raise SystemExit(f"could not parse job id from output:\n{r.stdout}")


def job_stage(job_id: str) -> str:
    r = run(["hf", "jobs", "inspect", f"srikarjy025/{job_id}", "--format", "json"])
    import json
    try:
        return json.loads(r.stdout)[0]["status"]["stage"]
    except Exception:
        return "UNKNOWN"


def job_logs(job_id: str) -> str:
    r = run(["hf", "jobs", "logs", f"srikarjy025/{job_id}"])
    return r.stdout


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("question")
    ap.add_argument("--peaks", type=str, default=None)
    ap.add_argument("-k", type=int, default=5)
    ap.add_argument("--poll-secs", type=int, default=15)
    ap.add_argument("--max-wait-secs", type=int, default=900)
    args = ap.parse_args()

    print(f"[1/3] Building evidence locally via Context Builder...")
    build_argv = [args.question, "-k", str(args.k)]
    if args.peaks:
        build_argv += ["--peaks", args.peaks]
    sys.argv = ["build_evidence.py"] + build_argv
    build_evidence_main()
    print(f"      -> uploaded to https://huggingface.co/datasets/{EVIDENCE_REPO}")

    print(f"\n[2/3] Launching generation job on {FLAVOR}...")
    job_id = launch_job()
    print(f"      job: https://huggingface.co/jobs/srikarjy025/{job_id}")

    print(f"\n[3/3] Waiting for completion (polling every {args.poll_secs}s)...")
    waited = 0
    while waited < args.max_wait_secs:
        stage = job_stage(job_id)
        if stage in ("COMPLETED", "ERROR", "CANCELED"):
            break
        time.sleep(args.poll_secs)
        waited += args.poll_secs
    else:
        print(f"      still running after {args.max_wait_secs}s -- check manually: "
              f"hf jobs logs srikarjy025/{job_id}", file=sys.stderr)
        return 1

    logs = job_logs(job_id)
    # Print from the RAG PANEL marker onward -- skip the pip-install/download noise.
    marker = "=== RAG PANEL"
    idx = logs.find(marker)
    print("\n" + (logs[idx:] if idx != -1 else logs))

    return 0 if stage == "COMPLETED" else 1


if __name__ == "__main__":
    raise SystemExit(main())
