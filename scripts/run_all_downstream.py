"""Run every stage that depends on a freshly trained model, in dependency order.

Why this exists
---------------
Changing the feature set invalidates SEVEN downstream artefacts. Running them
by hand invites forgetting one, and a stale artefact does not announce itself:
the API would happily serve predictions built from an old feature schema, and
the development log would quote numbers from a model that no longer exists.

Order matters and is not obvious:

    export_feature_schema  ->  needs the new feature list; everything serving
                               related depends on it
    run_eval               ->  needs the model; writes the threshold that the
                               API and the calibrator both read
    run_calibrate          ->  needs the threshold from run_eval
    run_genai              ->  needs the model for SHAP, and rebuilds the
                               narrative corpus and both FAISS indexes
    run_llm                ->  needs the indexes from run_genai
    build_notebooks        ->  needs every result JSON above
    update_devlog          ->  reads all of them and regenerates the log

Usage
-----
    python scripts/run_all_downstream.py
    python scripts/run_all_downstream.py --skip-llm      # no Ollama calls
    python scripts/run_all_downstream.py --skip-notebooks
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PY = str(ROOT / ".venv" / "Scripts" / "python.exe")
if not Path(PY).is_file():
    PY = sys.executable


def run(label: str, args: list[str], log_name: str) -> tuple[bool, float]:
    """Run one stage, tee its output to reports/, return (ok, seconds)."""
    print(f"\n{'=' * 74}\n  {label}\n{'=' * 74}", flush=True)
    log_path = ROOT / "reports" / log_name
    t0 = time.perf_counter()

    proc = subprocess.run(
        [PY, "-u", *args],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        env={**__import__("os").environ, "PYTHONUNBUFFERED": "1"},
    )
    elapsed = time.perf_counter() - t0
    log_path.write_text(proc.stdout + "\n" + proc.stderr, encoding="utf-8")

    ok = proc.returncode == 0
    if ok:
        print(f"  OK in {elapsed:.0f}s   (log: reports/{log_name})")
    else:
        print(f"  FAILED after {elapsed:.0f}s   (log: reports/{log_name})")
        print("  --- last 15 lines of stderr ---")
        for line in (proc.stderr or proc.stdout).splitlines()[-15:]:
            print("   ", line)
    return ok, elapsed


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--skip-llm", action="store_true")
    ap.add_argument("--skip-notebooks", action="store_true")
    args = ap.parse_args()

    stages: list[tuple[str, list[str], str]] = [
        ("1/7  SERVING SCHEMA - capture the new training dtypes",
         ["scripts/export_feature_schema.py"], "chain_1_schema.txt"),
        ("2/7  EVALUATION - thresholds on validation, then test once",
         ["scripts/run_eval.py"], "chain_2_eval.txt"),
        ("3/7  CALIBRATION - honest probabilities for the risk engine",
         ["scripts/run_calibrate.py"], "chain_3_calibrate.txt"),
        ("4/7  SHAP + UNSUPERVISED + NARRATIVES + INDEXES",
         ["scripts/run_genai.py", "--skip-llm", "--n-cases", "800"],
         "chain_4_genai.txt"),
    ]
    if not args.skip_llm:
        stages.append(
            ("5/7  RAG + INVESTIGATION COPILOT",
             ["scripts/run_llm.py"], "chain_5_llm.txt"))
    if not args.skip_notebooks:
        stages += [
            ("6/7  NOTEBOOK 01 - ingestion, EDA, split",
             ["scripts/build_notebooks.py"], "chain_6_nb1.txt"),
            ("6/7  NOTEBOOK 02 - modelling, evaluation, risk engine",
             ["scripts/build_notebook_02.py"], "chain_6_nb2.txt"),
        ]
    stages.append(
        ("7/7  DEVELOPMENT LOG - regenerate from the new artefacts",
         ["scripts/update_devlog.py"], "chain_7_devlog.txt"))

    results: list[tuple[str, bool, float]] = []
    t_start = time.perf_counter()

    for label, argv, log_name in stages:
        ok, secs = run(label, argv, log_name)
        results.append((label, ok, secs))
        # The chain is a dependency chain: a failed schema export makes every
        # later stage meaningless, so stop rather than produce stale artefacts.
        if not ok:
            print("\n  STOPPING - later stages depend on this one, and running "
                  "them now would write artefacts that disagree with each other.")
            break

    print(f"\n{'=' * 74}\n  SUMMARY\n{'=' * 74}")
    for label, ok, secs in results:
        print(f"  {'OK  ' if ok else 'FAIL'}  {secs:>6.0f}s   {label}")
    total = time.perf_counter() - t_start
    n_ok = sum(1 for _, ok, _ in results if ok)
    print(f"\n  {n_ok}/{len(stages)} stages completed in {total / 60:.1f} min")
    return 0 if n_ok == len(stages) else 1


if __name__ == "__main__":
    raise SystemExit(main())
