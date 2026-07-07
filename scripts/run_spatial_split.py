"""
Run the SPATIAL-split ResUNet++ experiment (3 seeds), fully isolated.
=====================================================================

Trains and evaluates ResUNet++ (lambda=0.4) on the spatial split
(configs/resunetpp_spatial_s*.yaml) for seeds 42, 123 and 7. Every artefact is
written under results/spatial_split/ so nothing from other experiments is
touched.

Safety: by default the runner SKIPS any seed whose weights already exist, so a
re-run never overwrites a previous result. Pass --force to retrain anyway.

Usage (from the repo root, with the project venv):
    .venv/Scripts/python.exe scripts/run_spatial_split.py
    .venv/Scripts/python.exe scripts/run_spatial_split.py --force
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
OUT_ROOT = _REPO / "results" / "spatial_split"
OUT_ROOT.mkdir(parents=True, exist_ok=True)
WEIGHTS = OUT_ROOT / "weights"
LOG = OUT_ROOT / "run_spatial_split_log.txt"
SPATIAL_CSV = _REPO / "dataset_v4_ms_sx200" / "dataset_v4_ms_sx200_spatial.csv"

SEEDS = [42, 123, 7]
CONFIGS = {s: _REPO / f"configs/resunetpp_spatial_s{s}.yaml" for s in SEEDS}


def already_done(seed: int) -> bool:
    return (WEIGHTS / f"resunetpp_spatial_s{seed}.pth").exists()


def run_seed(seed: int, log_fh) -> int:
    cfg = CONFIGS[seed]
    print(f"\n{'='*70}\n[spatial] seed {seed}  ->  {cfg.name}\n{'='*70}", flush=True)
    log_fh.write(f"\n{'='*70}\nseed {seed}  {cfg.name}\n{'='*70}\n"); log_fh.flush()
    proc = subprocess.run(
        [sys.executable, str(_REPO / "main.py"),
         "--config", str(cfg), "--mode", "both"],
        cwd=str(_REPO), stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
    )
    print(proc.stdout, flush=True)
    log_fh.write(proc.stdout); log_fh.flush()
    return proc.returncode


def summarise():
    import numpy as np
    keys = ["R2", "SPAEF", "MSPAEF", "MAE", "RMSE", "Bias"]
    per_seed, rows = {}, {k: [] for k in keys}
    for s in SEEDS:
        p = OUT_ROOT / f"s{s}" / f"resunetpp_spatial_s{s}_metrics.json"
        if not p.exists():
            print(f"  [warn] missing metrics: {p}")
            continue
        d = json.load(open(p))
        per_seed[s] = {k: d.get(k) for k in keys}
        for k in keys:
            if d.get(k) is not None:
                rows[k].append(d[k])
    summary = {"split": "spatial", "model": "resunetpp_lpear0.4",
               "per_seed": per_seed, "mean": {}, "std": {}}
    print("\n========= ResUNet++ spatial split summary [best-val] =========")
    for k in keys:
        if rows[k]:
            a = np.array(rows[k], dtype=float)
            summary["mean"][k] = round(float(a.mean()), 4)
            summary["std"][k] = round(float(a.std(ddof=1)), 4) if len(a) > 1 else 0.0
            print(f"  {k:7s} = {a.mean():+.3f} +/- "
                  f"{a.std(ddof=1) if len(a) > 1 else 0.0:.3f}")
    out = OUT_ROOT / "resunetpp_spatial_summary.json"
    json.dump(summary, open(out, "w"), indent=2)
    print(f"\nSaved: {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true",
                    help="retrain even if weights already exist (overwrites)")
    args = ap.parse_args()

    if not SPATIAL_CSV.exists():
        sys.exit(f"ERROR: spatial CSV not found: {SPATIAL_CSV}\n"
                 f"Run data/make_spatial_split.py first.")

    print(f"Repo: {_REPO}\nOutputs isolated under: {OUT_ROOT}\nLog: {LOG}")
    with open(LOG, "w", encoding="utf-8") as log_fh:
        for s in SEEDS:
            if already_done(s) and not args.force:
                msg = (f"[skip] seed {s}: weights already exist "
                       f"(resunetpp_spatial_s{s}.pth). Use --force to retrain.")
                print(msg, flush=True); log_fh.write(msg + "\n")
                continue
            rc = run_seed(s, log_fh)
            if rc != 0:
                print(f"[error] seed {s} exited with code {rc}; see {LOG}")
                log_fh.write(f"[error] seed {s} rc={rc}\n")
    summarise()


if __name__ == "__main__":
    main()
