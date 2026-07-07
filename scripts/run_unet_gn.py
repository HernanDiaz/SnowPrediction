"""
Run the GroupNorm U-Net experiment (3 seeds) end to end.
=========================================================

Trains and evaluates the GroupNorm U-Net (architecture 'unet_gn') for seeds
42, 123 and 7, using the isolated configs in configs/unet_gn_s*.yaml. Every
artefact (weights, metrics, curves, logs) is written under results/unet_gn/
so nothing mixes with other experiments.

Usage (from the repo root, with the project venv):
    .venv/Scripts/python.exe scripts/run_unet_gn.py

After training it prints a summary (best-validation checkpoint, mean +/- std
over the three seeds) and writes results/unet_gn/unet_gn_summary.json.
"""

import json
import subprocess
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
OUT_ROOT = _REPO / "results" / "unet_gn"
OUT_ROOT.mkdir(parents=True, exist_ok=True)
LOG = OUT_ROOT / "run_unet_gn_log.txt"

SEEDS = [42, 123, 7]
CONFIGS = {s: _REPO / f"configs/unet_gn_s{s}.yaml" for s in SEEDS}


def run_seed(seed: int, log_fh) -> int:
    cfg = CONFIGS[seed]
    print(f"\n{'='*70}\n[unet_gn] seed {seed}  ->  {cfg.name}\n{'='*70}", flush=True)
    log_fh.write(f"\n{'='*70}\nseed {seed}  {cfg.name}\n{'='*70}\n")
    log_fh.flush()
    # mode both = train then evaluate (best-val + last checkpoints)
    proc = subprocess.run(
        [sys.executable, str(_REPO / "main.py"),
         "--config", str(cfg), "--mode", "both"],
        cwd=str(_REPO), stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True,
    )
    print(proc.stdout, flush=True)
    log_fh.write(proc.stdout)
    log_fh.flush()
    return proc.returncode


def summarise():
    """Aggregate best-validation metrics across the three seeds."""
    import numpy as np
    keys = ["R2", "SPAEF", "MSPAEF", "MAE", "RMSE", "Bias"]
    per_seed, rows = {}, {k: [] for k in keys}
    for s in SEEDS:
        p = OUT_ROOT / f"s{s}" / f"unet_gn_s{s}_metrics.json"
        if not p.exists():
            print(f"  [warn] missing metrics: {p}")
            continue
        d = json.load(open(p))
        per_seed[s] = {k: d.get(k) for k in keys}
        for k in keys:
            if d.get(k) is not None:
                rows[k].append(d[k])
    summary = {"per_seed": per_seed, "mean": {}, "std": {}}
    print("\n================ U-Net (GroupNorm) summary [best-val] ============")
    for k in keys:
        if rows[k]:
            a = np.array(rows[k], dtype=float)
            summary["mean"][k] = round(float(a.mean()), 4)
            summary["std"][k] = round(float(a.std(ddof=1)), 4) if len(a) > 1 else 0.0
            print(f"  {k:7s} = {a.mean():+.3f} +/- "
                  f"{a.std(ddof=1) if len(a) > 1 else 0.0:.3f}")
    out = OUT_ROOT / "unet_gn_summary.json"
    json.dump(summary, open(out, "w"), indent=2)
    print(f"\nSaved: {out}")


def main():
    print(f"Repo: {_REPO}\nOutputs isolated under: {OUT_ROOT}\nLog: {LOG}")
    with open(LOG, "w", encoding="utf-8") as log_fh:
        for s in SEEDS:
            rc = run_seed(s, log_fh)
            if rc != 0:
                print(f"[error] seed {s} exited with code {rc}; see {LOG}")
                log_fh.write(f"[error] seed {s} rc={rc}\n")
    summarise()


if __name__ == "__main__":
    main()
