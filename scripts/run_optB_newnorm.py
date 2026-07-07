"""
Option B - standardise on the NEW normalisation (5 m-topography channels
normalised, norm_extended=True).
========================================================================

Re-runs the ResUNet++ spatial-loss sweep on the temporal split with the NEW
normalisation, so the reference model (lambda=0.4) and the whole lambda curve are
available under the new normalisation for a like-for-like comparison with the
existing OLD-norm results.

The NEW-norm U-Net (GroupNorm) and the NEW-norm spatial ResUNet++ already exist
(results/unet_gn, results/spatial_split) and complete the Option-B picture; the
RF is normalisation-invariant. Ablation / meteo can be added later.

Everything is written under results/optB_newnorm/ with norm_extended=True, so it
never mixes with or overwrites Option A (results/optA_oldnorm) or the existing
OLD-norm sweep.

Safety: skips any (lambda, seed) whose weights already exist; --force to retrain.
    .venv/Scripts/python.exe scripts/run_optB_newnorm.py --dry-run
    .venv/Scripts/python.exe scripts/run_optB_newnorm.py
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path

import yaml

_REPO = Path(__file__).resolve().parent.parent
OUT = _REPO / "results" / "optB_newnorm"
CFG_DIR = OUT / "configs"
WEIGHTS = OUT / "weights"
LOG = OUT / "run_optB_newnorm_log.txt"

# Edit to run fewer lambdas. lambda=0.4 is the reference model.
LAMBDAS = [0.0, 0.1, 0.25, 0.4, 0.5, 0.6, 0.75, 1.0]
SEEDS = [42, 123, 7]
TAG = {0.0: "sp00", 0.1: "sp01", 0.25: "sp025", 0.4: "sp04",
       0.5: "sp05", 0.6: "sp06", 0.75: "sp075", 1.0: "sp10"}


def make_config(lam, seed):
    tag = TAG[lam]
    name = f"resunetpp_newnorm_{tag}_s{seed}"
    cfg = {
        "experiment": {"name": name},
        "data": {"root": "dataset_v4_ms_sx200",
                 "csv_file": "dataset_v4_ms_sx200.csv",
                 "images_dir": "images", "masks_dir": "masks",
                 "source": "lidar", "split_type": "temporal",
                 "use_sce": False, "augmentation": False,
                 "norm_extended": True},          # <-- NEW normalisation
        "model": {"architecture": "resunetpp", "in_channels": 22, "out_channels": 1,
                  "features": [64, 128, 256, 512], "dropout_p": 0.077, "num_groups": 8},
        "training": {"seed": seed, "batch_size": 8, "learning_rate": 1.287e-04,
                     "epochs": 50, "loss": "spatial_mse", "lambda_pearson": lam,
                     "weight_decay": 1.239e-04, "optimizer": "adamw", "grad_clip": 1.0,
                     "early_stopping": False, "num_workers": 0, "device": "auto"},
        "output": {"models_dir": "results/optB_newnorm/weights",
                   "results_dir": f"results/optB_newnorm/lambda/{tag}_s{seed}",
                   "model_name": name},
    }
    CFG_DIR.mkdir(parents=True, exist_ok=True)
    p = CFG_DIR / f"{name}.yaml"
    yaml.safe_dump(cfg, open(p, "w"), sort_keys=False)
    return p


def already_done(lam, seed):
    return (WEIGHTS / f"resunetpp_newnorm_{TAG[lam]}_s{seed}.pth").exists()


def validate(p):
    sys.path.insert(0, str(_REPO))
    from models.unet import build_model
    from data.dataset import load_splits
    import os
    cfg = yaml.safe_load(open(p))
    d = cfg["data"]
    load_splits(os.path.join(d["root"], d["csv_file"]),
                source=d["source"], split_type=d["split_type"])
    build_model(cfg)
    assert cfg["data"]["norm_extended"] is True
    print(f"  [ok] {p.stem}: norm_extended=True lambda={cfg['training']['lambda_pearson']}")


def run(p, log_fh):
    print(f"\n{'='*70}\n[optB] {p.stem}\n{'='*70}", flush=True)
    log_fh.write(f"\n{'='*70}\n{p.stem}\n{'='*70}\n"); log_fh.flush()
    proc = subprocess.run([sys.executable, str(_REPO / "main.py"),
                           "--config", str(p), "--mode", "both"],
                          cwd=str(_REPO), stdout=subprocess.PIPE,
                          stderr=subprocess.STDOUT, text=True)
    print(proc.stdout, flush=True); log_fh.write(proc.stdout); log_fh.flush()
    return proc.returncode


def summarise():
    import numpy as np
    keys = ["R2", "SPAEF", "MSPAEF", "MAE", "RMSE", "Bias"]
    summary = {"model": "resunetpp", "split": "temporal", "norm": "new",
               "by_lambda": {}}
    print("\n====== Option B (NEW norm) ResUNet++ lambda sweep [best-val] ======")
    print(f"{'lambda':>7} | {'R2':>16} | {'SPAEF':>16}")
    for lam in LAMBDAS:
        tag = TAG[lam]
        rows = {k: [] for k in keys}
        for s in SEEDS:
            p = OUT / f"lambda/{tag}_s{s}" / f"resunetpp_newnorm_{tag}_s{s}_metrics.json"
            if not p.exists():
                continue
            d = json.load(open(p))
            for k in keys:
                if d.get(k) is not None:
                    rows[k].append(d[k])
        e = {}
        for k in keys:
            if rows[k]:
                a = np.array(rows[k], float)
                e[k] = {"mean": round(float(a.mean()), 4),
                        "std": round(float(a.std(ddof=1)), 4) if len(a) > 1 else 0.0,
                        "n": len(a)}
        summary["by_lambda"][lam] = e
        if "R2" in e and "SPAEF" in e:
            print(f"{lam:>7} | {e['R2']['mean']:+.3f} +/- {e['R2']['std']:.3f} "
                  f"| {e['SPAEF']['mean']:+.3f} +/- {e['SPAEF']['std']:.3f}")
    OUT.mkdir(parents=True, exist_ok=True)
    json.dump(summary, open(OUT / "optB_newnorm_summary.json", "w"), indent=2)
    print(f"\nSaved: {OUT / 'optB_newnorm_summary.json'}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    n = len(LAMBDAS) * len(SEEDS)
    print(f"Option B (NEW norm). Outputs isolated under: {OUT}")
    print(f"ResUNet++ lambda sweep: {LAMBDAS} x seeds {SEEDS} = {n} runs "
          f"(lambda=0.4 is the reference model)")

    if args.dry_run:
        print("\n[dry-run] validating configs (no training):")
        for lam in LAMBDAS:
            validate(make_config(lam, SEEDS[0]))
        print("\nDry-run OK. Remove --dry-run to train.")
        return

    with open(LOG, "w", encoding="utf-8") as log_fh:
        for lam in LAMBDAS:
            for s in SEEDS:
                if already_done(lam, s) and not args.force:
                    msg = f"[skip] {TAG[lam]}_s{s}: weights exist (use --force)."
                    print(msg, flush=True); log_fh.write(msg + "\n"); continue
                p = make_config(lam, s)
                rc = run(p, log_fh)
                if rc != 0:
                    print(f"[error] {p.stem} rc={rc}; see {LOG}")
                    log_fh.write(f"[error] {p.stem} rc={rc}\n")
    summarise()


if __name__ == "__main__":
    main()
