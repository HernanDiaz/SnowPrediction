"""
Completa el punto lambda=0 del barrido OLD-norm (ResUNet++).
=============================================================
sp00_s123 y sp00_s7 se entrenaron originalmente con norm NUEVA, asi que el punto
lambda=0 del barrido old-norm solo tenia 1 seed (s42). Aqui se reentrenan esos
dos seeds con norm VIEJA (norm_extended=False), mismos hiperparametros que el
barrido (Trial 17), loss spatial_mse lambda=0.

Aislado bajo results/paper_oldnorm/ (pesos en weights/, run en lambda_runs/), sin
tocar nada existente. Al terminar copia metrics.json a lambda/sp00_s{seed}/ para
completar el arbol del barrido. Guardia anti-sobrescritura (--force para rehacer).

    .venv/Scripts/python.exe scripts/run_oldnorm_sp00_missing.py --dry-run
    .venv/Scripts/python.exe scripts/run_oldnorm_sp00_missing.py
"""

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

import yaml

_REPO = Path(__file__).resolve().parent.parent
OUT = _REPO / "results" / "paper_oldnorm"
CFG_DIR = OUT / "lambda_runs" / "configs"
WEIGHTS = OUT / "weights"
LOG = OUT / "run_oldnorm_sp00_missing_log.txt"
SEEDS = [123, 7]


def cfg_for(seed):
    name = f"resunetpp_oldnorm_sp00_s{seed}"
    return name, {
        "experiment": {"name": name},
        "data": {"root": "dataset_v4_ms_sx200",
                 "csv_file": "dataset_v4_ms_sx200.csv",
                 "images_dir": "images", "masks_dir": "masks",
                 "source": "lidar", "split_type": "temporal",
                 "use_sce": False, "augmentation": False,
                 "norm_extended": False},
        "model": {"architecture": "resunetpp", "in_channels": 22, "out_channels": 1,
                  "features": [64, 128, 256, 512], "dropout_p": 0.077, "num_groups": 8},
        "training": {"seed": seed, "batch_size": 8, "learning_rate": 1.287e-04,
                     "epochs": 50, "loss": "spatial_mse", "lambda_pearson": 0.0,
                     "weight_decay": 1.239e-04, "optimizer": "adamw", "grad_clip": 1.0,
                     "early_stopping": False, "num_workers": 0, "device": "auto"},
        "output": {"models_dir": "results/paper_oldnorm/weights",
                   "results_dir": f"results/paper_oldnorm/lambda_runs/sp00_s{seed}",
                   "model_name": name},
    }


def write_cfg(name, cfg):
    CFG_DIR.mkdir(parents=True, exist_ok=True)
    p = CFG_DIR / f"{name}.yaml"
    yaml.safe_dump(cfg, open(p, "w"), sort_keys=False)
    return p


def validate(name, cfg):
    sys.path.insert(0, str(_REPO))
    from models.unet import build_model
    from data.dataset import load_splits
    import os
    d = cfg["data"]
    load_splits(os.path.join(d["root"], d["csv_file"]),
                source=d["source"], split_type=d["split_type"])
    build_model(cfg)
    assert cfg["data"]["norm_extended"] is False
    print(f"  [ok] {name}: norm_extended=False loss={cfg['training']['loss']} "
          f"lambda={cfg['training']['lambda_pearson']} seed={cfg['training']['seed']}")


def run(p, log_fh):
    print(f"\n{'='*70}\n[oldnorm-sp00] {p.stem}\n{'='*70}", flush=True)
    log_fh.write(f"\n{'='*70}\n{p.stem}\n{'='*70}\n"); log_fh.flush()
    proc = subprocess.run([sys.executable, str(_REPO / "main.py"),
                           "--config", str(p), "--mode", "both"],
                          cwd=str(_REPO), stdout=subprocess.PIPE,
                          stderr=subprocess.STDOUT, text=True)
    print(proc.stdout, flush=True); log_fh.write(proc.stdout); log_fh.flush()
    return proc.returncode


def consolidate():
    """Copia metrics al arbol del barrido y recalcula la media lambda=0 (n=3)."""
    import numpy as np
    for s in SEEDS:
        src = OUT / "lambda_runs" / f"sp00_s{s}" / f"resunetpp_oldnorm_sp00_s{s}_metrics.json"
        if src.exists():
            dst = OUT / "lambda" / f"sp00_s{s}"
            dst.mkdir(parents=True, exist_ok=True)
            shutil.copy(src, dst / "metrics.json")
    r2, sp = [], []
    for s in (42, 123, 7):
        p = OUT / "lambda" / f"sp00_s{s}" / "metrics.json"
        if p.exists():
            d = json.load(open(p)); r2.append(d["R2"]); sp.append(d.get("SPAEF"))
    if len(r2) == 3:
        print(f"\nlambda=0 OLD-norm (n=3): R2={np.mean(r2):+.3f}±{np.std(r2, ddof=1):.3f} "
              f"SPAEF={np.nanmean(sp):+.3f}")
        print("per-seed R2:", {s: round(v, 4) for s, v in zip((42, 123, 7), r2)})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    jobs = [cfg_for(s) for s in SEEDS]
    print(f"OLD-norm lambda=0 completion. Isolated under: {OUT}")
    print(f"{len(jobs)} runs: ResUNet++ sp00 (lambda=0) seeds {SEEDS}")

    if args.dry_run:
        print("\n[dry-run] validating configs (no training):")
        for name, cfg in jobs:
            validate(name, cfg)
        print("\nDry-run OK. Remove --dry-run to train.")
        return

    with open(LOG, "w", encoding="utf-8") as log_fh:
        for name, cfg in jobs:
            if (WEIGHTS / f"{name}.pth").exists() and not args.force:
                msg = f"[skip] {name}: weights exist (use --force)."
                print(msg, flush=True); log_fh.write(msg + "\n"); continue
            p = write_cfg(name, cfg)
            rc = run(p, log_fh)
            if rc != 0:
                print(f"[error] {name} rc={rc}; see {LOG}")
                log_fh.write(f"[error] {name} rc={rc}\n")
    consolidate()


if __name__ == "__main__":
    main()
