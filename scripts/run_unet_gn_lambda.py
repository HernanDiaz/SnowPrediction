"""
Spatial-loss sweep for the GroupNorm U-Net (generality of the hybrid loss).
===========================================================================

Trains the GroupNorm U-Net with the hybrid spatial loss
    L = (1 - lambda) * MSE + lambda * (1 - rho_spatial)
for several values of lambda and 3 seeds, on the temporal split. This tests
whether the spatial loss -- shown to help ResUNet++ -- also helps a plain U-Net.

Same U-Net hyperparameters as results/unet_gn (Optuna trial 11, GroupNorm).
Configs are generated on the fly into the isolated output tree; nothing else is
touched.

Outputs (all under results/unet_gn_lambda/):
  weights/unet_gn_<tag>_s<seed>.pth (+ _last)
  <tag>_s<seed>/...metrics.json, curves
  configs/unet_gn_<tag>_s<seed>.yaml      (generated)
  run_unet_gn_lambda_log.txt
  unet_gn_lambda_summary.json

Safety: skips any (lambda, seed) whose weights already exist; --force overrides.

Usage:
    .venv/Scripts/python.exe scripts/run_unet_gn_lambda.py --dry-run   # validate only
    .venv/Scripts/python.exe scripts/run_unet_gn_lambda.py
    .venv/Scripts/python.exe scripts/run_unet_gn_lambda.py --force
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path

import yaml

_REPO = Path(__file__).resolve().parent.parent
OUT_ROOT = _REPO / "results" / "unet_gn_lambda"
CFG_DIR = OUT_ROOT / "configs"
WEIGHTS = OUT_ROOT / "weights"
LOG = OUT_ROOT / "run_unet_gn_lambda_log.txt"

# Edit this list to run fewer lambdas (each lambda x 3 seeds = 3 runs).
# Matches the ResUNet++ sweep for a direct comparison.
LAMBDAS = [0.0, 0.1, 0.25, 0.4, 0.5, 0.6, 0.75, 1.0]
SEEDS = [42, 123, 7]

TAG = {0.0: "sp00", 0.1: "sp01", 0.25: "sp025", 0.4: "sp04",
       0.5: "sp05", 0.6: "sp06", 0.75: "sp075", 1.0: "sp10"}


def make_config(lam: float, seed: int) -> Path:
    tag = TAG[lam]
    name = f"unet_gn_{tag}_s{seed}"
    cfg = {
        "experiment": {"name": name},
        "data": {
            "root": "dataset_v4_ms_sx200",
            "csv_file": "dataset_v4_ms_sx200.csv",
            "images_dir": "images", "masks_dir": "masks",
            "source": "lidar", "split_type": "temporal",
            "use_sce": False, "augmentation": False,
        },
        "model": {
            "architecture": "unet_gn", "in_channels": 22, "out_channels": 1,
            "features": [48, 96, 192, 384], "dropout_p": 0.0012, "num_groups": 8,
        },
        "training": {
            "seed": seed, "batch_size": 16, "learning_rate": 3.064e-05,
            "epochs": 50, "loss": "spatial_mse", "lambda_pearson": lam,
            "weight_decay": 2.149e-05, "optimizer": "adam", "grad_clip": 1.0,
            "early_stopping": False, "num_workers": 0, "device": "auto",
        },
        "output": {
            "models_dir": "results/unet_gn_lambda/weights",
            "results_dir": f"results/unet_gn_lambda/{tag}_s{seed}",
            "model_name": name,
        },
    }
    CFG_DIR.mkdir(parents=True, exist_ok=True)
    p = CFG_DIR / f"{name}.yaml"
    yaml.safe_dump(cfg, open(p, "w"), sort_keys=False)
    return p


def already_done(lam: float, seed: int) -> bool:
    return (WEIGHTS / f"unet_gn_{TAG[lam]}_s{seed}.pth").exists()


def validate(cfg_path: Path):
    """Build the model and load the splits to catch errors before training."""
    sys.path.insert(0, str(_REPO))
    from models.unet import build_model
    from data.dataset import load_splits
    import os
    cfg = yaml.safe_load(open(cfg_path))
    d = cfg["data"]
    load_splits(os.path.join(d["root"], d["csv_file"]),
                source=d["source"], split_type=d["split_type"])
    m = build_model(cfg)
    assert cfg["training"]["loss"] == "spatial_mse"
    n = sum(p.numel() for p in m.parameters())
    print(f"  [ok] {cfg_path.name}: loss=spatial_mse "
          f"lambda={cfg['training']['lambda_pearson']} params={n:,}")


def run(cfg_path: Path, log_fh) -> int:
    print(f"\n{'='*70}\n[unet_gn_lambda] {cfg_path.stem}\n{'='*70}", flush=True)
    log_fh.write(f"\n{'='*70}\n{cfg_path.stem}\n{'='*70}\n"); log_fh.flush()
    proc = subprocess.run(
        [sys.executable, str(_REPO / "main.py"),
         "--config", str(cfg_path), "--mode", "both"],
        cwd=str(_REPO), stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
    )
    print(proc.stdout, flush=True); log_fh.write(proc.stdout); log_fh.flush()
    return proc.returncode


def summarise():
    import numpy as np
    keys = ["R2", "SPAEF", "MSPAEF", "MAE", "RMSE", "Bias"]
    summary = {"model": "unet_gn", "split": "temporal", "by_lambda": {}}
    print("\n========= U-Net (GroupNorm) spatial-loss sweep [best-val] =========")
    print(f"{'lambda':>7} | {'R2':>16} | {'SPAEF':>16}")
    for lam in LAMBDAS:
        tag = TAG[lam]
        rows = {k: [] for k in keys}
        for s in SEEDS:
            p = OUT_ROOT / f"{tag}_s{s}" / f"unet_gn_{tag}_s{s}_metrics.json"
            if not p.exists():
                continue
            d = json.load(open(p))
            for k in keys:
                if d.get(k) is not None:
                    rows[k].append(d[k])
        entry = {}
        for k in keys:
            if rows[k]:
                a = np.array(rows[k], dtype=float)
                entry[k] = {"mean": round(float(a.mean()), 4),
                            "std": round(float(a.std(ddof=1)), 4) if len(a) > 1 else 0.0,
                            "n": len(a)}
        summary["by_lambda"][lam] = entry
        if "R2" in entry and "SPAEF" in entry:
            print(f"{lam:>7} | {entry['R2']['mean']:+.3f} +/- {entry['R2']['std']:.3f} "
                  f"| {entry['SPAEF']['mean']:+.3f} +/- {entry['SPAEF']['std']:.3f}")
    out = OUT_ROOT / "unet_gn_lambda_summary.json"
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    json.dump(summary, open(out, "w"), indent=2)
    print(f"\nSaved: {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true",
                    help="generate and validate configs without training")
    ap.add_argument("--force", action="store_true",
                    help="retrain even if weights already exist")
    args = ap.parse_args()

    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    n_runs = len(LAMBDAS) * len(SEEDS)
    print(f"Outputs isolated under: {OUT_ROOT}")
    print(f"Lambdas: {LAMBDAS}  x seeds {SEEDS}  = {n_runs} runs")

    if args.dry_run:
        print("\n[dry-run] generating and validating configs (no training):")
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
                cfg = make_config(lam, s)
                rc = run(cfg, log_fh)
                if rc != 0:
                    print(f"[error] {cfg.stem} rc={rc}; see {LOG}")
                    log_fh.write(f"[error] {cfg.stem} rc={rc}\n")
    summarise()


if __name__ == "__main__":
    main()
