"""
Option A - standardise on the OLD normalisation (raw 5 m-topography channels).
==============================================================================

The 22-channel temporal experiments (ResUNet++ comparison, lambda sweep) were
trained with norm_extended=False (5 m-topo channels 17-21 fed RAW) and reproduce
as-is. Only two model sets were trained with the NEW normalisation and must be
retrained with the OLD one to make the whole paper consistent:

  * U-Net (GroupNorm), temporal split, lambda=0   (comparison baseline)
  * ResUNet++ (lambda=0.4), spatial split

Everything is written under results/optA_oldnorm/ with norm_extended=False, so it
never mixes with or overwrites the NEW-norm results (results/unet_gn,
results/spatial_split) that will be used for Option B.

Safety: skips any run whose weights already exist; --force to retrain.
    .venv/Scripts/python.exe scripts/run_optA_oldnorm.py --dry-run
    .venv/Scripts/python.exe scripts/run_optA_oldnorm.py
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path

import yaml

_REPO = Path(__file__).resolve().parent.parent
OUT = _REPO / "results" / "optA_oldnorm"
CFG_DIR = OUT / "configs"
WEIGHTS = OUT / "weights"
LOG = OUT / "run_optA_oldnorm_log.txt"
SEEDS = [42, 123, 7]


def unet_cfg(seed):
    name = f"unet_gn_oldnorm_s{seed}"
    return name, {
        "experiment": {"name": name},
        "data": {"root": "dataset_v4_ms_sx200",
                 "csv_file": "dataset_v4_ms_sx200.csv",
                 "images_dir": "images", "masks_dir": "masks",
                 "source": "lidar", "split_type": "temporal",
                 "use_sce": False, "augmentation": False,
                 "norm_extended": False},
        "model": {"architecture": "unet_gn", "in_channels": 22, "out_channels": 1,
                  "features": [48, 96, 192, 384], "dropout_p": 0.0012, "num_groups": 8},
        "training": {"seed": seed, "batch_size": 16, "learning_rate": 3.064e-05,
                     "epochs": 50, "loss": "mse", "lambda_pearson": 0.0,
                     "weight_decay": 2.149e-05, "optimizer": "adam", "grad_clip": 1.0,
                     "early_stopping": False, "num_workers": 0, "device": "auto"},
        "output": {"models_dir": "results/optA_oldnorm/weights",
                   "results_dir": f"results/optA_oldnorm/unet_gn/s{seed}",
                   "model_name": name},
    }


def res_spatial_cfg(seed):
    name = f"resunetpp_spatial_oldnorm_s{seed}"
    return name, {
        "experiment": {"name": name},
        "data": {"root": "dataset_v4_ms_sx200",
                 "csv_file": "dataset_v4_ms_sx200_spatial.csv",
                 "images_dir": "images", "masks_dir": "masks",
                 "source": "lidar", "split_type": "spatial",
                 "use_sce": False, "augmentation": False,
                 "norm_extended": False},
        "model": {"architecture": "resunetpp", "in_channels": 22, "out_channels": 1,
                  "features": [64, 128, 256, 512], "dropout_p": 0.077, "num_groups": 8},
        "training": {"seed": seed, "batch_size": 8, "learning_rate": 1.287e-04,
                     "epochs": 50, "loss": "spatial_mse", "lambda_pearson": 0.4,
                     "weight_decay": 1.239e-04, "optimizer": "adamw", "grad_clip": 1.0,
                     "early_stopping": False, "num_workers": 0, "device": "auto"},
        "output": {"models_dir": "results/optA_oldnorm/weights",
                   "results_dir": f"results/optA_oldnorm/spatial/s{seed}",
                   "model_name": name},
    }


def all_jobs():
    jobs = []
    for s in SEEDS:
        jobs.append(unet_cfg(s))
    for s in SEEDS:
        jobs.append(res_spatial_cfg(s))
    return jobs


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
    print(f"  [ok] {name}: norm_extended=False split={d['split_type']} "
          f"loss={cfg['training']['loss']} lambda={cfg['training']['lambda_pearson']}")


def run(p, log_fh):
    print(f"\n{'='*70}\n[optA] {p.stem}\n{'='*70}", flush=True)
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
    groups = {"U-Net GN (temporal, old norm)":
              [OUT / f"unet_gn/s{s}/unet_gn_oldnorm_s{s}_metrics.json" for s in SEEDS],
              "ResUNet++ (spatial, old norm)":
              [OUT / f"spatial/s{s}/resunetpp_spatial_oldnorm_s{s}_metrics.json" for s in SEEDS]}
    out = {}
    print("\n============ Option A (OLD norm) summary [best-val] ============")
    for label, paths in groups.items():
        rows = {k: [] for k in keys}
        for p in paths:
            if p.exists():
                d = json.load(open(p))
                for k in keys:
                    if d.get(k) is not None:
                        rows[k].append(d[k])
        e = {}
        for k in keys:
            if rows[k]:
                a = np.array(rows[k], float)
                e[k] = [round(float(a.mean()), 4),
                        round(float(a.std(ddof=1)), 4) if len(a) > 1 else 0.0]
        out[label] = e
        if "R2" in e:
            print(f"  {label:34s} R2={e['R2'][0]:+.3f}±{e['R2'][1]:.3f}  "
                  f"SPAEF={e['SPAEF'][0]:+.3f}±{e['SPAEF'][1]:.3f}")
    json.dump(out, open(OUT / "optA_oldnorm_summary.json", "w"), indent=2)
    print(f"\nSaved: {OUT / 'optA_oldnorm_summary.json'}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    jobs = all_jobs()
    print(f"Option A (OLD norm). Outputs isolated under: {OUT}")
    print(f"{len(jobs)} runs: U-Net GN x{len(SEEDS)} + ResUNet++ spatial x{len(SEEDS)}")

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
    summarise()


if __name__ == "__main__":
    main()
