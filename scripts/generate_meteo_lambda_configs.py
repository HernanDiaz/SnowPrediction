"""
Genera los 24 configs del barrido de lambda para ResUNet++ meteo1 (26ch).
Parametros HPO del Trial 1: base=48, lr=4.653e-5, adamw, bs=8, dropout=0.12, wd=1.055e-5, gc=1.0
"""
from pathlib import Path
import yaml

ROOT = Path(__file__).resolve().parent.parent
CFG_DIR = ROOT / "configs"

LAMBDAS = [
    ("sp00",  0.0),
    ("sp01",  0.1),
    ("sp025", 0.25),
    ("sp04",  0.4),
    ("sp05",  0.5),
    ("sp06",  0.6),
    ("sp075", 0.75),
    ("sp10",  1.0),
]
SEEDS = [42, 123, 7]

for lam_name, lam_val in LAMBDAS:
    for seed in SEEDS:
        name = f"resunetpp_meteo_{lam_name}_s{seed}"
        cfg = {
            "experiment": {"name": name},
            "data": {
                "root":        "dataset_v4_ms_sx200_meteo",
                "csv_file":    "dataset_v4_ms_sx200_meteo.csv",
                "images_dir":  "images",
                "masks_dir":   "masks",
                "source":      "lidar",
                "split_type":  "temporal",
                "use_sce":     False,
                "augmentation": False,
            },
            "model": {
                "architecture": "resunetpp",
                "in_channels":  26,
                "out_channels": 1,
                "features":     [48, 96, 192, 384],
                "dropout_p":    0.11973169683940732,
                "num_groups":   8,
            },
            "training": {
                "seed":            seed,
                "batch_size":      8,
                "learning_rate":   4.653e-5,
                "epochs":          50,
                "loss":            "spatial_mse",
                "lambda_pearson":  lam_val,
                "weight_decay":    1.055e-5,
                "optimizer":       "adamw",
                "grad_clip":       1.0,
                "early_stopping":  False,
                "num_workers":     0,
                "device":          "auto",
            },
            "output": {
                "models_dir":  "Articulo 1/Models",
                "results_dir": f"results/lambda_sweep_meteo/{name}",
                "model_name":  name,
            },
        }
        out_path = CFG_DIR / f"{name}.yaml"
        with open(out_path, "w") as f:
            yaml.dump(cfg, f, default_flow_style=False, sort_keys=False)
        print(f"  Creado: {out_path.name}")

print(f"\nTotal: {len(LAMBDAS) * len(SEEDS)} configs generados en {CFG_DIR}")
