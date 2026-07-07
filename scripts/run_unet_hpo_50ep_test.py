"""
Test de reproducibilidad: UNet HPO con 50 epocas.
==================================================
Comprueba si replicamos el resultado del Optuna trial #11:
  R2=0.351 | MAE=0.436 | RMSE=0.587

Uso:
    .venv/Scripts/python.exe scripts/run_unet_hpo_50ep_test.py
"""

import subprocess
import sys
import json
from pathlib import Path

ROOT   = Path(__file__).resolve().parent.parent
PYTHON = ROOT / ".venv/Scripts/python.exe"
MAIN   = ROOT / "main.py"
CONFIG = ROOT / "configs/unet_v4_ms_sx200_hpo.yaml"

# Resultados esperados del Optuna trial #11
OPTUNA_R2   = 0.351
OPTUNA_MAE  = 0.436
OPTUNA_RMSE = 0.587

import yaml

# Cargar config, cambiar epochs a 50, guardar en temporal
with open(CONFIG, "r") as f:
    cfg = yaml.safe_load(f)

cfg["training"]["epochs"] = 50
cfg["experiment"]["name"] = "unet_v4_ms_sx200_hpo_50ep"
cfg["output"]["results_dir"] = "results/unet_v4_ms_sx200_hpo_50ep"
cfg["output"]["model_name"]  = "unet_v4_ms_sx200_hpo_50ep"

tmp_config = ROOT / "configs/unet_v4_ms_sx200_hpo_50ep_tmp.yaml"
with open(tmp_config, "w") as f:
    yaml.dump(cfg, f, default_flow_style=False, allow_unicode=True)

print("=" * 60)
print("  Test reproducibilidad UNet HPO — 50 epocas")
print(f"  Optuna trial #11: R2={OPTUNA_R2} | MAE={OPTUNA_MAE} | RMSE={OPTUNA_RMSE}")
print("=" * 60)

proc = subprocess.run(
    [str(PYTHON), str(MAIN), "--config", str(tmp_config), "--mode", "both"],
    env={**__import__("os").environ, "PYTHONUNBUFFERED": "1", "PYTHONIOENCODING": "utf-8"},
)

# Limpiar config temporal
tmp_config.unlink(missing_ok=True)

if proc.returncode != 0:
    print(f"\nERROR: entrenamiento termino con exit={proc.returncode}")
    sys.exit(proc.returncode)

# Leer resultados del last checkpoint (que es el equivalente al Optuna)
results_dir = ROOT / "results/unet_v4_ms_sx200_hpo_50ep"
last_metrics_path = results_dir / "unet_v4_ms_sx200_hpo_50ep_last_metrics.json"
best_metrics_path = results_dir / "unet_v4_ms_sx200_hpo_50ep_metrics.json"

print("\n" + "=" * 60)
print("  RESULTADOS")
print("=" * 60)
print(f"  {'Metrica':<10} {'Optuna':>10} {'Last ep':>10} {'Best val':>10}")
print("  " + "-" * 42)

for path, label in [(last_metrics_path, "Last ep"), (best_metrics_path, "Best val")]:
    if path.exists():
        m = json.load(open(path))
        if label == "Last ep":
            for k, optuna_v in [("R2", OPTUNA_R2), ("MAE", OPTUNA_MAE), ("RMSE", OPTUNA_RMSE)]:
                diff = m[k] - optuna_v
                print(f"  {k:<10} {optuna_v:>10.4f} {m[k]:>10.4f}   (diff={diff:+.4f})")

print("=" * 60)
