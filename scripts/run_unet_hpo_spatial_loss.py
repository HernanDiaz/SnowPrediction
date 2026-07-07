"""
Barrido de lambda (spatial loss) para UNet HPO.
=================================================
Entrena UNet con hiperparametros optimos del Optuna (trial #11) usando
diferentes valores de lambda en la loss espacial MSE + lambda*(1-Pearson).

Lambdas: 0.1, 0.25, 0.4, 0.5, 0.6, 0.75, 1.0
Baseline MSE puro: results/unet_v4_ms_sx200_hpo_50ep/ (ya entrenado)

Uso:
    .venv/Scripts/python.exe scripts/run_unet_hpo_spatial_loss.py
"""

import subprocess
import sys
import time
from pathlib import Path

ROOT   = Path(__file__).resolve().parent.parent
PYTHON = ROOT / ".venv/Scripts/python.exe"
MAIN   = ROOT / "main.py"

CONFIGS = [
    "configs/unet_v4_ms_sx200_hpo_sp00.yaml",
    "configs/unet_v4_ms_sx200_hpo_sp01.yaml",
    "configs/unet_v4_ms_sx200_hpo_sp025.yaml",
    "configs/unet_v4_ms_sx200_hpo_sp04.yaml",
    "configs/unet_v4_ms_sx200_hpo_sp05.yaml",
    "configs/unet_v4_ms_sx200_hpo_sp06.yaml",
    "configs/unet_v4_ms_sx200_hpo_sp075.yaml",
    "configs/unet_v4_ms_sx200_hpo_sp10.yaml",
]

LAMBDAS = [0.0, 0.1, 0.25, 0.4, 0.5, 0.6, 0.75, 1.0]

print("=" * 65)
print("  Barrido lambda UNet HPO — spatial_mse")
print(f"  Lambdas: {LAMBDAS}")
print(f"  Epochs : 50 | No augmentation | No early stopping")
print("=" * 65)

t_total = time.time()
timings = {}

for cfg, lam in zip(CONFIGS, LAMBDAS):
    print(f"\n{'='*65}")
    print(f"  lambda = {lam}")
    print(f"  Config : {cfg}")
    print("=" * 65)

    t0 = time.time()
    proc = subprocess.run(
        [str(PYTHON), str(MAIN), "--config", cfg, "--mode", "both"],
        env={**__import__("os").environ,
             "PYTHONUNBUFFERED": "1",
             "PYTHONIOENCODING": "utf-8"},
    )
    elapsed = time.time() - t0
    timings[lam] = elapsed

    if proc.returncode not in (0, 1):
        print(f"\n[ERROR] lambda={lam} termino con exit={proc.returncode}. Abortando.")
        sys.exit(proc.returncode)

    print(f"\n  lambda={lam} completado en {elapsed/60:.1f} min")

print(f"\n{'='*65}")
print(f"  BARRIDO COMPLETADO | Tiempo total: {(time.time()-t_total)/3600:.1f} h")
print("=" * 65)
for lam, t in timings.items():
    print(f"  lambda={lam:<5}: {t/60:.1f} min")
