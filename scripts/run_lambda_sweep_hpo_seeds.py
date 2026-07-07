"""
Barrido lambda HPO con seeds adicionales (123 y 7).
====================================================
Repite el barrido de SpatialMSELoss con hiperparametros Optuna
(base=64, adamw, lr=1.287e-4, bs=8, 50 epocas) usando dos seeds
adicionales para cuantificar varianza.

Seed=42 ya esta en: results/resunetpp_v4_ms_sx200_hpo_sp*/
Seeds 123 y 7 van a: results/lambda_sweep_hpo/

Lambdas: 0.1, 0.25, 0.4, 0.5, 0.6, 0.75, 1.0
Lambda=0.0 (MSE puro) ya tiene 3 seeds en ablacion/full_s*

Uso:
    .venv/Scripts/python.exe scripts/run_lambda_sweep_hpo_seeds.py
"""

import subprocess
import sys
from pathlib import Path

ROOT   = Path(__file__).resolve().parent.parent
PYTHON = ROOT / ".venv/Scripts/python.exe"
MAIN   = ROOT / "main.py"

CONFIGS = [
    "configs/resunetpp_hpo_sp01_s123.yaml",
    "configs/resunetpp_hpo_sp01_s7.yaml",
    "configs/resunetpp_hpo_sp025_s123.yaml",
    "configs/resunetpp_hpo_sp025_s7.yaml",
    "configs/resunetpp_hpo_sp04_s123.yaml",
    "configs/resunetpp_hpo_sp04_s7.yaml",
    "configs/resunetpp_hpo_sp05_s123.yaml",
    "configs/resunetpp_hpo_sp05_s7.yaml",
    "configs/resunetpp_hpo_sp06_s123.yaml",
    "configs/resunetpp_hpo_sp06_s7.yaml",
    "configs/resunetpp_hpo_sp075_s123.yaml",
    "configs/resunetpp_hpo_sp075_s7.yaml",
    "configs/resunetpp_hpo_sp10_s123.yaml",
    "configs/resunetpp_hpo_sp10_s7.yaml",
]

if __name__ == "__main__":
    for config in CONFIGS:
        print(f"\n{'='*60}")
        print(f"  Lanzando: {config}")
        print(f"{'='*60}\n")
        ret = subprocess.run(
            [str(PYTHON), str(MAIN), "--config", config, "--mode", "both"],
            cwd=str(ROOT),
        )
        if ret.returncode != 0:
            print(f"\n[ERROR] Fallo en {config} (codigo {ret.returncode}). Abortando.")
            sys.exit(ret.returncode)

    print("\n\nBarrido lambda HPO seeds completado.")
    print("Resultados en results/lambda_sweep_hpo/")
