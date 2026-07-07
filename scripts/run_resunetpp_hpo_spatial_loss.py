"""
Barrido de lambda (SpatialMSELoss) con ResUNet++ HPO hyperparameters.
=====================================================================
Entrena secuencialmente 7 valores de lambda usando los hiperparametros
del mejor trial de Optuna (Trial 17, dataset_v4_ms_sx200).

Lambda=0.0 corresponde al experimento base:
    results/resunetpp_v4_ms_sx200_hpo/  (ya ejecutado)

Este script cubre lambda = 0.1, 0.25, 0.4, 0.5, 0.6, 0.75, 1.0

Uso:
    .venv/Scripts/python.exe scripts/run_resunetpp_hpo_spatial_loss.py
"""

import subprocess
import sys
from pathlib import Path

ROOT   = Path(__file__).resolve().parent.parent
PYTHON = ROOT / ".venv/Scripts/python.exe"
MAIN   = ROOT / "main.py"

CONFIGS = [
    "configs/resunetpp_v4_ms_sx200_hpo_sp01.yaml",
    "configs/resunetpp_v4_ms_sx200_hpo_sp025.yaml",
    "configs/resunetpp_v4_ms_sx200_hpo_sp04.yaml",
    "configs/resunetpp_v4_ms_sx200_hpo_sp05.yaml",
    "configs/resunetpp_v4_ms_sx200_hpo_sp06.yaml",
    "configs/resunetpp_v4_ms_sx200_hpo_sp075.yaml",
    "configs/resunetpp_v4_ms_sx200_hpo_sp10.yaml",
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

    print("\n\nBarrido lambda completado.")
    print("Resultados en results/resunetpp_v4_ms_sx200_hpo_sp*/")
