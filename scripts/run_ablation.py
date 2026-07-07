"""
Estudio de ablacion de canales — ResUNet++ HPO hyperparameters.
===============================================================
Leave-one-group-out con 3 seeds (42, 123, 7).

Grupos eliminados:
  full     — todos los canales (22ch) — baseline
  sin_sx   — sin Sx_200m (8 dirs)    -> 14ch
  sin_pers — sin Persistencia (3ch)  -> 19ch
  sin_topo5— sin Topo_5m (5ch)       -> 17ch
  sin_sce  — sin SCE (1ch)           -> 21ch
  sin_topo1— sin Topo_1m (5ch)       -> 17ch

El modelo full con seed=42 ya existe:
    results/resunetpp_v4_ms_sx200_hpo/

Uso:
    .venv/Scripts/python.exe scripts/run_ablation.py
"""

import subprocess
import sys
from pathlib import Path

ROOT   = Path(__file__).resolve().parent.parent
PYTHON = ROOT / ".venv/Scripts/python.exe"
MAIN   = ROOT / "main.py"

CONFIGS = [
    # full model con seeds adicionales
    "configs/resunetpp_ablation_full_s123.yaml",
    "configs/resunetpp_ablation_full_s7.yaml",
    # sin Sx_200m
    "configs/resunetpp_ablation_sin_sx_s42.yaml",
    "configs/resunetpp_ablation_sin_sx_s123.yaml",
    "configs/resunetpp_ablation_sin_sx_s7.yaml",
    # sin Persistencia
    "configs/resunetpp_ablation_sin_pers_s42.yaml",
    "configs/resunetpp_ablation_sin_pers_s123.yaml",
    "configs/resunetpp_ablation_sin_pers_s7.yaml",
    # sin Topo_5m
    "configs/resunetpp_ablation_sin_topo5_s42.yaml",
    "configs/resunetpp_ablation_sin_topo5_s123.yaml",
    "configs/resunetpp_ablation_sin_topo5_s7.yaml",
    # sin SCE
    "configs/resunetpp_ablation_sin_sce_s42.yaml",
    "configs/resunetpp_ablation_sin_sce_s123.yaml",
    "configs/resunetpp_ablation_sin_sce_s7.yaml",
    # sin Topo_1m
    "configs/resunetpp_ablation_sin_topo1_s42.yaml",
    "configs/resunetpp_ablation_sin_topo1_s123.yaml",
    "configs/resunetpp_ablation_sin_topo1_s7.yaml",
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

    print("\n\nAblacion completada.")
    print("Resultados en results/ablation/")
