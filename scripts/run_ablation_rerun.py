"""
Re-ejecucion de experimentos de ablacion afectados por bug de normalizacion.
=============================================================================
Bug: channel_indices se aplicaba ANTES de _normalize(), causando que la
normalizacion usara posiciones incorrectas cuando se eliminaban canales
del inicio o del medio del array.

Experimentos afectados:
  - sin_sx   (14ch): Pers/Topo5m recibian normalizacion Sx incorrecta
  - sin_sce  (21ch): posicion 5 en adelante desplazada
  - sin_topo1 (17ch): SCE en posicion 0 -> normalizacion DEM -> NaN

Fix aplicado en data/dataset.py: normalizar ANTES de seleccionar canales.

Uso:
    .venv/Scripts/python.exe scripts/run_ablation_rerun.py
"""

import subprocess
import sys
from pathlib import Path

ROOT   = Path(__file__).resolve().parent.parent
PYTHON = ROOT / ".venv/Scripts/python.exe"
MAIN   = ROOT / "main.py"

CONFIGS = [
    # sin_sx (normalizacion incorrecta en posiciones 6-13)
    "configs/resunetpp_ablation_sin_sx_s42.yaml",
    "configs/resunetpp_ablation_sin_sx_s123.yaml",
    "configs/resunetpp_ablation_sin_sx_s7.yaml",
    # sin_sce (normalizacion incorrecta desde posicion 5)
    "configs/resunetpp_ablation_sin_sce_s42.yaml",
    "configs/resunetpp_ablation_sin_sce_s123.yaml",
    "configs/resunetpp_ablation_sin_sce_s7.yaml",
    # sin_topo1 (NaN por normalizacion DEM aplicada a SCE)
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

    print("\n\nRe-ejecucion completada.")
    print("Resultados actualizados en results/ablation/")
