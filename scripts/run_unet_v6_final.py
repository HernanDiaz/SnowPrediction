"""
Entrenamiento final UNet v6 (17 canales: topo + SCE + Sx_100m + persistencia).
==============================================================================

Uso:
    E:\\PycharmProjects\\SnowPrediction\\.venv\\Scripts\\python.exe run_unet_v6_final.py

Salidas:
    Pesos    : Articulo 1/Models/unet_v6_final.pth
    Metricas : results/unet_v6_final/unet_v6_final_metrics.json
    Curva    : results/unet_v6_final/unet_v6_final_training_curve.png
    Scatter  : results/unet_v6_final/unet_v6_final_scatter.png
"""

import subprocess
import sys
import os
import time
from pathlib import Path

# ---------------------------------------------------------------------------
# Rutas
# ---------------------------------------------------------------------------
ROOT   = Path("E:/PycharmProjects/SnowPrediction")
PYTHON = ROOT / ".venv/Scripts/python.exe"
MAIN   = ROOT / "main.py"
CONFIG = ROOT / "configs/unet_v6_final.yaml"
LOG    = ROOT / "results/unet_v6_final/run_log.txt"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def banner(msg):
    sep = "=" * 60
    print(f"\n{sep}\n  {msg}\n{sep}")

def run_step(label, cmd, log_path):
    banner(label)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    with open(log_path, "a", encoding="utf-8") as logf:
        logf.write(f"\n{'='*60}\n{label}\n{'='*60}\n")
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        for line in proc.stdout:
            print(line, end="")
            logf.write(line)
        proc.wait()
    elapsed = time.time() - t0
    print(f"\n  Tiempo: {elapsed/60:.1f} min  |  Exit code: {proc.returncode}")
    return proc.returncode

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print(f"Python  : {PYTHON}")
    print(f"Config  : {CONFIG}")
    print(f"Log     : {LOG}")

    if not PYTHON.exists():
        print(f"ERROR: no se encuentra el entorno virtual en {PYTHON}")
        sys.exit(1)
    if not CONFIG.exists():
        print(f"ERROR: config no encontrado: {CONFIG}")
        sys.exit(1)

    # Paso 1: Entrenamiento
    rc = run_step(
        "UNet v6 Final  [1/2]  ENTRENAMIENTO",
        [str(PYTHON), str(MAIN), "--config", str(CONFIG), "--mode", "train"],
        LOG,
    )
    if rc not in (0, 1):
        print(f"ERROR en entrenamiento (exit code {rc}). Abortando.")
        sys.exit(rc)

    # Paso 2: Evaluacion en test
    rc = run_step(
        "UNet v6 Final  [2/2]  EVALUACION TEST",
        [str(PYTHON), str(MAIN), "--config", str(CONFIG), "--mode", "evaluate"],
        LOG,
    )
    if rc not in (0, 1):
        print(f"ERROR en evaluacion (exit code {rc}).")
        sys.exit(rc)

    banner("UNet v6 Final  COMPLETADO")
    print(f"  Resultados en: {ROOT / 'results/unet_v6_final'}")
    print(f"  Pesos en     : {ROOT / 'Articulo 1/Models/unet_v6_final.pth'}")
