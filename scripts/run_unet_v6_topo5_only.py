"""
UNet en dataset v6 (5m) con solo 5 canales topograficos.
==========================================================

Comparacion de resolucion directa:
    UNet v4 (1m, 5 topo, 3134 tiles)  -> experiment: unet_v4_1m_topo5
    UNet v6 (5m, 5 topo, ~60 tiles)   -> este experimento

Objetivo: aislar el efecto de la resolucion (contexto espacial) del efecto
de los canales adicionales (Sx, persistencia) en el dataset v6.

Salidas:
    Pesos    : Articulo 1/Models/unet_v6_topo5_only.pth
    Metricas : results/unet_v6_topo5_only/unet_v6_topo5_only_metrics.json
"""

import subprocess
import sys
import time
from pathlib import Path

ROOT   = Path("E:/PycharmProjects/SnowPrediction")
PYTHON = ROOT / ".venv/Scripts/python.exe"
MAIN   = ROOT / "main.py"
CONFIG = ROOT / "configs/unet_v6_topo5_only.yaml"
LOG    = ROOT / "results/unet_v6_topo5_only/run_log.txt"


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
            env={**__import__("os").environ, "PYTHONUNBUFFERED": "1"},
        )
        for line in proc.stdout:
            print(line, end="")
            logf.write(line)
        proc.wait()
    elapsed = time.time() - t0
    print(f"\n  Tiempo: {elapsed/60:.1f} min  |  Exit code: {proc.returncode}")
    return proc.returncode


if __name__ == "__main__":
    if not PYTHON.exists():
        sys.exit(1)
    if not CONFIG.exists():
        print(f"ERROR: config no encontrado: {CONFIG}")
        sys.exit(1)

    rc = run_step(
        "UNet v6-5m  5 topo only  [1/2]  TRAIN",
        [str(PYTHON), str(MAIN), "--config", str(CONFIG), "--mode", "train"],
        LOG,
    )
    if rc not in (0, 1):
        sys.exit(rc)

    rc = run_step(
        "UNet v6-5m  5 topo only  [2/2]  EVALUATE",
        [str(PYTHON), str(MAIN), "--config", str(CONFIG), "--mode", "evaluate"],
        LOG,
    )
    if rc not in (0, 1):
        sys.exit(rc)

    banner("UNet v6 topo5-only  COMPLETADO")
    print(f"  Resultados en: {ROOT / 'results/unet_v6_topo5_only'}")
