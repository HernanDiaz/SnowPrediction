"""
Lanzador del ResUNet++ en dataset v4_17ch (1m, 17 canales, 300 epocas).

Replica exactamente la configuracion del mejor modelo v6 (resunetpp_v6_300ep)
cambiando solo el dataset (1m vs 5m). Permite comparacion directa de resolucion.

Dataset : dataset_v4_17ch
Split   : train=2021-2023 (4241), val=2024 (431), test=2025 (220)
Config  : configs/resunetpp_v4_17ch.yaml
Salidas : results/resunetpp_v4_17ch/
"""

import subprocess
import sys
import time
from pathlib import Path

ROOT   = Path(__file__).resolve().parent.parent
PYTHON = ROOT / ".venv/Scripts/python.exe"
MAIN   = ROOT / "main.py"
CFG    = ROOT / "configs/resunetpp_v4_17ch.yaml"
LOG    = ROOT / "results/resunetpp_v4_17ch/run_log.txt"


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
        print(f"ERROR: venv no encontrado: {PYTHON}")
        sys.exit(1)
    if not CFG.exists():
        print(f"ERROR: config no encontrado: {CFG}")
        sys.exit(1)

    rc = run_step(
        "ResUNet++ v4-17ch  |  TRAIN  (300 ep)",
        [str(PYTHON), str(MAIN), "--config", str(CFG), "--mode", "train"],
        LOG,
    )
    if rc not in (0, 1):
        print(f"ERROR entrenamiento (exit {rc}). Abortando.")
        sys.exit(rc)

    rc = run_step(
        "ResUNet++ v4-17ch  |  EVALUATE",
        [str(PYTHON), str(MAIN), "--config", str(CFG), "--mode", "evaluate"],
        LOG,
    )
    if rc not in (0, 1):
        print(f"ERROR evaluacion (exit {rc}).")
        sys.exit(rc)

    banner("ResUNet++ v4-17ch  COMPLETADO")
    print(f"  Resultados en: {ROOT / 'results/resunetpp_v4_17ch'}")
