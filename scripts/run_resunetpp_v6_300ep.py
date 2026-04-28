"""
ResUNet++ v6 entrenado 300 epocas con cosine scheduler y augmentation.
=======================================================================

Objetivo: superar R2=0.2495 (mejor resultado quick) sin overfitting al val set.

Cambios respecto al quick (resunetpp_v6_5m):
  - 300 epochs (vs 150)
  - Cosine LR scheduler (vs sin scheduler)
  - Augmentation activado, modo 'h' (solo flip horizontal)
  - Mismos features [48, 96, 192, 384] y LR=0.00039

Salidas:
  Pesos    : Articulo 1/Models/resunetpp_v6_300ep.pth
  Metricas : results/resunetpp_v6_300ep/resunetpp_v6_300ep_metrics.json
  Curva    : results/resunetpp_v6_300ep/resunetpp_v6_300ep_training_curve.png
"""

import subprocess
import sys
import time
from pathlib import Path

ROOT   = Path(__file__).resolve().parent.parent
PYTHON = ROOT / ".venv/Scripts/python.exe"
MAIN   = ROOT / "main.py"
CONFIG = ROOT / "configs/resunetpp_v6_300ep.yaml"
LOG    = ROOT / "results/resunetpp_v6_300ep/run_log.txt"


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
        print(f"ERROR: venv no encontrado")
        sys.exit(1)
    if not CONFIG.exists():
        print(f"ERROR: config no encontrado: {CONFIG}")
        sys.exit(1)

    rc = run_step(
        "ResUNet++ v6  300ep  [1/2]  ENTRENAMIENTO",
        [str(PYTHON), str(MAIN), "--config", str(CONFIG), "--mode", "train"],
        LOG,
    )
    if rc not in (0, 1):
        print(f"ERROR entrenamiento (exit {rc}). Abortando.")
        sys.exit(rc)

    rc = run_step(
        "ResUNet++ v6  300ep  [2/2]  EVALUACION TEST",
        [str(PYTHON), str(MAIN), "--config", str(CONFIG), "--mode", "evaluate"],
        LOG,
    )
    if rc not in (0, 1):
        print(f"ERROR evaluacion (exit {rc}).")
        sys.exit(rc)

    banner("ResUNet++ v6 300ep  COMPLETADO")
    print(f"  Resultados en: {ROOT / 'results/resunetpp_v6_300ep'}")
