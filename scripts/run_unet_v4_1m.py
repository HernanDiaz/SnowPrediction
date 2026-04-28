"""
Experimento UNet en dataset v4 a 1m de resolucion.
===================================================

Dos variantes:
  (A) unet_v4_1m_topo5 : 5 canales topograficos (comparable a RF v5)
  (B) unet_v4_1m_sce6  : 5 topo + SCE          (comparable a RF v6)

Dataset: dataset_v4_fisico (7888 tiles LiDAR a 1m, filtrados >30% cobertura)
Split  : temporal (train=2021-2022, val=2023, test=2024-2025)
Tiles  : 3134 train / 1107 val / 2575 test

Salidas:
  Pesos    : Articulo 1/Models/unet_v4_1m_{topo5|sce6}.pth
  Metricas : results/unet_v4_1m_{topo5|sce6}/*_metrics.json
  Curva    : results/unet_v4_1m_{topo5|sce6}/*_training_curve.png
"""

import subprocess
import sys
import time
from pathlib import Path

ROOT   = Path(__file__).resolve().parent.parent
PYTHON = ROOT / ".venv/Scripts/python.exe"
MAIN   = ROOT / "main.py"
LOG    = ROOT / "results/unet_v4_1m/run_log.txt"

EXPERIMENTS = [
    ("UNet v4-1m  [A]  5 topo channels", "unet_v4_1m_topo5"),
    ("UNet v4-1m  [B]  5 topo + SCE",   "unet_v4_1m_sce6"),
]

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
            env={**__import__("os").environ, "PYTHONUNBUFFERED": "1"},
        )
        for line in proc.stdout:
            print(line, end="")
            logf.write(line)
        proc.wait()
    elapsed = time.time() - t0
    print(f"\n  Tiempo: {elapsed/60:.1f} min  |  Exit code: {proc.returncode}")
    return proc.returncode

# ---------------------------------------------------------------------------
if __name__ == "__main__":
    if not PYTHON.exists():
        print(f"ERROR: venv no encontrado: {PYTHON}")
        sys.exit(1)

    for label, exp_name in EXPERIMENTS:
        cfg = ROOT / f"configs/{exp_name}.yaml"
        if not cfg.exists():
            print(f"ERROR: config no encontrado: {cfg}")
            sys.exit(1)

        # Train
        rc = run_step(
            f"{label}  |  TRAIN",
            [str(PYTHON), str(MAIN), "--config", str(cfg), "--mode", "train"],
            LOG,
        )
        if rc not in (0, 1):
            print(f"ERROR entrenamiento (exit {rc}). Abortando.")
            sys.exit(rc)

        # Evaluate
        rc = run_step(
            f"{label}  |  EVALUATE",
            [str(PYTHON), str(MAIN), "--config", str(cfg), "--mode", "evaluate"],
            LOG,
        )
        if rc not in (0, 1):
            print(f"ERROR evaluacion (exit {rc}).")
            sys.exit(rc)

    banner("UNet v4-1m  COMPLETADO")
    print(f"  Resultados en: {ROOT / 'results/unet_v4_1m_topo5'}")
    print(f"                 {ROOT / 'results/unet_v4_1m_sce6'}")
