"""
Master launcher: todos los experimentos v4_2p5m en secuencia.
=============================================================

Orden de ejecucion:
  1. UNet v4-2p5m       (~30-60 min, 150 ep, 63 steps/ep)
  2. ResUNet++ v4-2p5m  (~60-120 min, 300 ep, early stopping)

Dataset: dataset_v4_2p5m
  - Resolucion: 2.5m | Cobertura: 640m x 640m por tile
  - Canales: 17 (DEM, Slope, N, E, TPI, SCE, Sx_200m x8, Pers x3)
  - Split: 507 train / 55 val / 28 test

Uso:
    .venv\\Scripts\\python.exe scripts/run_all_v4_2p5m.py
"""

import subprocess
import sys
import time
from pathlib import Path

ROOT   = Path(__file__).resolve().parent.parent
PYTHON = ROOT / ".venv/Scripts/python.exe"
MAIN   = ROOT / "main.py"

EXPERIMENTS = [
    ("UNet v4-2p5m      [150 ep]", "unet_v4_2p5m",      "results/unet_v4_2p5m/run_log.txt"),
    ("ResUNet++ v4-2p5m [300 ep]", "resunetpp_v4_2p5m", "results/resunetpp_v4_2p5m/run_log.txt"),
]


def banner(msg):
    sep = "=" * 60
    print(f"\n{sep}\n  {msg}\n{sep}", flush=True)


def run_step(label, cmd, log_path):
    log_path = ROOT / log_path
    log_path.parent.mkdir(parents=True, exist_ok=True)
    banner(label)
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
            print(line, end="", flush=True)
            logf.write(line)
        proc.wait()
    elapsed = time.time() - t0
    print(f"\n  Tiempo: {elapsed/60:.1f} min  |  Exit code: {proc.returncode}",
          flush=True)
    return proc.returncode, elapsed


if __name__ == "__main__":
    if not PYTHON.exists():
        print(f"ERROR: venv no encontrado: {PYTHON}")
        sys.exit(1)

    timings   = {}
    t_global  = time.time()

    for label, exp_name, log_rel in EXPERIMENTS:
        cfg = ROOT / f"configs/{exp_name}.yaml"
        if not cfg.exists():
            print(f"ERROR: config no encontrado: {cfg}")
            sys.exit(1)

        rc, elapsed = run_step(
            f"{label}  |  TRAIN",
            [str(PYTHON), str(MAIN), "--config", str(cfg), "--mode", "train"],
            log_rel,
        )
        timings[f"{exp_name} train"] = elapsed
        if rc not in (0, 1):
            print(f"ERROR train (exit {rc}). Abortando.")
            sys.exit(rc)

        rc, elapsed = run_step(
            f"{label}  |  EVALUATE",
            [str(PYTHON), str(MAIN), "--config", str(cfg), "--mode", "evaluate"],
            log_rel,
        )
        timings[f"{exp_name} eval"] = elapsed

    total = time.time() - t_global
    banner("PIPELINE v4-2p5m  COMPLETADO")
    print(f"  Tiempo total: {total/60:.1f} min\n")
    for k, v in timings.items():
        print(f"  {k:35s}: {v/60:.1f} min")
    print(f"\n  Resultados en:")
    print(f"    {ROOT / 'results/unet_v4_2p5m'}")
    print(f"    {ROOT / 'results/resunetpp_v4_2p5m'}")
