"""
Pipeline v6_improved: genera dataset 5m mejorado y entrena ResUNet++.
=====================================================================

Pasos:
  1. Genera dataset_v6_improved (5m, Sx_200m, split 2021-23/2024/2025)
  2. Entrena  ResUNet++ v6_improved (300 ep, early stopping patience=30)
  3. Evalua   ResUNet++ v6_improved

Uso:
    .venv\\Scripts\\python.exe scripts/run_v6_improved.py
"""

import subprocess
import sys
import time
from pathlib import Path

ROOT   = Path(__file__).resolve().parent.parent
PYTHON = ROOT / ".venv/Scripts/python.exe"
MAIN   = ROOT / "main.py"
GEN    = ROOT / "data/generate_dataset_v6_improved.py"
CFG    = ROOT / "configs/resunetpp_v6_improved.yaml"
LOG    = ROOT / "results/resunetpp_v6_improved/run_log.txt"


def banner(msg):
    sep = "=" * 60
    print(f"\n{sep}\n  {msg}\n{sep}", flush=True)


def run_step(label, cmd, log_path):
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
            env={**__import__("os").environ,
                 "PYTHONUNBUFFERED": "1",
                 "PYTHONIOENCODING": "utf-8"},
        )
        for line in proc.stdout:
            sys.stdout.buffer.write(line.encode("utf-8", errors="replace"))
            sys.stdout.buffer.flush()
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

    t_global = time.time()
    timings  = {}

    rc, el = run_step("PASO 1/3  |  Generar dataset_v6_improved",
                      [str(PYTHON), str(GEN)], LOG)
    timings["generate"] = el
    if rc != 0:
        print(f"ERROR generando dataset (exit {rc}).")
        sys.exit(rc)

    rc, el = run_step("PASO 2/3  |  Train ResUNet++ v6_improved",
                      [str(PYTHON), str(MAIN), "--config", str(CFG),
                       "--mode", "train"], LOG)
    timings["train"] = el
    if rc not in (0, 1):
        print(f"ERROR entrenamiento (exit {rc}).")
        sys.exit(rc)

    rc, el = run_step("PASO 3/3  |  Evaluate ResUNet++ v6_improved",
                      [str(PYTHON), str(MAIN), "--config", str(CFG),
                       "--mode", "evaluate"], LOG)
    timings["eval"] = el

    total = time.time() - t_global
    banner("PIPELINE v6_improved COMPLETADO")
    print(f"  Tiempo total : {total/60:.1f} min\n")
    for k, v in timings.items():
        print(f"  {k:12s}: {v/60:.1f} min")
    print(f"\n  Resultados en: {ROOT / 'results/resunetpp_v6_improved'}")
