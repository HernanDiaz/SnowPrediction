"""
Lanzador del Random Forest en dataset v4_17ch (1m, 17 canales).
"""

import subprocess
import sys
import time
from pathlib import Path

ROOT   = Path(__file__).resolve().parent.parent
PYTHON = ROOT / ".venv/Scripts/python.exe"
SCRIPT = ROOT / "baselines/rf_v4_17ch.py"
LOG    = ROOT / "results/rf_v4_17ch/run_log.txt"


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

    rc = run_step(
        "RF v4-17ch  |  Train + Evaluate",
        [str(PYTHON), str(SCRIPT)],
        LOG,
    )
    if rc != 0:
        print(f"ERROR (exit {rc}).")
        sys.exit(rc)

    banner("RF v4-17ch  COMPLETADO")
    print(f"  Resultados en: {ROOT / 'results/rf_v4_17ch'}")
