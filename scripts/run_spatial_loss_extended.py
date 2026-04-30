"""
Experimento: frente de Pareto R2-SPAEF con loss espacial.
==========================================================

Completa el barrido de lambda con valores intermedios:
  - sp025: lambda=0.25
  - sp04 : lambda=0.40
  - sp06 : lambda=0.60
  - sp075: lambda=0.75

Los extremos (0.0, 0.1, 0.5, 1.0) ya estan entrenados.
Con estos 4 nuevos puntos el frente de Pareto queda denso y publicable.

Uso:
    .venv\\Scripts\\python.exe scripts/run_spatial_loss_extended.py
"""

import subprocess
import sys
import time
from pathlib import Path

ROOT   = Path(__file__).resolve().parent.parent
PYTHON = ROOT / ".venv/Scripts/python.exe"
MAIN   = ROOT / "main.py"
LOG    = ROOT / "results/run_spatial_loss_extended_log.txt"

EXPERIMENTS = [
    ("sp025  lambda=0.25", "configs/resunetpp_v4_ms_sx200_sp025.yaml"),
    ("sp04   lambda=0.40", "configs/resunetpp_v4_ms_sx200_sp04.yaml"),
    ("sp06   lambda=0.60", "configs/resunetpp_v4_ms_sx200_sp06.yaml"),
    ("sp075  lambda=0.75", "configs/resunetpp_v4_ms_sx200_sp075.yaml"),
]


def banner(msg):
    sep = "=" * 60
    print(f"\n{sep}\n  {msg}\n{sep}", flush=True)


def run_step(label, cmd):
    LOG.parent.mkdir(parents=True, exist_ok=True)
    banner(label)
    t0 = time.time()
    with open(LOG, "a", encoding="utf-8") as logf:
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
    errors   = []

    for label, cfg_rel in EXPERIMENTS:
        cfg = ROOT / cfg_rel
        rc, el = run_step(
            f"Spatial loss | {label}",
            [str(PYTHON), str(MAIN), "--config", str(cfg), "--mode", "both"]
        )
        key = cfg_rel.split("/")[-1].replace(".yaml", "")
        timings[key] = el
        if rc not in (0, 1):
            errors.append(f"{key}: exit {rc}")
            print(f"  [WARN] {key} termino con exit {rc} — continuando...", flush=True)

    total = time.time() - t_global
    banner("FRENTE PARETO EXTENDIDO COMPLETADO")
    print(f"  Tiempo total : {total/60:.1f} min\n")
    for k, v in timings.items():
        print(f"  {k:45s}: {v/60:.1f} min")

    if errors:
        print(f"\n  ERRORES ({len(errors)}):")
        for e in errors:
            print(f"    - {e}")
    else:
        print("\n  Sin errores.")

    print(f"\n  Log: {LOG}")
    print(f"  Resultados en: {ROOT / 'results'}")
