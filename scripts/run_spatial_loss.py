"""
Experimento: loss espacial MSE + lambda*(1-Pearson).
=====================================================

Entrena ResUNet++ v4-ms-Sx200 con tres valores de lambda:
  - sp01: lambda=0.1  (penalizacion suave del patron espacial)
  - sp05: lambda=0.5  (equilibrio)
  - sp10: lambda=1.0  (fuerte enfasis en patron espacial)

El modelo base (MSE puro, lambda=0.0) es resunetpp_v4_ms_sx200.
Los resultados permiten trazar la curva SPAEF vs R2 en funcion de lambda,
que es la contribucion empirica clave del articulo.

Uso:
    .venv\\Scripts\\python.exe scripts/run_spatial_loss.py
"""

import subprocess
import sys
import time
from pathlib import Path

ROOT   = Path(__file__).resolve().parent.parent
PYTHON = ROOT / ".venv/Scripts/python.exe"
MAIN   = ROOT / "main.py"
LOG    = ROOT / "results/run_spatial_loss_log.txt"

EXPERIMENTS = [
    ("sp01  lambda=0.1", "configs/resunetpp_v4_ms_sx200_sp01.yaml"),
    ("sp05  lambda=0.5", "configs/resunetpp_v4_ms_sx200_sp05.yaml"),
    ("sp10  lambda=1.0", "configs/resunetpp_v4_ms_sx200_sp10.yaml"),
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
    banner("EXPERIMENTO SPATIAL LOSS COMPLETADO")
    print(f"  Tiempo total : {total/60:.1f} min\n")
    for k, v in timings.items():
        print(f"  {k:40s}: {v/60:.1f} min")

    if errors:
        print(f"\n  ERRORES ({len(errors)}):")
        for e in errors:
            print(f"    - {e}")
    else:
        print("\n  Sin errores.")

    print(f"\n  Log: {LOG}")
    print(f"  Resultados en: {ROOT / 'results'}")
