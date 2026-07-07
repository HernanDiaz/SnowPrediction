"""
Calcula MSPAEF en los experimentos del paper que ya tienen modelo entrenado.
==========================================================================
Re-ejecuta --mode evaluate sobre cada experimento (sin reentrenar).
Los JSON de metricas se actualizan añadiendo MSPAEF, MSPAEF_std, MSPAEF_n_tiles.

Experimentos procesados:
    resunetpp_v4_ms_sx200        (lambda=0, sin seed fijo — referencia base)
    resunetpp_v4_ms_sx200_sp01   (lambda=0.10)
    resunetpp_v4_ms_sx200_sp025  (lambda=0.25)
    resunetpp_v4_ms_sx200_sp04   (lambda=0.40)
    resunetpp_v4_ms_sx200_sp05   (lambda=0.50)
    resunetpp_v4_ms_sx200_sp06   (lambda=0.60)
    resunetpp_v4_ms_sx200_sp075  (lambda=0.75)
    resunetpp_v4_ms_sx200_sp10   (lambda=1.00)

Uso:
    .venv\\Scripts\\python.exe scripts/run_compute_mspaef.py
"""

import subprocess
import sys
import time
from pathlib import Path

ROOT   = Path(__file__).resolve().parent.parent
PYTHON = ROOT / ".venv/Scripts/python.exe"
MAIN   = ROOT / "main.py"
LOG    = ROOT / "results/run_compute_mspaef_log.txt"

EXPERIMENTS = [
    "configs/resunetpp_v4_ms_sx200.yaml",
    "configs/resunetpp_v4_ms_sx200_sp01.yaml",
    "configs/resunetpp_v4_ms_sx200_sp025.yaml",
    "configs/resunetpp_v4_ms_sx200_sp04.yaml",
    "configs/resunetpp_v4_ms_sx200_sp05.yaml",
    "configs/resunetpp_v4_ms_sx200_sp06.yaml",
    "configs/resunetpp_v4_ms_sx200_sp075.yaml",
    "configs/resunetpp_v4_ms_sx200_sp10.yaml",
]


def banner(msg):
    sep = "=" * 60
    print(f"\n{sep}\n  {msg}\n{sep}", flush=True)


def run_evaluate(config_rel):
    cfg = ROOT / config_rel
    if not cfg.exists():
        print(f"  [SKIP] Config no encontrado: {cfg}", flush=True)
        return None, 0.0

    exp_name = cfg.stem
    banner(f"EVALUATE (MSPAEF): {exp_name}")

    t0 = time.time()
    with open(LOG, "a", encoding="utf-8") as logf:
        logf.write(f"\n{'='*60}\n{exp_name}\n{'='*60}\n")
        proc = subprocess.Popen(
            [str(PYTHON), str(MAIN), "--config", str(cfg), "--mode", "evaluate"],
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
    print(f"\n  Tiempo: {elapsed/60:.1f} min  |  Exit: {proc.returncode}", flush=True)
    return proc.returncode, elapsed


if __name__ == "__main__":
    if not PYTHON.exists():
        print(f"ERROR: venv no encontrado: {PYTHON}")
        sys.exit(1)

    t_global = time.time()
    banner(f"Calculando MSPAEF en {len(EXPERIMENTS)} experimentos existentes")
    print(f"  Log: {LOG}\n")

    errors = []
    for cfg_rel in EXPERIMENTS:
        rc, _ = run_evaluate(cfg_rel)
        if rc is not None and rc != 0:
            errors.append(Path(cfg_rel).stem)

    total = time.time() - t_global
    banner("COMPLETADO")
    print(f"  Tiempo total: {total/60:.1f} min")
    if errors:
        print(f"  Errores en: {errors}")
    else:
        print("  Todos los JSON actualizados con MSPAEF.")
    print(f"  Log: {LOG}")
