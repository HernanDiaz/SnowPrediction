"""
PAPER - Block 1: Comparacion de arquitecturas (script unico).
=============================================================
Ejecuta TODOS los experimentos del Block 1 en secuencia:

    1. DEM regression  (baseline fisico, ~3 min, CPU)
    2. RF 22ch         (baseline ML, ~2-3 h, CPU)
    3. U-Net 22ch      (CNN baseline, ~3-4 h, GPU)
    4. ResUNet++ 22ch  (modelo propuesto, ~3-4 h, GPU)

Todos usan dataset_v4_ms_sx200, split temporal identico y seed=42.
Los resultados se guardan en paper/results/block1/.

Si un experimento ya tiene resultados en paper/results/, se salta
automaticamente (util para reanudar ejecuciones interrumpidas).

Uso:
    .venv\\Scripts\\python.exe paper/scripts/run_block1.py

    # Solo CNNs (si RF y DEM ya estan listos):
    .venv\\Scripts\\python.exe paper/scripts/run_block1.py --skip-cpu
"""

import argparse
import subprocess
import sys
import time
from pathlib import Path

ROOT   = Path(__file__).resolve().parent.parent.parent
PYTHON = ROOT / ".venv/Scripts/python.exe"
MAIN   = ROOT / "main.py"
LOG    = ROOT / "paper/results/block1/run_block1_log.txt"
PAPER_RES = ROOT / "paper/results"


def banner(msg):
    sep = "=" * 60
    print(f"\n{sep}\n  {msg}\n{sep}", flush=True)


def run_script(script_rel, desc):
    """Ejecuta un script Python externo (RF, DEM regression)."""
    script = ROOT / script_rel
    metrics_name = script.stem + "_metrics.json"
    metrics_path = PAPER_RES / f"block1/{script.stem}/{metrics_name}"

    if metrics_path.exists():
        print(f"  [SKIP] Ya existe: {metrics_path.relative_to(ROOT)}", flush=True)
        return 0, 0.0

    banner(desc)
    LOG.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    with open(LOG, "a", encoding="utf-8") as logf:
        logf.write(f"\n{'='*60}\n{desc}\n{'='*60}\n")
        proc = subprocess.Popen(
            [str(PYTHON), str(script)],
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


def run_config(config_rel, desc):
    """Ejecuta un experimento CNN via main.py."""
    cfg = ROOT / config_rel
    exp_name = cfg.stem
    metrics_path = PAPER_RES / f"block1/{exp_name}/{exp_name}_metrics.json"

    if metrics_path.exists():
        print(f"  [SKIP] Ya existe: {metrics_path.relative_to(ROOT)}", flush=True)
        return 0, 0.0

    banner(desc)
    LOG.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    with open(LOG, "a", encoding="utf-8") as logf:
        logf.write(f"\n{'='*60}\n{desc}\n{'='*60}\n")
        proc = subprocess.Popen(
            [str(PYTHON), str(MAIN), "--config", str(cfg), "--mode", "both"],
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
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-cpu", action="store_true",
                        help="Saltar experimentos CPU (DEM regression y RF)")
    args = parser.parse_args()

    if not PYTHON.exists():
        print(f"ERROR: venv no encontrado: {PYTHON}")
        sys.exit(1)

    t_global = time.time()
    timings = {}
    errors  = []

    banner("BLOCK 1 — Comparacion de arquitecturas")
    print(f"  Dataset : dataset_v4_ms_sx200 (22ch, 1m)")
    print(f"  Split   : train=2021-2023 | val=2024 | test=2025")
    print(f"  Seed    : 42")
    print(f"  Log     : {LOG}\n")

    # --- Paso 1: DEM regression (CPU, ~3 min) ---
    if not args.skip_cpu:
        rc, t = run_script(
            "paper/scripts/run_block1_dem_regression.py",
            "1/4 DEM regression (baseline fisico)"
        )
        timings["b1_dem_regression"] = t
        if rc not in (0, 1):
            errors.append("b1_dem_regression")

    # --- Paso 2: RF 22ch (CPU, ~2-3 h) ---
    if not args.skip_cpu:
        rc, t = run_script(
            "paper/scripts/run_block1_rf.py",
            "2/4 Random Forest 22ch (baseline ML)"
        )
        timings["b1_rf"] = t
        if rc not in (0, 1):
            errors.append("b1_rf")

    # --- Paso 3: U-Net 22ch (GPU, ~3-4 h) ---
    rc, t = run_config(
        "paper/configs/block1_baselines/b1_unet.yaml",
        "3/4 U-Net 22ch (CNN baseline)"
    )
    timings["b1_unet"] = t
    if rc not in (0, 1):
        errors.append("b1_unet")

    # --- Paso 4: ResUNet++ 22ch (GPU, ~3-4 h) ---
    rc, t = run_config(
        "paper/configs/block1_baselines/b1_resunetpp.yaml",
        "4/4 ResUNet++ 22ch (modelo propuesto)"
    )
    timings["b1_resunetpp"] = t
    if rc not in (0, 1):
        errors.append("b1_resunetpp")

    total = time.time() - t_global
    banner("BLOCK 1 COMPLETADO")
    print(f"  Tiempo total: {total/60:.1f} min\n")
    for k, v in timings.items():
        print(f"  {k:<25s}: {v/60:.1f} min")
    if errors:
        print(f"\n  ERRORES: {errors}")
    else:
        print("\n  Sin errores.")
    print(f"\n  Resultados en: paper/results/block1/")
    print(f"  Log           : {LOG}")
