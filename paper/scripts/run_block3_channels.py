"""
PAPER - Block 3: Channel ablation study.
=========================================
Trains and evaluates ResUNet++ with different channel configurations,
all on the same temporal split (2021-2023 train, 2024 val, 2025 test).

Experiments:
    b3_ch_5ch         : 5 topo channels (DEM, Slope, Northness, Eastness, TPI)
    b3_ch_17ch        : 17ch (+ SCE + Sx_200m x8 + Persistence x3)
    b3_ch_22ch_sx100  : 22ch with Sx at 100m radius
    b3_ch_22ch_sx200  : 22ch with Sx at 200m radius (reference)

NOTE: b3_ch_22ch_sx200 is identical to b1_resunetpp. If block1 already ran,
this script will re-train and overwrite results (same outcome, wasted compute).
Comment it out or add a skip check if block1 results already exist.

Uso:
    .venv\\Scripts\\python.exe paper/scripts/run_block3_channels.py
"""

import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT   = Path(__file__).resolve().parent.parent.parent
PYTHON = ROOT / ".venv/Scripts/python.exe"
MAIN   = ROOT / "main.py"
LOG    = ROOT / "paper/results/block3_channels/run_block3_channels_log.txt"

EXPERIMENTS = [
    "paper/configs/block3_channels/b3_ch_5ch.yaml",
    "paper/configs/block3_channels/b3_ch_17ch.yaml",
    "paper/configs/block3_channels/b3_ch_22ch_sx100.yaml",
    "paper/configs/block3_channels/b3_ch_22ch_sx200.yaml",
]

# b3_ch_22ch_sx200 == b1_resunetpp: reuse results if available to avoid re-training
B1_RESUNETPP_RESULTS = ROOT / "paper/results/block1/b1_resunetpp"
B3_SX200_RESULTS     = ROOT / "paper/results/block3_channels/b3_ch_22ch_sx200"


def banner(msg):
    sep = "=" * 60
    print(f"\n{sep}\n  {msg}\n{sep}", flush=True)


def run_experiment(config_rel, mode="both"):
    cfg = ROOT / config_rel
    if not cfg.exists():
        print(f"  [SKIP] Config no encontrado: {cfg}", flush=True)
        return None, 0.0

    exp_name = cfg.stem

    # Saltar si ya existen resultados en paper/results/
    metrics_path = PAPER_RES / f"block3_channels/{exp_name}/{exp_name}_metrics.json"
    if metrics_path.exists():
        print(f"  [SKIP] Ya existe: {metrics_path.relative_to(ROOT)}", flush=True)
        return 0, 0.0

    LOG.parent.mkdir(parents=True, exist_ok=True)
    label = f"{mode.upper()}: {exp_name}"
    banner(label)

    t0 = time.time()
    with open(LOG, "a", encoding="utf-8") as logf:
        logf.write(f"\n{'='*60}\n{label}\n{'='*60}\n")
        proc = subprocess.Popen(
            [str(PYTHON), str(MAIN), "--config", str(cfg), "--mode", mode],
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
    print(f"\n  Tiempo: {elapsed/60:.1f} min  |  Exit code: {proc.returncode}", flush=True)
    return proc.returncode, elapsed


if __name__ == "__main__":
    if not PYTHON.exists():
        print(f"ERROR: venv no encontrado: {PYTHON}")
        sys.exit(1)

    # Reuse b1_resunetpp results for b3_ch_22ch_sx200 if available
    metrics_b1 = B1_RESUNETPP_RESULTS / "b1_resunetpp_metrics.json"
    metrics_b3 = B3_SX200_RESULTS / "b3_ch_22ch_sx200_metrics.json"
    if metrics_b1.exists() and not metrics_b3.exists():
        print(f"  [REUSE] Copiando resultados de b1_resunetpp -> b3_ch_22ch_sx200")
        B3_SX200_RESULTS.mkdir(parents=True, exist_ok=True)
        shutil.copy2(metrics_b1, metrics_b3)
        # Remove sx200 from experiment list to avoid re-training
        EXPERIMENTS_RUN = [e for e in EXPERIMENTS if "sx200" not in e]
    else:
        EXPERIMENTS_RUN = EXPERIMENTS

    t_global = time.time()
    timings = {}
    errors = []

    banner(f"BLOCK 3 - Channel ablation  |  {len(EXPERIMENTS_RUN)} experiments to run")
    print(f"  Log: {LOG}\n", flush=True)

    for cfg_rel in EXPERIMENTS_RUN:
        rc, elapsed = run_experiment(cfg_rel, mode="both")
        exp_name = Path(cfg_rel).stem
        timings[exp_name] = elapsed
        if rc is not None and rc not in (0, 1):
            errors.append(f"{exp_name}: exit {rc}")

    total = time.time() - t_global
    banner("BLOCK 3 CHANNELS COMPLETADO")
    print(f"  Tiempo total: {total/60:.1f} min\n")
    for k, v in timings.items():
        print(f"  {k:<35s}: {v/60:.1f} min")
    if errors:
        print(f"\n  ERRORES: {errors}")
    else:
        print("\n  Sin errores.")
    print(f"\n  Log: {LOG}")
