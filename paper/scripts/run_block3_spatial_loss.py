"""
PAPER - Block 3: Spatial loss (SpatialMSELoss) lambda sweep.
=============================================================
Trains and evaluates ResUNet++ (22ch Sx200m) with 8 values of lambda,
tracing the Pareto front in R2-SPAEF space.

Lambda values: 0.0, 0.10, 0.25, 0.40, 0.50, 0.60, 0.75, 1.00

NOTE: b3_sp_l000 (lambda=0) is identical to b1_resunetpp. If block1 already ran,
results are copied to avoid redundant training.

Uso:
    .venv\\Scripts\\python.exe paper/scripts/run_block3_spatial_loss.py
"""

import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT   = Path(__file__).resolve().parent.parent.parent
PYTHON = ROOT / ".venv/Scripts/python.exe"
MAIN   = ROOT / "main.py"
LOG    = ROOT / "paper/results/block3_spatial_loss/run_block3_spatial_loss_log.txt"

EXPERIMENTS = [
    "paper/configs/block3_spatial_loss/b3_sp_l000.yaml",
    "paper/configs/block3_spatial_loss/b3_sp_l010.yaml",
    "paper/configs/block3_spatial_loss/b3_sp_l025.yaml",
    "paper/configs/block3_spatial_loss/b3_sp_l040.yaml",
    "paper/configs/block3_spatial_loss/b3_sp_l050.yaml",
    "paper/configs/block3_spatial_loss/b3_sp_l060.yaml",
    "paper/configs/block3_spatial_loss/b3_sp_l075.yaml",
    "paper/configs/block3_spatial_loss/b3_sp_l100.yaml",
]

B1_RESUNETPP_RESULTS = ROOT / "paper/results/block1/b1_resunetpp"
B3_L000_RESULTS      = ROOT / "paper/results/block3_spatial_loss/b3_sp_l000"


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
    metrics_path = PAPER_RES / f"block3_spatial_loss/{exp_name}/{exp_name}_metrics.json"
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

    # Reuse b1_resunetpp results for b3_sp_l000 (lambda=0 == pure MSE == b1_resunetpp)
    metrics_b1   = B1_RESUNETPP_RESULTS / "b1_resunetpp_metrics.json"
    metrics_l000 = B3_L000_RESULTS / "b3_sp_l000_metrics.json"
    if metrics_b1.exists() and not metrics_l000.exists():
        print(f"  [REUSE] Copiando resultados de b1_resunetpp -> b3_sp_l000")
        B3_L000_RESULTS.mkdir(parents=True, exist_ok=True)
        shutil.copy2(metrics_b1, metrics_l000)
        EXPERIMENTS_RUN = [e for e in EXPERIMENTS if "l000" not in e]
    else:
        EXPERIMENTS_RUN = EXPERIMENTS

    t_global = time.time()
    timings = {}
    errors = []

    banner(f"BLOCK 3 - Spatial loss sweep  |  {len(EXPERIMENTS_RUN)} experiments to run")
    print(f"  Log: {LOG}\n", flush=True)

    for cfg_rel in EXPERIMENTS_RUN:
        rc, elapsed = run_experiment(cfg_rel, mode="both")
        exp_name = Path(cfg_rel).stem
        timings[exp_name] = elapsed
        if rc is not None and rc not in (0, 1):
            errors.append(f"{exp_name}: exit {rc}")

    total = time.time() - t_global
    banner("BLOCK 3 SPATIAL LOSS COMPLETADO")
    print(f"  Tiempo total: {total/60:.1f} min\n")
    for k, v in timings.items():
        print(f"  {k:<35s}: {v/60:.1f} min")
    if errors:
        print(f"\n  ERRORES: {errors}")
    else:
        print("\n  Sin errores.")
    print(f"\n  Log: {LOG}")
