"""
PAPER - Block 3: Seed sensitivity analysis.
============================================
Trains ResUNet++ (22ch Sx200m, MSE loss) with seeds 1, 2, 3 to assess
variance in R2 and SPAEF due to random initialization.
Compare with b1_resunetpp (seed=42) as the reference run.

IMPORTANT: These use dataset_v4_ms_sx200, unlike the old exploratory
configs (resunetpp_v4_ms_s1/s2/s3) which accidentally used dataset_v4_ms.

Uso:
    .venv\\Scripts\\python.exe paper/scripts/run_block3_seeds.py
"""

import subprocess
import sys
import time
from pathlib import Path

ROOT   = Path(__file__).resolve().parent.parent.parent
PYTHON = ROOT / ".venv/Scripts/python.exe"
MAIN   = ROOT / "main.py"
LOG    = ROOT / "paper/results/block3_seeds/run_block3_seeds_log.txt"

EXPERIMENTS = [
    "paper/configs/block3_seeds/b3_seed1.yaml",
    "paper/configs/block3_seeds/b3_seed2.yaml",
    "paper/configs/block3_seeds/b3_seed3.yaml",
]


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
    metrics_path = PAPER_RES / f"block3_seeds/{exp_name}/{exp_name}_metrics.json"
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

    t_global = time.time()
    timings = {}
    errors = []

    banner(f"BLOCK 3 - Seed sensitivity  |  {len(EXPERIMENTS)} experiments")
    print(f"  Log: {LOG}\n", flush=True)
    print("  Reference run: b1_resunetpp (seed=42) from block1\n", flush=True)

    for cfg_rel in EXPERIMENTS:
        rc, elapsed = run_experiment(cfg_rel, mode="both")
        exp_name = Path(cfg_rel).stem
        timings[exp_name] = elapsed
        if rc is not None and rc not in (0, 1):
            errors.append(f"{exp_name}: exit {rc}")

    total = time.time() - t_global
    banner("BLOCK 3 SEEDS COMPLETADO")
    print(f"  Tiempo total: {total/60:.1f} min\n")
    for k, v in timings.items():
        print(f"  {k:<35s}: {v/60:.1f} min")
    if errors:
        print(f"\n  ERRORES: {errors}")
    else:
        print("\n  Sin errores.")
    print(f"\n  Log: {LOG}")
