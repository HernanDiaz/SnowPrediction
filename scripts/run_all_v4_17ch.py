"""
Master launcher: todos los experimentos v4_17ch en secuencia.
=============================================================

Orden de ejecucion:
  1. RF v4-17ch          (~30-40 min)
  2. UNet v4-17ch        (~2-3 h, 150 ep)
  3. ResUNet++ v4-17ch   (~4-6 h, 300 ep)

Total estimado: ~7-10 h en GPU RTX.

Uso:
    .venv\\Scripts\\python.exe scripts/run_all_v4_17ch.py

Los logs se guardan en results/<experimento>/run_log.txt
"""

import subprocess
import sys
import time
from pathlib import Path

ROOT   = Path(__file__).resolve().parent.parent
PYTHON = ROOT / ".venv/Scripts/python.exe"
MAIN   = ROOT / "main.py"

LOG_RF         = ROOT / "results/rf_v4_17ch/run_log.txt"
LOG_UNET       = ROOT / "results/unet_v4_17ch/run_log.txt"
LOG_RESUNETPP  = ROOT / "results/resunetpp_v4_17ch/run_log.txt"


def banner(msg):
    sep = "=" * 60
    print(f"\n{sep}\n  {msg}\n{sep}", flush=True)


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
            print(line, end="", flush=True)
            logf.write(line)
        proc.wait()
    elapsed = time.time() - t0
    print(f"\n  Tiempo: {elapsed/60:.1f} min  |  Exit code: {proc.returncode}",
          flush=True)
    return proc.returncode, elapsed


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    if not PYTHON.exists():
        print(f"ERROR: venv no encontrado: {PYTHON}")
        sys.exit(1)

    timings = {}
    t_global = time.time()

    # ------------------------------------------------------------------
    # 1. Random Forest v4-17ch
    # ------------------------------------------------------------------
    banner("PASO 1/3 — RF v4-17ch  (17 canales, 1m)")
    cfg_rf = ROOT / "baselines/rf_v4_17ch.py"
    rc, elapsed = run_step(
        "RF v4-17ch  |  Train + Evaluate",
        [str(PYTHON), str(cfg_rf)],
        LOG_RF,
    )
    timings["RF v4-17ch"] = elapsed
    if rc != 0:
        print(f"ERROR en RF (exit {rc}). Abortando pipeline.")
        sys.exit(rc)

    # ------------------------------------------------------------------
    # 2. UNet v4-17ch
    # ------------------------------------------------------------------
    banner("PASO 2/3 — UNet v4-17ch  (17 canales, 1m, 150 ep)")
    cfg_unet = ROOT / "configs/unet_v4_17ch.yaml"

    rc, elapsed = run_step(
        "UNet v4-17ch  |  TRAIN",
        [str(PYTHON), str(MAIN), "--config", str(cfg_unet), "--mode", "train"],
        LOG_UNET,
    )
    timings["UNet v4-17ch train"] = elapsed
    if rc not in (0, 1):
        print(f"ERROR UNet train (exit {rc}). Abortando.")
        sys.exit(rc)

    rc, elapsed = run_step(
        "UNet v4-17ch  |  EVALUATE",
        [str(PYTHON), str(MAIN), "--config", str(cfg_unet), "--mode", "evaluate"],
        LOG_UNET,
    )
    timings["UNet v4-17ch eval"] = elapsed

    # ------------------------------------------------------------------
    # 3. ResUNet++ v4-17ch
    # ------------------------------------------------------------------
    banner("PASO 3/3 — ResUNet++ v4-17ch  (17 canales, 1m, 300 ep)")
    cfg_res = ROOT / "configs/resunetpp_v4_17ch.yaml"

    rc, elapsed = run_step(
        "ResUNet++ v4-17ch  |  TRAIN  (300 ep)",
        [str(PYTHON), str(MAIN), "--config", str(cfg_res), "--mode", "train"],
        LOG_RESUNETPP,
    )
    timings["ResUNet++ v4-17ch train"] = elapsed
    if rc not in (0, 1):
        print(f"ERROR ResUNet++ train (exit {rc}). Abortando.")
        sys.exit(rc)

    rc, elapsed = run_step(
        "ResUNet++ v4-17ch  |  EVALUATE",
        [str(PYTHON), str(MAIN), "--config", str(cfg_res), "--mode", "evaluate"],
        LOG_RESUNETPP,
    )
    timings["ResUNet++ v4-17ch eval"] = elapsed

    # ------------------------------------------------------------------
    # Resumen
    # ------------------------------------------------------------------
    total = time.time() - t_global
    banner("PIPELINE v4-17ch  COMPLETADO")
    print(f"  Tiempo total: {total/3600:.2f} h\n")
    for k, v in timings.items():
        print(f"  {k:35s}: {v/60:.1f} min")
    print(f"\n  Resultados en:")
    print(f"    {ROOT / 'results/rf_v4_17ch'}")
    print(f"    {ROOT / 'results/unet_v4_17ch'}")
    print(f"    {ROOT / 'results/resunetpp_v4_17ch'}")
