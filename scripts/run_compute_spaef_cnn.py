"""
Re-evaluacion de experimentos CNN para calcular SPAEF.
======================================================

Ejecuta --mode evaluate sobre los 11 experimentos CNN que no tienen SPAEF
en su metrics.json. No hay reentrenamiento: carga los pesos .pth existentes
y re-genera el metrics.json incluyendo SPAEF, SPAEF_std y SPAEF_n_tiles.

Uso:
    .venv\\Scripts\\python.exe scripts/run_compute_spaef_cnn.py

Tiempo estimado: ~5-15 min total (la evaluacion es mucho mas rapida que el
entrenamiento; los modelos 1m con 220 tiles de test tardan mas que los 5m).
"""

import subprocess
import sys
import time
from pathlib import Path

ROOT   = Path(__file__).resolve().parent.parent
PYTHON = ROOT / ".venv/Scripts/python.exe"
MAIN   = ROOT / "main.py"
LOG    = ROOT / "results/run_compute_spaef_cnn_log.txt"

# Experimentos CNN sin SPAEF, ordenados de menor a mayor coste computacional
EXPERIMENTS = [
    # --- Dataset v6 5m (rapidos: pocos tiles de test) ---
    "unet_v6_5m",
    "attention_unet_v6_5m",
    "resunetpp_v6_5m",
    "unet_v6_final",
    "attention_unet_v6_final",
    "resunetpp_v6_final",
    "resunetpp_v6_improved",
    # --- Dataset v4 2.5m ---
    "unet_v4_2p5m",
    "resunetpp_v4_2p5m",
    # --- Dataset v4 1m (lentos: 220 tiles de 256x256 a 1m) ---
    "unet_v4_1m_topo5",
    "unet_v4_1m_sce6",
]


def banner(msg):
    sep = "=" * 60
    print(f"\n{sep}\n  {msg}\n{sep}", flush=True)


def run_eval(exp_name, log_path=LOG):
    cfg = ROOT / f"configs/{exp_name}.yaml"
    if not cfg.exists():
        print(f"  [SKIP] Config no encontrado: {cfg}", flush=True)
        return None, 0.0

    log_path.parent.mkdir(parents=True, exist_ok=True)
    label = f"Evaluate: {exp_name}"
    banner(label)

    t0 = time.time()
    with open(log_path, "a", encoding="utf-8") as logf:
        logf.write(f"\n{'='*60}\n{label}\n{'='*60}\n")
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
    skipped  = []

    banner(f"SPAEF CNN  |  {len(EXPERIMENTS)} experimentos")
    print(f"  Log: {LOG}\n", flush=True)

    for exp in EXPERIMENTS:
        rc, elapsed = run_eval(exp)
        if rc is None:
            skipped.append(exp)
        elif rc not in (0, 1):
            errors.append(f"{exp}: exit {rc}")
            print(f"  [WARN] {exp} termino con exit {rc} — continuando...", flush=True)
        timings[exp] = elapsed

    total = time.time() - t_global
    banner("COMPLETADO")
    print(f"  Tiempo total: {total/60:.1f} min\n")
    for k, v in timings.items():
        print(f"  {k:<35s}: {v/60:.1f} min")

    if skipped:
        print(f"\n  SALTADOS (config no encontrado): {skipped}")
    if errors:
        print(f"\n  ERRORES ({len(errors)}):")
        for e in errors:
            print(f"    - {e}")
    else:
        print("\n  Sin errores.")

    print(f"\n  Log completo: {LOG}")
