"""
Pipeline nocturno: 3 experimentos en secuencia.
================================================

Experimento 1: ResUNet++ v4-ms  x3 semillas
  - Reusa dataset_v4_ms existente (no regenera)
  - Entrena y evalua con seed=1, seed=2, seed=3

Experimento 2: ResUNet++ v4-ms-Sx200
  - Genera dataset_v4_ms_sx200 (Sx_200m en escala fina 1m)
  - Entrena y evalua

Experimento 3: v6_improved eval combinado (val+test)
  - Evalua modelo v6_improved ya entrenado sobre 2024+2025
  - Calcula SPAEF con mas tiles

Uso:
    .venv\\Scripts\\python.exe scripts/run_overnight.py
"""

import subprocess
import sys
import time
from pathlib import Path

ROOT   = Path(__file__).resolve().parent.parent
PYTHON = ROOT / ".venv/Scripts/python.exe"
MAIN   = ROOT / "main.py"
LOG    = ROOT / "results/run_overnight_log.txt"


def banner(msg):
    sep = "=" * 60
    print(f"\n{sep}\n  {msg}\n{sep}", flush=True)


def run_step(label, cmd, log_path=LOG):
    log_path.parent.mkdir(parents=True, exist_ok=True)
    banner(label)
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

    # =========================================================================
    # BLOQUE 1: ResUNet++ v4-ms con 3 semillas
    # =========================================================================
    banner("BLOQUE 1/3  |  ResUNet++ v4-ms x3 semillas")
    print("  Reutilizando dataset_v4_ms existente (sin regenerar)", flush=True)

    for seed in [1, 2, 3]:
        cfg = ROOT / f"configs/resunetpp_v4_ms_s{seed}.yaml"
        lbl = f"  Seed {seed}/3  |  Train+Eval resunetpp_v4_ms_s{seed}"
        rc, el = run_step(lbl,
                          [str(PYTHON), str(MAIN),
                           "--config", str(cfg),
                           "--mode", "both",
                           "--seed", str(seed)])
        timings[f"v4_ms_s{seed}"] = el
        if rc not in (0, 1):
            errors.append(f"v4_ms_s{seed}: exit {rc}")
            print(f"  [WARN] Seed {seed} termino con exit {rc} — continuando...",
                  flush=True)

    # =========================================================================
    # BLOQUE 2: ResUNet++ v4-ms-Sx200
    # =========================================================================
    banner("BLOQUE 2/3  |  ResUNet++ v4-ms-Sx200  (genera + train + eval)")

    gen_sx200 = ROOT / "data/generate_dataset_v4_ms_sx200.py"
    cfg_sx200 = ROOT / "configs/resunetpp_v4_ms_sx200.yaml"

    rc, el = run_step("  PASO 2a  |  Generar dataset_v4_ms_sx200",
                      [str(PYTHON), str(gen_sx200)])
    timings["gen_sx200"] = el
    if rc != 0:
        errors.append(f"gen_sx200: exit {rc}")
        print(f"  [WARN] Generacion Sx200 fallo (exit {rc}) — saltando train...",
              flush=True)
    else:
        rc, el = run_step("  PASO 2b  |  Train+Eval resunetpp_v4_ms_sx200",
                          [str(PYTHON), str(MAIN),
                           "--config", str(cfg_sx200),
                           "--mode", "both"])
        timings["v4_ms_sx200"] = el
        if rc not in (0, 1):
            errors.append(f"v4_ms_sx200: exit {rc}")

    # =========================================================================
    # BLOQUE 3: Evaluacion combinada v6_improved (val+test = 2024+2025)
    # =========================================================================
    rc, el = run_step("BLOQUE 3/3  |  Eval v6_improved combinado (val+test)",
                      [str(PYTHON), str(ROOT / "baselines/evaluate_v6_combined.py")])
    timings["v6_combined_eval"] = el
    if rc != 0:
        errors.append(f"v6_combined_eval: exit {rc}")

    # =========================================================================
    # Resumen final
    # =========================================================================
    total = time.time() - t_global
    banner("PIPELINE NOCTURNO COMPLETADO")
    print(f"  Tiempo total : {total/60:.1f} min\n")
    for k, v in timings.items():
        print(f"  {k:22s}: {v/60:.1f} min")

    if errors:
        print(f"\n  ERRORES ({len(errors)}):")
        for e in errors:
            print(f"    - {e}")
    else:
        print("\n  Sin errores.")

    print(f"\n  Log completo: {LOG}")
    print(f"  Resultados en: {ROOT / 'results'}")
