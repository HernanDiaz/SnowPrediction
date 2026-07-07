"""
PAPER - Lanzador completo de todos los experimentos del paper.
==============================================================
Ejecuta en orden todos los bloques de experimentos necesarios para reproducir
los resultados del paper. Cada bloque puede ejecutarse independientemente.

Orden de ejecucion:
    1. Block 1 RF baseline       (run_block1_rf.py)
    2. Block 1 CNN               (run_block1.py)
    3. Block 3 channel ablation  (run_block3_channels.py)
    4. Block 3 spatial loss      (run_block3_spatial_loss.py)
    5. Block 3 seed sensitivity  (run_block3_seeds.py)
    6. Compile tables            (compile_tables.py)

Tiempo estimado total: ~20-30 h (GPU).
    Block 1 RF   : ~2-3 h (CPU, sin GPU)
    Block 1 CNNs : ~4-6 h (2 modelos x 300 ep)
    Block 3 ch   : ~8-12 h (3 nuevos modelos x 300 ep)
    Block 3 sp   : ~16-24 h (7 lambdas x 300 ep)
    Block 3 seeds: ~12-18 h (3 semillas x 300 ep)

Uso:
    .venv\\Scripts\\python.exe paper/scripts/run_all_paper.py

    # Para ejecutar solo desde block3 (si block1 ya esta listo):
    .venv\\Scripts\\python.exe paper/scripts/run_all_paper.py --from block3
"""

import argparse
import subprocess
import sys
import time
from pathlib import Path

ROOT   = Path(__file__).resolve().parent.parent.parent
PYTHON = ROOT / ".venv/Scripts/python.exe"
SCRIPTS = ROOT / "paper/scripts"


def banner(msg):
    sep = "=" * 70
    print(f"\n{sep}\n  {msg}\n{sep}", flush=True)


def run_script(script_name):
    script = SCRIPTS / script_name
    if not script.exists():
        print(f"  [SKIP] Script no encontrado: {script}", flush=True)
        return None, 0.0

    banner(f"Ejecutando: {script_name}")
    t0 = time.time()
    proc = subprocess.run(
        [str(PYTHON), str(script)],
        env={**__import__("os").environ,
             "PYTHONUNBUFFERED": "1",
             "PYTHONIOENCODING": "utf-8"},
    )
    elapsed = time.time() - t0
    print(f"\n  {script_name} completado en {elapsed/60:.1f} min  |  exit={proc.returncode}",
          flush=True)
    return proc.returncode, elapsed


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--from", dest="from_block", default="block1",
                        choices=["block1", "block3"],
                        help="Empezar desde este bloque (default: block1)")
    args = parser.parse_args()

    if not PYTHON.exists():
        print(f"ERROR: venv no encontrado: {PYTHON}")
        sys.exit(1)

    t_global = time.time()
    timings = {}

    banner("PAPER - Reproduccion completa de experimentos")

    # Paso 0: copiar resultados ya existentes (evita re-entrenar lo que ya esta hecho)
    banner("Paso 0: Recolectar resultados existentes")
    run_script("collect_existing_results.py")

    steps = []
    if args.from_block == "block1":
        steps += [
            ("run_block1.py", "Block 1: todos los experimentos (DEM + RF + UNet + ResUNet++)"),
        ]
    steps += [
        ("run_block3_channels.py",     "Block 3: Channel ablation"),
        ("run_block3_spatial_loss.py", "Block 3: Spatial loss sweep"),
        ("run_block3_seeds.py",        "Block 3: Seed sensitivity"),
        ("compile_tables.py",          "Compilar tablas CSV"),
    ]

    for script_name, desc in steps:
        banner(desc)
        rc, elapsed = run_script(script_name)
        timings[script_name] = elapsed
        if rc is not None and rc not in (0, 1):
            print(f"\n  [ERROR] {script_name} termino con exit={rc}. Abortando.", flush=True)
            sys.exit(rc)

    total = time.time() - t_global
    banner("TODOS LOS EXPERIMENTOS COMPLETADOS")
    print(f"  Tiempo total: {total/3600:.1f} h\n")
    for k, v in timings.items():
        print(f"  {k:<40s}: {v/60:.1f} min")
    print("\n  Tablas en: paper/results/tables/")
