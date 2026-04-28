"""
Experimentos a ejecutar DESPUES de que terminen los experimentos v4_1m.
=======================================================================

Cola secuencial (sin solapar GPU):
  1. ResUNet++ v6 300ep  : mayor entrenamiento, augmentation, cosine LR
  2. UNet v6 topo5-only  : comparacion justa de resolucion (5 topo @ 5m vs 1m)

Este script puede lanzarse en background y esperara a que la GPU este libre
(verifica que los archivos de resultado de UNet v4 existan).

Uso:
    python run_post_v4_experiments.py
"""

import subprocess
import sys
import time
import os
from pathlib import Path

ROOT   = Path(__file__).resolve().parent.parent
PYTHON = ROOT / ".venv/Scripts/python.exe"
MAIN   = ROOT / "main.py"

QUEUE = [
    {
        "name":   "ResUNet++ v6 300ep",
        "config": ROOT / "configs/resunetpp_v6_300ep.yaml",
        "log":    ROOT / "results/resunetpp_v6_300ep/run_log.txt",
        "modes":  ["train", "evaluate"],
    },
    {
        "name":   "UNet v6 5-topo-only",
        "config": ROOT / "configs/unet_v6_topo5_only.yaml",
        "log":    ROOT / "results/unet_v6_topo5_only/run_log.txt",
        "modes":  ["train", "evaluate"],
    },
]

# Esperar a que UNet v4 termine
WAIT_FOR = [
    ROOT / "results/unet_v4_1m_topo5/unet_v4_1m_topo5_metrics.json",
    ROOT / "results/unet_v4_1m_sce6/unet_v4_1m_sce6_metrics.json",
]


def banner(msg):
    sep = "=" * 60
    print(f"\n{sep}\n  {msg}\n{sep}")


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
            env={**os.environ, "PYTHONUNBUFFERED": "1"},
        )
        for line in proc.stdout:
            print(line, end="")
            logf.write(line)
        proc.wait()
    elapsed = time.time() - t0
    print(f"\n  Tiempo: {elapsed/60:.1f} min  |  Exit code: {proc.returncode}")
    return proc.returncode


if __name__ == "__main__":
    # Esperar a que UNet v4 termine
    print("Esperando a que terminen los experimentos UNet v4_1m...")
    while True:
        all_done = all(f.exists() for f in WAIT_FOR)
        if all_done:
            print("UNet v4 completados. Lanzando experimentos post-v4...")
            break
        missing = [f.name for f in WAIT_FOR if not f.exists()]
        print(f"  Esperando: {missing}  (polling cada 5 min...)")
        time.sleep(300)  # check every 5 minutes

    # Ejecutar cola
    for exp in QUEUE:
        banner(f"Iniciando: {exp['name']}")
        for mode in exp["modes"]:
            rc = run_step(
                f"{exp['name']}  [{mode.upper()}]",
                [str(PYTHON), str(MAIN), "--config", str(exp["config"]), "--mode", mode],
                exp["log"],
            )
            if rc not in (0, 1):
                print(f"ERROR {mode} (exit {rc}). Abortando cola.")
                sys.exit(rc)

    banner("TODOS LOS EXPERIMENTOS POST-V4 COMPLETADOS")
