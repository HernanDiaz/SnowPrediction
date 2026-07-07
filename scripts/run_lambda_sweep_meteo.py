"""
Barrido de lambda para ResUNet++ meteo1 (26ch).
================================================
HPO params Trial 1: base=48, lr=4.653e-5, adamw, bs=8, dropout=0.12, wd=1.055e-5, gc=1.0
Dataset: dataset_v4_ms_sx200_meteo (26ch)
Lambdas: 0.0, 0.1, 0.25, 0.4, 0.5, 0.6, 0.75, 1.0
Seeds:   42, 123, 7
Total:   24 experimentos x 50 epocas

Resultados en: results/lambda_sweep_meteo/

Uso:
    .venv/Scripts/python.exe scripts/run_lambda_sweep_meteo.py
"""

import subprocess
import sys
from pathlib import Path

ROOT   = Path(__file__).resolve().parent.parent
PYTHON = ROOT / ".venv/Scripts/python.exe"
MAIN   = ROOT / "main.py"

LAMBDAS = ["sp00", "sp01", "sp025", "sp04", "sp05", "sp06", "sp075", "sp10"]
SEEDS   = [42, 123, 7]

CONFIGS = [
    f"configs/resunetpp_meteo_{lam}_s{seed}.yaml"
    for lam in LAMBDAS
    for seed in SEEDS
]

if __name__ == "__main__":
    print(f"Barrido lambda meteo1 | {len(CONFIGS)} experimentos")
    print(f"Lambdas: {LAMBDAS}")
    print(f"Seeds:   {SEEDS}\n")

    for i, config in enumerate(CONFIGS, 1):
        # Derivar nombre del experimento del config (e.g. resunetpp_meteo_sp04_s42)
        exp_name = Path(config).stem
        metrics_file = ROOT / "results" / "lambda_sweep_meteo" / exp_name / f"{exp_name}_metrics.json"
        if metrics_file.exists():
            print(f"  [{i}/{len(CONFIGS)}] SKIP (ya existe): {exp_name}")
            continue

        print(f"\n{'='*60}")
        print(f"  [{i}/{len(CONFIGS)}] {config}")
        print(f"{'='*60}\n")
        ret = subprocess.run(
            [str(PYTHON), str(MAIN), "--config", config, "--mode", "both"],
            cwd=str(ROOT),
        )
        if ret.returncode != 0:
            print(f"\n[ERROR] Fallo en {config} (codigo {ret.returncode}). Abortando.")
            sys.exit(ret.returncode)

    print("\n\nBarrido lambda meteo1 COMPLETADO.")
    print("Resultados en results/lambda_sweep_meteo/")
