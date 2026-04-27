"""
Busqueda Optuna de hiperparametros para Random Forest v6 (17 canales).
========================================================================

Canales: DEM, Slope, Northness, Eastness, TPI, SCE, Sx_100m x8, Pers_15d, Pers_30d, Pers_60d

Uso:
    E:\\PycharmProjects\\SnowPrediction\\.venv\\Scripts\\python.exe run_rf_v6_optuna.py

Salidas:
    Modelo   : results/optuna_rf_v6/rf_v6_best.joblib
    Metricas : results/optuna_rf_v6/rf_v6_test_metrics.json
    Ranking  : results/optuna_rf_v6/ranking_rf_v6.json
    BD Optuna: results/optuna_rf_v6/optuna_rf_v6.db
"""

import subprocess
import sys
import time
from pathlib import Path

# ---------------------------------------------------------------------------
# Rutas
# ---------------------------------------------------------------------------
ROOT   = Path("E:/PycharmProjects/SnowPrediction")
PYTHON = ROOT / ".venv/Scripts/python.exe"
SCRIPT = ROOT / "baselines/optuna_rf_v6.py"
LOG    = ROOT / "results/optuna_rf_v6/run_log.txt"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def banner(msg):
    sep = "=" * 60
    print(f"\n{sep}\n  {msg}\n{sep}")

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    banner("Optuna RF v6  |  80 trials  |  17 canales")
    print(f"Python  : {PYTHON}")
    print(f"Script  : {SCRIPT}")
    print(f"Log     : {LOG}")

    if not PYTHON.exists():
        print(f"ERROR: no se encuentra el entorno virtual en {PYTHON}")
        sys.exit(1)
    if not SCRIPT.exists():
        print(f"ERROR: script no encontrado: {SCRIPT}")
        sys.exit(1)

    LOG.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    with open(LOG, "a", encoding="utf-8") as logf:
        logf.write(f"\n{'='*60}\nOptuna RF v6\n{'='*60}\n")
        proc = subprocess.Popen(
            [str(PYTHON), str(SCRIPT)],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            cwd=str(ROOT),
        )
        for line in proc.stdout:
            print(line, end="")
            logf.write(line)
        proc.wait()

    elapsed = time.time() - t0
    print(f"\n  Tiempo total: {elapsed/60:.1f} min  |  Exit code: {proc.returncode}")

    if proc.returncode in (0, 1):
        banner("Optuna RF v6  COMPLETADO")
        print(f"  Resultados en: {ROOT / 'results/optuna_rf_v6'}")
    else:
        print(f"ERROR: exit code {proc.returncode}")
        sys.exit(proc.returncode)
