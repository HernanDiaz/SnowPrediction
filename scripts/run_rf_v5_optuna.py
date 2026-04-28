"""
Busqueda Optuna de hiperparametros para Random Forest v5 (5 canales topograficos).
====================================================================================

Uso:
    E:\\PycharmProjects\\SnowPrediction\\.venv\\Scripts\\python.exe run_rf_v5_optuna.py

Salidas:
    Modelo   : results/optuna_rf_v5/rf_v5_best.joblib
    Metricas : results/optuna_rf_v5/rf_v5_test_metrics.json
    Ranking  : results/optuna_rf_v5/ranking_rf_v5.json
    BD Optuna: results/optuna_rf_v5/optuna_rf_v5.db
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
SCRIPT = ROOT / "baselines/optuna_rf_v5.py"
LOG    = ROOT / "results/optuna_rf_v5/run_log.txt"

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
    banner("Optuna RF v5  |  80 trials  |  5 canales topograficos")
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
        logf.write(f"\n{'='*60}\nOptuna RF v5\n{'='*60}\n")
        proc = subprocess.Popen(
            [str(PYTHON), "-u", str(SCRIPT)],  # -u: unbuffered output
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            env={**__import__("os").environ, "PYTHONUNBUFFERED": "1"},
            cwd=str(ROOT),
        )
        for line in proc.stdout:
            print(line, end="")
            logf.write(line)
        proc.wait()

    elapsed = time.time() - t0
    print(f"\n  Tiempo total: {elapsed/60:.1f} min  |  Exit code: {proc.returncode}")

    if proc.returncode in (0, 1):
        banner("Optuna RF v5  COMPLETADO")
        print(f"  Resultados en: {ROOT / 'results/optuna_rf_v5'}")
    else:
        print(f"ERROR: exit code {proc.returncode}")
        sys.exit(proc.returncode)
