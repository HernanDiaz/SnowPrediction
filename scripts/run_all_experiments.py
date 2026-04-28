"""
Lanzador maestro — ejecuta todos los experimentos en secuencia.
================================================================

Orden de ejecucion:
    1. RF v5 Optuna       (~30-40 min, CPU)
    2. RF v6 Optuna       (~30-40 min, CPU)
    3. UNet v6 Final      (~2.5 h, GPU)
    4. Attention UNet v6  (~2.5 h, GPU)
    5. ResUNet++ v6       (~3 h, GPU)

    Tiempo estimado total: ~9-10 h

Uso:
    E:\\PycharmProjects\\SnowPrediction\\.venv\\Scripts\\python.exe run_all_experiments.py

    Para lanzar solo algunos, comenta los que no quieras en la lista EXPERIMENTS.

Todos los logs se guardan en results/<experimento>/run_log.txt.
Si un experimento falla, el script muestra el error y continua con el siguiente.
"""

import subprocess
import sys
import time
from pathlib import Path
from datetime import datetime

# ---------------------------------------------------------------------------
# Configuracion
# ---------------------------------------------------------------------------
ROOT       = Path(__file__).resolve().parent.parent
PYTHON     = ROOT / ".venv/Scripts/python.exe"
SCRIPTS    = ROOT / "scripts"
MASTER_LOG = ROOT / "results/run_all_log.txt"

# Orden de ejecucion — comenta los que no quieras lanzar
EXPERIMENTS = [
    ("RF v5 Optuna",      SCRIPTS / "run_rf_v5_optuna.py"),
    ("RF v6 Optuna",      SCRIPTS / "run_rf_v6_optuna.py"),
    ("UNet v6 Final",     SCRIPTS / "run_unet_v6_final.py"),
    ("Attention UNet v6", SCRIPTS / "run_attention_unet_v6_final.py"),
    ("ResUNet++ v6",      SCRIPTS / "run_resunetpp_v6_final.py"),
]

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def banner(msg):
    sep = "=" * 60
    print(f"\n{sep}\n  {msg}\n{sep}")

def log(msg, logf=None):
    ts = datetime.now().strftime("%H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line)
    if logf:
        logf.write(line + "\n")
        logf.flush()

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    MASTER_LOG.parent.mkdir(parents=True, exist_ok=True)

    banner("LANZADOR MAESTRO DE EXPERIMENTOS")
    print(f"  Inicio   : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  Python   : {PYTHON}")
    print(f"  Log      : {MASTER_LOG}")
    print(f"\n  Experimentos programados ({len(EXPERIMENTS)}):")
    for name, script in EXPERIMENTS:
        status = "OK" if script.exists() else "FALTA EL SCRIPT"
        print(f"    [{status}]  {name}  ->  {script.name}")

    if not PYTHON.exists():
        print(f"\nERROR: entorno virtual no encontrado en {PYTHON}")
        sys.exit(1)

    results = []
    t_global = time.time()

    with open(MASTER_LOG, "a", encoding="utf-8") as logf:
        logf.write(f"\n{'='*60}\n")
        logf.write(f"INICIO: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        logf.write(f"{'='*60}\n")

        for name, script in EXPERIMENTS:
            banner(f"INICIANDO: {name}")

            if not script.exists():
                msg = f"SALTADO — script no encontrado: {script}"
                log(msg, logf)
                results.append((name, "SALTADO", 0))
                continue

            log(f"Iniciando {name}...", logf)
            t0 = time.time()

            proc = subprocess.run(
                [str(PYTHON), str(script)],
                cwd=str(ROOT),
            )

            elapsed = time.time() - t0
            rc = proc.returncode
            status = "OK" if rc in (0, 1) else f"ERROR (code {rc})"
            log(f"{name} -> {status}  |  {elapsed/60:.1f} min", logf)
            results.append((name, status, elapsed))

    # Resumen final
    total = time.time() - t_global
    banner("RESUMEN FINAL")
    print(f"  {'Experimento':<25} {'Estado':<15} {'Tiempo':>10}")
    print(f"  {'-'*52}")
    for name, status, elapsed in results:
        mins = f"{elapsed/60:.1f} min"
        print(f"  {name:<25} {status:<15} {mins:>10}")
    print(f"\n  Tiempo total: {total/60:.1f} min  ({total/3600:.1f} h)")

    # Escribir resumen en log maestro
    with open(MASTER_LOG, "a", encoding="utf-8") as logf:
        logf.write(f"\n{'='*60}\nRESUMEN\n{'='*60}\n")
        for name, status, elapsed in results:
            logf.write(f"  {name:<25} {status:<15} {elapsed/60:.1f} min\n")
        logf.write(f"\nTiempo total: {total/3600:.1f} h\n")
        logf.write(f"FIN: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
