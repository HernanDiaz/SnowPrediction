"""Barrido lambda con normalizacion v3 — runner con orden de prioridad.
====================================================================

Entrena y evalua ResUNet++ (HPO) sobre el split temporal con norm v3
(configs/norm_v3/resunetpp_v3_sp*_s*.yaml), 8 lambdas x 3 seeds = 24 runs.

ORDEN DE PRIORIDAD: primero los 6 runs "puerta" (lambda=0.0 y lambda=0.4,
3 seeds cada uno), que responden la pregunta decisiva: sobrevive el R2
temporal con normalizacion sana y sigue lambda=0.4 > lambda=0? Tras la
puerta se imprime y guarda un resumen parcial (GATE SUMMARY) y se continua
automaticamente con el resto del barrido. Si por la manyana la puerta es
mala, basta matar el proceso.

Seguridad: se SALTA cualquier run cuyos pesos ya existan (--force para
reentrenar). Todo aislado en results/norm_v3/lambda/.

Uso (desde la raiz del repo):
    .venv/Scripts/python.exe scripts/run_norm_v3_lambda.py
    .venv/Scripts/python.exe scripts/run_norm_v3_lambda.py --force
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
OUT_ROOT = _REPO / "results" / "norm_v3" / "lambda"
OUT_ROOT.mkdir(parents=True, exist_ok=True)
WEIGHTS = OUT_ROOT / "weights"
LOG = OUT_ROOT / "run_norm_v3_lambda_log.txt"

SEEDS = [42, 123, 7]
LAMBDAS = {'0.0': '00', '0.1': '01', '0.25': '025', '0.4': '04',
           '0.5': '05', '0.6': '06', '0.75': '075', '1.0': '10'}
GATE_LAMBDAS = ['0.0', '0.4']
REST_LAMBDAS = [l for l in LAMBDAS if l not in GATE_LAMBDAS]

METRIC_KEYS = ["R2", "SPAEF", "MSPAEF", "MAE", "RMSE", "Bias"]


def name_of(lam: str, seed: int) -> str:
    return f"resunetpp_v3_sp{LAMBDAS[lam]}_s{seed}"


def already_done(lam: str, seed: int) -> bool:
    # Un run esta COMPLETO solo si existe su metrics.json (se escribe al final,
    # tras entrenar 50 epocas + evaluar). Los pesos .pth aparecen ya en la
    # epoca 1 (best-val), asi que NO sirven como senal de "terminado": si el
    # barrido se corta a media y se reanuda, un run interrumpido no tendra
    # metrics.json y se reentrenara entero (correcto).
    name = name_of(lam, seed)
    return (OUT_ROOT / name / f"{name}_metrics.json").exists()


def run_one(lam: str, seed: int, log_fh) -> int:
    name = name_of(lam, seed)
    cfg = _REPO / "configs" / "norm_v3" / f"{name}.yaml"
    print(f"\n{'='*70}\n[norm_v3] lambda={lam} seed={seed}  ->  {cfg.name}\n{'='*70}",
          flush=True)
    log_fh.write(f"\n{'='*70}\nlambda={lam} seed={seed}  {cfg.name}\n{'='*70}\n")
    log_fh.flush()
    proc = subprocess.run(
        [sys.executable, str(_REPO / "main.py"),
         "--config", str(cfg), "--mode", "both"],
        cwd=str(_REPO), stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
    )
    print(proc.stdout, flush=True)
    log_fh.write(proc.stdout); log_fh.flush()
    return proc.returncode


def summarise(lambdas, out_name, title):
    import numpy as np
    by_lambda = {}
    print(f"\n========= {title} (norm v3, best-val) =========")
    for lam in lambdas:
        rows = {k: [] for k in METRIC_KEYS}
        for s in SEEDS:
            p = OUT_ROOT / name_of(lam, s) / f"{name_of(lam, s)}_metrics.json"
            if not p.exists():
                continue
            d = json.load(open(p))
            for k in METRIC_KEYS:
                if d.get(k) is not None:
                    rows[k].append(d[k])
        entry = {}
        for k in METRIC_KEYS:
            if rows[k]:
                a = np.array(rows[k], dtype=float)
                entry[k] = {"mean": round(float(a.mean()), 4),
                            "std": round(float(a.std(ddof=1)), 4) if len(a) > 1 else 0.0,
                            "n": len(a)}
        if entry:
            by_lambda[lam] = entry
            r2 = entry.get("R2", {})
            sp = entry.get("SPAEF", {})
            print(f"  lambda={lam:<5} R2={r2.get('mean', float('nan')):+.3f}"
                  f"+/-{r2.get('std', 0):.3f}  "
                  f"SPAEF={sp.get('mean', float('nan')):+.3f}"
                  f"+/-{sp.get('std', 0):.3f}  (n={r2.get('n', 0)})")
    out = OUT_ROOT / out_name
    json.dump({"model": "resunetpp", "split": "temporal", "norm": "v3",
               "by_lambda": by_lambda}, open(out, "w"), indent=2)
    print(f"Guardado: {out}")


def run_batch(lambdas, log_fh, force):
    for lam in lambdas:
        for s in SEEDS:
            if already_done(lam, s) and not force:
                msg = (f"[skip] lambda={lam} seed={s}: pesos ya existen. "
                       f"Usa --force para reentrenar.")
                print(msg, flush=True); log_fh.write(msg + "\n")
                continue
            rc = run_one(lam, s, log_fh)
            if rc != 0:
                print(f"[error] lambda={lam} seed={s} rc={rc}; ver {LOG}")
                log_fh.write(f"[error] lambda={lam} seed={s} rc={rc}\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true",
                    help="reentrena aunque existan pesos (sobrescribe)")
    ap.add_argument("--gate-only", action="store_true",
                    help="solo los 6 runs puerta (lambda 0.0 y 0.4)")
    args = ap.parse_args()

    print(f"Repo: {_REPO}\nSalida aislada en: {OUT_ROOT}\nLog: {LOG}")
    with open(LOG, "a", encoding="utf-8") as log_fh:
        # --- Fase 1: puerta (lambda=0.0 y 0.4, 3 seeds) -------------------
        run_batch(GATE_LAMBDAS, log_fh, args.force)
        summarise(GATE_LAMBDAS, "gate_summary.json",
                  "GATE SUMMARY  lambda=0.0 vs 0.4")
        print("\n[gate] Criterio: R2(lambda=0.4) cerca de ~0.2 y > R2(lambda=0)"
              " -> la norm v3 sostiene el resultado temporal.\n", flush=True)
        if args.gate_only:
            return
        # --- Fase 2: resto del barrido ------------------------------------
        run_batch(REST_LAMBDAS, log_fh, args.force)
        summarise(list(LAMBDAS), "norm_v3_lambda_summary.json",
                  "BARRIDO LAMBDA COMPLETO")


if __name__ == "__main__":
    main()
