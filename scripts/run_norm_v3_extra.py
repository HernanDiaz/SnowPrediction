"""Runner de los experimentos v3 que faltan: comparativa U-Net, split
espacial y ablacion (norm v3 + loss enmascarada).
=====================================================================

Entrena/evalua las 21 configs generadas por
generate_norm_v3_extra_configs.py. Orden por utilidad:
  1. U-Net GN (3)     -> cierra la comparativa (RF invariante, ResUNet++ del barrido)
  2. Split espacial (3) -> exp 5
  3. Ablacion (15)    -> exp 3

Guardia: salta cualquier run con metrics.json ya presente (reanudable tras
un reinicio). --force reentrena. --only {unet,spatial,ablation} filtra.
Todo aislado bajo results/norm_v3/{comparison,spatial,ablation}/.

Uso (lanzar DESACOPLADO para que sobreviva al terminal):
    Start-Process -FilePath ".venv\\Scripts\\python.exe" `
      -ArgumentList "scripts\\run_norm_v3_extra.py" `
      -RedirectStandardOutput "results\\norm_v3\\extra.out.log" `
      -RedirectStandardError  "results\\norm_v3\\extra.err.log" -WindowStyle Hidden
"""
import argparse
import json
import subprocess
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
CFG = _REPO / "configs" / "norm_v3"
SEEDS = [42, 123, 7]
ABLATION_GROUPS = ['sin_sx', 'sin_pers', 'sin_topo5', 'sin_sce', 'sin_topo1']
METRIC_KEYS = ["R2", "SPAEF", "MSPAEF", "MAE", "RMSE", "Bias"]

# (grupo, nombre_config_sin_ext, subcarpeta_resultados)
def build_jobs():
    jobs = {'unet': [], 'spatial': [], 'ablation': []}
    for s in SEEDS:
        jobs['unet'].append((f'unet_gn_v3_s{s}', 'comparison'))
    for s in SEEDS:
        jobs['spatial'].append((f'resunetpp_v3_spatial_s{s}', 'spatial'))
    for g in ABLATION_GROUPS:
        for s in SEEDS:
            jobs['ablation'].append((f'resunetpp_v3_abl_{g}_s{s}', 'ablation'))
    return jobs


def metrics_path(name, sub):
    return _REPO / "results" / "norm_v3" / sub / name / f"{name}_metrics.json"


def run_one(name, sub, log_fh):
    cfg = CFG / f"{name}.yaml"
    print(f"\n{'='*70}\n[{sub}] {name}\n{'='*70}", flush=True)
    log_fh.write(f"\n{'='*70}\n{sub}: {name}\n{'='*70}\n"); log_fh.flush()
    proc = subprocess.run(
        [sys.executable, str(_REPO / "main.py"),
         "--config", str(cfg), "--mode", "both"],
        cwd=str(_REPO), stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
    )
    print(proc.stdout, flush=True)
    log_fh.write(proc.stdout); log_fh.flush()
    return proc.returncode


def run_group(group, jobs, log_fh, force):
    for name, sub in jobs[group]:
        if metrics_path(name, sub).exists() and not force:
            msg = f"[skip] {name}: metrics.json ya existe."
            print(msg, flush=True); log_fh.write(msg + "\n")
            continue
        rc = run_one(name, sub, log_fh)
        if rc != 0:
            print(f"[error] {name} rc={rc}")
            log_fh.write(f"[error] {name} rc={rc}\n")


def summarise(jobs):
    import numpy as np

    def stats(names_subs):
        rows = {k: [] for k in METRIC_KEYS}
        for name, sub in names_subs:
            p = metrics_path(name, sub)
            if not p.exists():
                continue
            d = json.load(open(p))
            for k in METRIC_KEYS:
                if d.get(k) is not None:
                    rows[k].append(d[k])
        out = {}
        for k in METRIC_KEYS:
            if rows[k]:
                a = np.array(rows[k], float)
                out[k] = {"mean": round(float(a.mean()), 4),
                          "std": round(float(a.std(ddof=1)), 4) if len(a) > 1 else 0.0,
                          "n": len(a)}
        return out

    summary = {"norm": "v3", "masked_loss": True}

    # U-Net (3 seeds juntos)
    summary["unet_gn"] = stats(jobs['unet'])
    # Split espacial (3 seeds juntos)
    summary["spatial"] = stats(jobs['spatial'])
    # Ablacion por grupo
    summary["ablation"] = {}
    for g in ABLATION_GROUPS:
        names = [(f'resunetpp_v3_abl_{g}_s{s}', 'ablation') for s in SEEDS]
        summary["ablation"][g] = stats(names)

    out = _REPO / "results" / "norm_v3" / "extra_summary.json"
    json.dump(summary, open(out, "w"), indent=2)

    print("\n========= RESUMEN v3 EXTRA (norm v3 + masked, best-val) =========")
    for key in ["unet_gn", "spatial"]:
        e = summary[key]
        if e:
            print(f"  {key:10s} R2={e.get('R2',{}).get('mean','?')}"
                  f"+/-{e.get('R2',{}).get('std','?')}  "
                  f"SPAEF={e.get('SPAEF',{}).get('mean','?')}  "
                  f"(n={e.get('R2',{}).get('n',0)})")
    print("  ablacion:")
    for g in ABLATION_GROUPS:
        e = summary["ablation"][g]
        if e:
            print(f"    {g:10s} R2={e.get('R2',{}).get('mean','?')}"
                  f"+/-{e.get('R2',{}).get('std','?')}  "
                  f"SPAEF={e.get('SPAEF',{}).get('mean','?')}  "
                  f"(n={e.get('R2',{}).get('n',0)})")
    print(f"\nGuardado: {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--only", choices=["unet", "spatial", "ablation"],
                    action="append", help="ejecutar solo estos grupos (repetible)")
    args = ap.parse_args()

    jobs = build_jobs()
    groups = args.only if args.only else ["unet", "spatial", "ablation"]

    # el split espacial necesita el CSV espacial
    spatial_csv = _REPO / "dataset_v4_ms_sx200" / "dataset_v4_ms_sx200_spatial.csv"
    if "spatial" in groups and not spatial_csv.exists():
        sys.exit(f"ERROR: falta el CSV espacial: {spatial_csv}")

    log = _REPO / "results" / "norm_v3" / "run_norm_v3_extra_log.txt"
    log.parent.mkdir(parents=True, exist_ok=True)
    print(f"Repo: {_REPO}\nGrupos: {groups}\nLog: {log}")
    with open(log, "a", encoding="utf-8") as log_fh:
        for grp in groups:
            run_group(grp, jobs, log_fh, args.force)
    summarise(jobs)


if __name__ == "__main__":
    main()
