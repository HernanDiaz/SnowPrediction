"""
Reorganizacion de ficheros sueltos en la raiz del proyecto.
Uso: .venv\Scripts\python.exe scripts\_reorganize.py
"""
import shutil
import subprocess
from pathlib import Path

ROOT = Path("E:/PycharmProjects/SnowPrediction")

# ---------------------------------------------------------------------------
# Crear carpetas destino
# ---------------------------------------------------------------------------
(ROOT / "scripts/optuna").mkdir(parents=True, exist_ok=True)

moves = [
    # (origen, destino)
    # CRITICO: generacion del dataset v6
    (ROOT / "generate_dataset_v6.py",          ROOT / "data/generate_dataset_v6.py"),
    # Scripts Optuna CNN v6
    (ROOT / "optuna_v6_unet.py",               ROOT / "scripts/optuna/optuna_v6_unet.py"),
    (ROOT / "optuna_v6_attention_unet.py",     ROOT / "scripts/optuna/optuna_v6_attention_unet.py"),
    (ROOT / "optuna_v6_resunetpp.py",          ROOT / "scripts/optuna/optuna_v6_resunetpp.py"),
    # Scripts Optuna v5 (para replicabilidad)
    (ROOT / "optuna_unet.py",                  ROOT / "scripts/optuna/optuna_unet_v5.py"),
    (ROOT / "optuna_resunetpp.py",             ROOT / "scripts/optuna/optuna_resunetpp_v5.py"),
    (ROOT / "optuna_search.py",                ROOT / "scripts/optuna/optuna_search.py"),
    # Utilidad para compilar resultados
    (ROOT / "compile_results.py",              ROOT / "scripts/compile_results.py"),
]

to_delete = [
    ROOT / "run_v6_all.bat",
    ROOT / "run_v6_train.bat",
    ROOT / "run_v6_evaluate.bat",
    ROOT / "run_final.py",
    ROOT / "ensemble_evaluate.py",
    ROOT / "ensemble_evaluate4.py",
    ROOT / "check_optuna.py",
    ROOT / "scripts/_do_commit.py",
]

print("=== MOVIENDO FICHEROS ===")
for src, dst in moves:
    if src.exists():
        shutil.move(str(src), str(dst))
        print(f"  {src.name} -> {dst.relative_to(ROOT)}")
    else:
        print(f"  [NO ENCONTRADO] {src.name}")

print("\n=== ELIMINANDO FICHEROS ===")
for f in to_delete:
    if f.exists():
        f.unlink()
        print(f"  Borrado: {f.name}")
    else:
        print(f"  [YA NO EXISTE] {f.name}")

print("\n=== ACTUALIZANDO GIT ===")
# git add selectivo — solo codigo, nunca datos ni modelos
folders_to_add = [
    ".gitignore",
    "data/",
    "models/",
    "training/",
    "utils/",
    "baselines/",
    "configs/",
    "scripts/",
    "main.py",
    "requirements.txt",
]
for f in folders_to_add:
    r = subprocess.run(["git", "add", f], cwd=str(ROOT), capture_output=True, text=True)
    if r.returncode == 0:
        print(f"  staged: {f}")
    else:
        print(f"  [warn] {f}: {r.stderr.strip()}")

# Marcar los ficheros eliminados
subprocess.run(["git", "add", "-u"], cwd=str(ROOT), capture_output=True, text=True)
print("  staged: eliminaciones")

result = subprocess.run(
    ["git", "commit", "-m",
     "Reorganize project structure for reproducibility\n\n"
     "- Move generate_dataset_v6.py to data/ (critical for reproducibility)\n"
     "- Move Optuna CNN scripts to scripts/optuna/\n"
     "- Move compile_results.py to scripts/\n"
     "- Remove superseded .bat files and auxiliary scripts\n"
     "- Remove temporary _do_commit.py helper\n"
     "- Update .gitignore: exclude all of Articulo 1/ and data file types (.tif, .npy, .pth)"],
    cwd=str(ROOT),
    capture_output=True, text=True
)
print(result.stdout.strip())
if result.stderr.strip():
    print(result.stderr.strip())

print("\n=== ESTRUCTURA FINAL ===")
for folder in ["data", "models", "training", "utils", "baselines", "configs", "scripts"]:
    p = ROOT / folder
    if p.exists():
        files = [f.relative_to(ROOT) for f in sorted(p.rglob("*.py")) if ".venv" not in str(f)]
        for f in files:
            print(f"  {f}")
        yamls = [f.name for f in sorted(p.glob("*.yaml"))]
        for y in yamls:
            print(f"  {folder}/{y}")
