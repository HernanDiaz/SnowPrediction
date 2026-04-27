"""
Prepara y ejecuta el commit inicial del proyecto.
Uso: .venv\Scripts\python.exe scripts\_do_commit.py
"""
import os
import subprocess
import sys
from pathlib import Path

ROOT  = Path("E:/PycharmProjects/SnowPrediction")
NAME  = "HernanDiaz"
EMAIL = "hernan.diaz.rodriguez@gmail.com"

def run(cmd, cwd=ROOT):
    result = subprocess.run(cmd, cwd=str(cwd), capture_output=True, text=True)
    if result.stdout.strip():
        print(result.stdout.strip())
    if result.stderr.strip():
        print(result.stderr.strip())
    return result.returncode

# 1. Borrar script temporal
tmp = ROOT / "scripts/_organize_repo.py"
if tmp.exists():
    tmp.unlink()
    print(f"Borrado: {tmp.name}")

# 2. Verificar que es un repo git
rc = run(["git", "rev-parse", "--is-inside-work-tree"])
if rc != 0:
    print("Inicializando repositorio git...")
    run(["git", "init"])

# 3. Configurar usuario local
run(["git", "config", "user.name",  NAME])
run(["git", "config", "user.email", EMAIL])
print(f"Git user: {NAME} <{EMAIL}>")

# 4. Ver estado antes de staging
print("\n--- Estado del repo ---")
run(["git", "status", "--short"])

# 5. Añadir ficheros al staging (respetando .gitignore)
print("\n--- Staging ---")
files_to_add = [
    ".gitignore",
    "requirements.txt",
    "main.py",
    "data/",
    "models/",
    "utils/",
    "baselines/",
    "configs/",
    "scripts/",
    "results/",
]
for f in files_to_add:
    path = ROOT / f
    if path.exists():
        run(["git", "add", f])
        print(f"  + {f}")

# 6. Ver qué va a entrar en el commit
print("\n--- Ficheros en staging ---")
run(["git", "diff", "--cached", "--name-only"])

# 7. Commit
msg = (
    "Add deep learning pipeline for snow depth prediction (v6)\n\n"
    "- Dataset v6: 17 channels (topo + SCE + Sx_100m x8 + snow persistence 15/30/60d)\n"
    "- Models: UNet, Attention UNet, ResUNet++ with Optuna-tuned hyperparameters\n"
    "- Baselines: Random Forest with Optuna (v5 5-ch and v6 17-ch)\n"
    "- Scripts: standalone run scripts per experiment + master launcher\n"
    "- Configs: v6 final configs, archive of v4/v5 exploration configs\n"
    "- .gitignore: excludes venv, datasets, model weights, plots\n"
)
print("\n--- Commit ---")
rc = run(["git", "commit", "-m", msg])
if rc == 0:
    print("\nCommit realizado correctamente.")
    run(["git", "log", "--oneline", "-5"])
else:
    print("\nERROR al hacer commit (puede que no haya cambios staged).")
