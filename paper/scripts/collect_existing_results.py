"""
PAPER - Recolectar resultados ya existentes en results/.
=========================================================
Copia los JSON de metricas de experimentos ya ejecutados a paper/results/,
evitando re-entrenar innecesariamente.

Ejecutar ANTES de los runners de entrenamiento. Los runners ya comprueban
si los resultados existen en paper/results/ y se saltan el entrenamiento.

Mapeo de experimentos existentes -> destinos en paper/results/:

  Block 3 — Spatial loss (todos tienen seed=42, dataset_v4_ms_sx200):
    resunetpp_v4_ms_sx200_sp01   -> block3_spatial_loss/b3_sp_l010
    resunetpp_v4_ms_sx200_sp025  -> block3_spatial_loss/b3_sp_l025
    resunetpp_v4_ms_sx200_sp04   -> block3_spatial_loss/b3_sp_l040
    resunetpp_v4_ms_sx200_sp05   -> block3_spatial_loss/b3_sp_l050
    resunetpp_v4_ms_sx200_sp06   -> block3_spatial_loss/b3_sp_l060
    resunetpp_v4_ms_sx200_sp075  -> block3_spatial_loss/b3_sp_l075
    resunetpp_v4_ms_sx200_sp10   -> block3_spatial_loss/b3_sp_l100

Uso:
    .venv\\Scripts\\python.exe paper/scripts/collect_existing_results.py
"""

import json
import shutil
from pathlib import Path

ROOT      = Path(__file__).resolve().parent.parent.parent
PAPER_RES = ROOT / "paper/results"

# (origen_json, destino_dir, nuevo_nombre_json)
MAPPINGS = [
    # Block 3 — Spatial loss
    (
        ROOT / "results/resunetpp_v4_ms_sx200_sp01/resunetpp_v4_ms_sx200_sp01_metrics.json",
        PAPER_RES / "block3_spatial_loss/b3_sp_l010",
        "b3_sp_l010_metrics.json",
    ),
    (
        ROOT / "results/resunetpp_v4_ms_sx200_sp025/resunetpp_v4_ms_sx200_sp025_metrics.json",
        PAPER_RES / "block3_spatial_loss/b3_sp_l025",
        "b3_sp_l025_metrics.json",
    ),
    (
        ROOT / "results/resunetpp_v4_ms_sx200_sp04/resunetpp_v4_ms_sx200_sp04_metrics.json",
        PAPER_RES / "block3_spatial_loss/b3_sp_l040",
        "b3_sp_l040_metrics.json",
    ),
    (
        ROOT / "results/resunetpp_v4_ms_sx200_sp05/resunetpp_v4_ms_sx200_sp05_metrics.json",
        PAPER_RES / "block3_spatial_loss/b3_sp_l050",
        "b3_sp_l050_metrics.json",
    ),
    (
        ROOT / "results/resunetpp_v4_ms_sx200_sp06/resunetpp_v4_ms_sx200_sp06_metrics.json",
        PAPER_RES / "block3_spatial_loss/b3_sp_l060",
        "b3_sp_l060_metrics.json",
    ),
    (
        ROOT / "results/resunetpp_v4_ms_sx200_sp075/resunetpp_v4_ms_sx200_sp075_metrics.json",
        PAPER_RES / "block3_spatial_loss/b3_sp_l075",
        "b3_sp_l075_metrics.json",
    ),
    (
        ROOT / "results/resunetpp_v4_ms_sx200_sp10/resunetpp_v4_ms_sx200_sp10_metrics.json",
        PAPER_RES / "block3_spatial_loss/b3_sp_l100",
        "b3_sp_l100_metrics.json",
    ),
]


def collect():
    copied  = []
    missing = []
    skipped = []

    for src, dst_dir, dst_name in MAPPINGS:
        dst = dst_dir / dst_name

        if dst.exists():
            skipped.append(dst_name)
            continue

        if not src.exists():
            missing.append(str(src.relative_to(ROOT)))
            continue

        dst_dir.mkdir(parents=True, exist_ok=True)

        # Cargar JSON original, añadir nota de procedencia y guardar
        with open(src, encoding="utf-8") as f:
            data = json.load(f)

        data["_paper_note"] = (
            f"Copied from {src.relative_to(ROOT).as_posix()} — "
            "same experiment, same hyperparams and seed=42."
        )

        with open(dst, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

        print(f"  [OK] {src.name}")
        print(f"       -> {dst.relative_to(ROOT)}")
        copied.append(dst_name)

    print(f"\nResumen:")
    print(f"  Copiados : {len(copied)}")
    print(f"  Ya exist.: {len(skipped)}")
    print(f"  Faltantes: {len(missing)}")
    if missing:
        print("  Archivos no encontrados:")
        for m in missing:
            print(f"    {m}")


if __name__ == "__main__":
    print("=" * 60)
    print("  Recolectando resultados existentes -> paper/results/")
    print("=" * 60 + "\n")
    collect()
    print("\nListo. Ahora puedes ejecutar run_all_paper.py --from block1")
    print("Los experimentos ya copiados se saltaran automaticamente.")
