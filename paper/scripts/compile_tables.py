"""
PAPER - Compilar tablas de resultados para el paper.
=====================================================
Lee todos los JSON de metricas de paper/results/ y genera tablas CSV
en paper/results/tables/ listas para importar en el paper.

Tablas generadas:
    table1_block1.csv          : Comparacion arquitecturas (RF, UNet, ResUNet++)
    table2_block3_channels.csv : Ablacion de canales
    table3_block3_spatial.csv  : Barrido de lambda (Pareto front)
    table4_block3_seeds.csv    : Sensibilidad a semilla

Uso:
    .venv\\Scripts\\python.exe paper/scripts/compile_tables.py
"""

import json
import csv
from pathlib import Path

ROOT       = Path(__file__).resolve().parent.parent.parent
PAPER_RES  = ROOT / "paper/results"
TABLES_DIR = PAPER_RES / "tables"
TABLES_DIR.mkdir(parents=True, exist_ok=True)

# Metricas a exportar (en orden de columna)
METRICS = ["R2", "RMSE", "MAE", "Bias", "SPAEF"]


def load_metrics(json_path: Path) -> dict:
    """Carga test_metrics de un JSON de resultados. Devuelve {} si no existe."""
    if not json_path.exists():
        return {}
    with open(json_path, encoding="utf-8") as f:
        data = json.load(f)
    # Soporte para distintos formatos de clave
    return data.get("test_metrics", data.get("metrics", {}))


def row(label, metrics: dict) -> dict:
    r = {"model": label}
    for m in METRICS:
        val = metrics.get(m, metrics.get(m.lower(), ""))
        r[m] = f"{val:.4f}" if isinstance(val, float) else (val if val != "" else "n/a")
    return r


def write_csv(path: Path, rows: list, fieldnames=None):
    if not rows:
        print(f"  [SKIP] Sin datos para: {path.name}")
        return
    if fieldnames is None:
        fieldnames = list(rows[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)
    print(f"  Guardada: {path}")


# ---------------------------------------------------------------------------
# Table 1 — Block 1: Architecture comparison
# ---------------------------------------------------------------------------
def make_table1():
    entries = [
        ("DEM regression", PAPER_RES / "block1/b1_dem_regression/b1_dem_regression_metrics.json"),
        ("RF (22ch)",      PAPER_RES / "block1/b1_rf/b1_rf_metrics.json"),
        ("U-Net (22ch)",   PAPER_RES / "block1/b1_unet/b1_unet_metrics.json"),
        ("ResUNet++ (22ch)", PAPER_RES / "block1/b1_resunetpp/b1_resunetpp_metrics.json"),
    ]
    rows = []
    for label, path in entries:
        m = load_metrics(path)
        if not m:
            print(f"  [MISSING] {path.name}")
        rows.append(row(label, m))
    write_csv(TABLES_DIR / "table1_block1.csv", rows)


# ---------------------------------------------------------------------------
# Table 2 — Block 3: Channel ablation
# ---------------------------------------------------------------------------
def make_table2():
    entries = [
        ("ResUNet++ 5ch (topo)",         PAPER_RES / "block3_channels/b3_ch_5ch/b3_ch_5ch_metrics.json"),
        ("ResUNet++ 17ch (no ms)",       PAPER_RES / "block3_channels/b3_ch_17ch/b3_ch_17ch_metrics.json"),
        ("ResUNet++ 22ch Sx100m",        PAPER_RES / "block3_channels/b3_ch_22ch_sx100/b3_ch_22ch_sx100_metrics.json"),
        ("ResUNet++ 22ch Sx200m (ref)",  PAPER_RES / "block3_channels/b3_ch_22ch_sx200/b3_ch_22ch_sx200_metrics.json"),
    ]
    rows = []
    for label, path in entries:
        m = load_metrics(path)
        if not m:
            # Try block1 resunetpp for sx200 (same experiment)
            if "sx200" in str(path):
                alt = PAPER_RES / "block1/b1_resunetpp/b1_resunetpp_metrics.json"
                m = load_metrics(alt)
                if m:
                    print(f"  [REUSE] {label} <- b1_resunetpp_metrics.json")
            if not m:
                print(f"  [MISSING] {path.name}")
        rows.append(row(label, m))
    write_csv(TABLES_DIR / "table2_block3_channels.csv", rows)


# ---------------------------------------------------------------------------
# Table 3 — Block 3: Spatial loss lambda sweep
# ---------------------------------------------------------------------------
def make_table3():
    lambdas = [
        ("0.00", "b3_sp_l000"),
        ("0.10", "b3_sp_l010"),
        ("0.25", "b3_sp_l025"),
        ("0.40", "b3_sp_l040"),
        ("0.50", "b3_sp_l050"),
        ("0.60", "b3_sp_l060"),
        ("0.75", "b3_sp_l075"),
        ("1.00", "b3_sp_l100"),
    ]
    rows = []
    for lam, exp in lambdas:
        path = PAPER_RES / f"block3_spatial_loss/{exp}/{exp}_metrics.json"
        m = load_metrics(path)
        if not m and lam == "0.00":
            # l000 == b1_resunetpp
            alt = PAPER_RES / "block1/b1_resunetpp/b1_resunetpp_metrics.json"
            m = load_metrics(alt)
            if m:
                print(f"  [REUSE] lambda=0.00 <- b1_resunetpp_metrics.json")
        if not m:
            print(f"  [MISSING] {path.name}")
        r = {"lambda": lam}
        for met in METRICS:
            val = m.get(met, m.get(met.lower(), ""))
            r[met] = f"{val:.4f}" if isinstance(val, float) else (val if val != "" else "n/a")
        rows.append(r)
    write_csv(TABLES_DIR / "table3_block3_spatial.csv", rows,
              fieldnames=["lambda"] + METRICS)


# ---------------------------------------------------------------------------
# Table 4 — Block 3: Seed sensitivity
# ---------------------------------------------------------------------------
def make_table4():
    entries = [
        ("seed=42 (ref)", PAPER_RES / "block1/b1_resunetpp/b1_resunetpp_metrics.json"),
        ("seed=1",        PAPER_RES / "block3_seeds/b3_seed1/b3_seed1_metrics.json"),
        ("seed=2",        PAPER_RES / "block3_seeds/b3_seed2/b3_seed2_metrics.json"),
        ("seed=3",        PAPER_RES / "block3_seeds/b3_seed3/b3_seed3_metrics.json"),
    ]
    rows = []
    for label, path in entries:
        m = load_metrics(path)
        if not m:
            print(f"  [MISSING] {path.name}")
        rows.append(row(label, m))

    # Compute mean ± std for each metric across seeds 1,2,3,42
    import statistics
    summary_row = {"model": "mean ± std"}
    seed_rows = [r for r in rows]
    for met in METRICS:
        vals = []
        for r in seed_rows:
            v = r.get(met, "n/a")
            try:
                vals.append(float(v))
            except (ValueError, TypeError):
                pass
        if len(vals) >= 2:
            summary_row[met] = f"{statistics.mean(vals):.4f} ± {statistics.stdev(vals):.4f}"
        else:
            summary_row[met] = "n/a"
    rows.append(summary_row)

    write_csv(TABLES_DIR / "table4_block3_seeds.csv", rows)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("  Compilando tablas de resultados del paper")
    print(f"  Salida: {TABLES_DIR}")
    print("=" * 60 + "\n")

    print("Table 1 — Block 1: Architecture comparison")
    make_table1()

    print("\nTable 2 — Block 3: Channel ablation")
    make_table2()

    print("\nTable 3 — Block 3: Spatial loss lambda sweep")
    make_table3()

    print("\nTable 4 — Block 3: Seed sensitivity")
    make_table4()

    print("\nDone. Revisa paper/results/tables/ para los CSV.")
