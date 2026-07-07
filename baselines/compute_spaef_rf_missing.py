"""
Calcula SPAEF para los modelos RF que no tienen esa metrica.
============================================================

Modelos procesados:
  1. rf_v4_1m    — dataset_v4_fisico (1m, 5 canales topo)
  2. rf_v5_optuna — dataset_v5_5m    (5m, 5 canales topo)
  3. rf_v6_optuna — dataset_v6_5m    (5m, 17 canales)

En cada caso:
  - Carga el modelo .joblib ya entrenado (sin reentrenar)
  - Itera tile a tile sobre el test set preservando estructura espacial
  - Calcula SPAEF por tile y promedia
  - Actualiza el metrics.json existente añadiendo SPAEF, SPAEF_std, SPAEF_n_tiles

Uso:
    .venv\\Scripts\\python.exe baselines/compute_spaef_rf_missing.py
"""

import json
import sys
import numpy as np
import pandas as pd
import joblib
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO))
from utils.metrics import compute_spaef


# ---------------------------------------------------------------------------
# Normalizaciones (identicas a los scripts originales de entrenamiento)
# ---------------------------------------------------------------------------

def normalize_v4_5ch(X: np.ndarray) -> np.ndarray:
    """5 canales topo: DEM, Slope, Northness, Eastness, TPI."""
    X = X.copy().astype(np.float32)
    X[:, 0] = (X[:, 0] - 2100.0) / 1000.0          # DEM
    X[:, 1] = X[:, 1] / 90.0                        # Slope
    # Northness, Eastness ya en [-1, 1]
    X[:, 4] = np.clip(X[:, 4] / 9200.0, -1.0, 1.0) # TPI
    return X


def normalize_v6_17ch(img: np.ndarray) -> np.ndarray:
    """
    17 canales: indices [0..13, 30..32] de un array de 33 canales.
    Devuelve array (H*W, 17).
    """
    CHANNEL_INDICES = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 30, 31, 32]
    img = img[CHANNEL_INDICES].copy().astype(np.float32)  # (17, H, W)
    img[img == -9999] = 0
    img = np.nan_to_num(img, nan=0.0, posinf=0.0, neginf=0.0)
    img[0] = (img[0] - 2100.0) / 1000.0            # DEM
    img[1] = img[1] / 90.0                          # Slope
    # Northness, Eastness ya en [-1, 1]
    img[4] = np.clip(img[4] / 9200.0, -1.0, 1.0)   # TPI
    img[5] = (img[5] > 5).astype(np.float32)        # SCE -> binario
    img[6:14] = np.clip(img[6:14] / 90.0, -1.0, 1.0)  # Sx
    # Pers [14:17] ya en [0, 1]
    return img.reshape(17, -1).T                    # (H*W, 17)


# ---------------------------------------------------------------------------
# Funcion generica de calculo SPAEF tile a tile
# ---------------------------------------------------------------------------

def compute_spaef_for_rf(
    rf,
    test_df: pd.DataFrame,
    img_dir: Path,
    msk_dir: Path,
    n_channels: int,
    normalize_fn,
    label: str,
) -> tuple:
    """
    Itera sobre test_df tile a tile, predice con rf y calcula SPAEF espacial.

    Returns:
        (spaef_vals, n_tiles_ok) donde spaef_vals es la lista de SPAEF por tile.
    """
    spaef_vals = []
    n_missing  = 0

    for _, row in test_df.iterrows():
        img_path = img_dir / row["tile_id"]
        msk_path = msk_dir / row["tile_id"]

        if not img_path.exists() or not msk_path.exists():
            n_missing += 1
            continue

        img  = np.load(img_path)            # (C, H, W)
        mask = np.load(msk_path)            # (H, W)

        H, W = mask.shape

        # Preparar features segun el modelo
        X_tile = normalize_fn(img)          # (H*W, n_channels)

        # Limpiar valores extremos
        X_tile[X_tile == -9999] = 0.0
        X_tile = np.nan_to_num(X_tile, nan=0.0, posinf=0.0, neginf=0.0)

        y_pred = rf.predict(X_tile)         # (H*W,)
        y_true = mask.flatten()

        valid = y_true > 0.01
        if valid.sum() < 10:
            continue

        spaef_val = compute_spaef(y_true[valid], y_pred[valid])
        if not np.isnan(spaef_val):
            spaef_vals.append(spaef_val)

    if n_missing > 0:
        print(f"  [WARN] {label}: {n_missing} tiles no encontrados en disco.")

    return spaef_vals


def update_metrics_json(metrics_path: Path, spaef_vals: list, nested_key: str = None):
    """
    Añade SPAEF al JSON de metricas existente.
    Si nested_key se indica, inserta dentro de ese sub-diccionario (p.ej. 'test_metrics').
    """
    with open(metrics_path, encoding="utf-8") as f:
        data = json.load(f)

    spaef_dict = {
        "SPAEF":         round(float(np.mean(spaef_vals)), 4),
        "SPAEF_std":     round(float(np.std(spaef_vals)),  4),
        "SPAEF_n_tiles": len(spaef_vals),
    }

    if nested_key:
        data[nested_key].update(spaef_dict)
    else:
        data.update(spaef_dict)

    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)

    print(f"  Guardado: {metrics_path}")


def print_spaef(label, spaef_vals):
    if spaef_vals:
        print(f"\n  SPAEF {label}:")
        print(f"    Media   : {np.mean(spaef_vals):.4f}")
        print(f"    Std     : {np.std(spaef_vals):.4f}")
        print(f"    N tiles : {len(spaef_vals)}")
    else:
        print(f"\n  [WARN] {label}: no se pudo calcular SPAEF (sin tiles validos).")


# ---------------------------------------------------------------------------
# Modelo 1: RF v4-1m  (dataset_v4_fisico, 5 canales topo, 1m)
# ---------------------------------------------------------------------------

def run_rf_v4_1m():
    label       = "rf_v4_1m"
    root_data   = _REPO / "dataset_v4_fisico"
    root_out    = _REPO / "results/rf_v4_1m"
    model_path  = root_out / "rf_v4_1m_best.joblib"
    metrics_path= root_out / "rf_v4_1m_metrics.json"
    csv_path    = root_data / "dataset_v4_fisico.csv"

    print(f"\n{'='*60}\n  {label}\n{'='*60}")

    for p in [model_path, csv_path, root_data / "images"]:
        if not p.exists():
            print(f"  [SKIP] No encontrado: {p}")
            return

    rf      = joblib.load(model_path)
    df      = pd.read_csv(csv_path)
    test_df = df[df["exp_temporal_split"] == "test"].reset_index(drop=True)
    print(f"  Test tiles: {len(test_df)}")

    def norm_5ch(img):
        feat = img[[0, 1, 2, 3, 4]].copy().astype(np.float32)  # (5, H, W)
        feat[feat == -9999] = 0.0
        feat = np.nan_to_num(feat, nan=0.0, posinf=0.0, neginf=0.0)
        X = feat.reshape(5, -1).T   # (H*W, 5)
        return normalize_v4_5ch(X)

    spaef_vals = compute_spaef_for_rf(
        rf, test_df,
        img_dir=root_data / "images",
        msk_dir=root_data / "masks",
        n_channels=5,
        normalize_fn=norm_5ch,
        label=label,
    )

    print_spaef(label, spaef_vals)
    if spaef_vals and metrics_path.exists():
        update_metrics_json(metrics_path, spaef_vals, nested_key="test_metrics")


# ---------------------------------------------------------------------------
# Modelo 2: RF v5 Optuna  (dataset_v5_5m, 5 canales topo, 5m)
# ---------------------------------------------------------------------------

def run_rf_v5_optuna():
    label        = "rf_v5_optuna"
    root_data    = _REPO / "Articulo 1/Data/processed/dataset_v5_5m"
    root_out     = _REPO / "results/optuna_rf_v5"
    model_path   = root_out / "rf_v5_best.joblib"
    metrics_path = root_out / "rf_v5_test_metrics.json"
    # El script original reutiliza el CSV de v4
    csv_path     = root_data / "dataset_v4_fisico.csv"

    print(f"\n{'='*60}\n  {label}\n{'='*60}")

    for p in [model_path, csv_path, root_data / "images"]:
        if not p.exists():
            print(f"  [SKIP] No encontrado: {p}")
            return

    rf      = joblib.load(model_path)
    df      = pd.read_csv(csv_path)
    test_df = df[df["exp_temporal_split"] == "test"].reset_index(drop=True)
    print(f"  Test tiles: {len(test_df)}")

    def norm_v5(img):
        # v5 tiene 6 canales (topo + SCE), el RF usa solo los 5 primeros
        feat = img[:5].copy().astype(np.float32)
        feat[feat == -9999] = 0.0
        feat = np.nan_to_num(feat, nan=0.0, posinf=0.0, neginf=0.0)
        X = feat.reshape(5, -1).T   # (H*W, 5)
        return normalize_v4_5ch(X)

    spaef_vals = compute_spaef_for_rf(
        rf, test_df,
        img_dir=root_data / "images",
        msk_dir=root_data / "masks",
        n_channels=5,
        normalize_fn=norm_v5,
        label=label,
    )

    print_spaef(label, spaef_vals)
    if spaef_vals and metrics_path.exists():
        update_metrics_json(metrics_path, spaef_vals, nested_key="test_metrics")


# ---------------------------------------------------------------------------
# Modelo 3: RF v6 Optuna  (dataset_v6_5m, 17 canales, 5m)
# ---------------------------------------------------------------------------

def run_rf_v6_optuna():
    label        = "rf_v6_optuna"
    root_data    = _REPO / "Articulo 1/Data/processed/dataset_v6_5m"
    root_out     = _REPO / "results/optuna_rf_v6"
    model_path   = root_out / "rf_v6_best.joblib"
    metrics_path = root_out / "rf_v6_test_metrics.json"
    csv_path     = root_data / "dataset_v6_fisico.csv"

    print(f"\n{'='*60}\n  {label}\n{'='*60}")

    for p in [model_path, csv_path, root_data / "images"]:
        if not p.exists():
            print(f"  [SKIP] No encontrado: {p}")
            return

    rf      = joblib.load(model_path)
    df      = pd.read_csv(csv_path)
    test_df = df[df["exp_temporal_split"] == "test"].reset_index(drop=True)
    print(f"  Test tiles: {len(test_df)}")

    spaef_vals = compute_spaef_for_rf(
        rf, test_df,
        img_dir=root_data / "images",
        msk_dir=root_data / "masks",
        n_channels=17,
        normalize_fn=normalize_v6_17ch,
        label=label,
    )

    print_spaef(label, spaef_vals)
    if spaef_vals and metrics_path.exists():
        update_metrics_json(metrics_path, spaef_vals, nested_key="test_metrics")


# ---------------------------------------------------------------------------
# Modelo 4: RF v4_ms_sx200 Optuna  (dataset_v4_ms_sx200, 22 canales, 1m)
# ---------------------------------------------------------------------------

def normalize_v4_ms_sx200(img: np.ndarray) -> np.ndarray:
    """22 canales — replica SnowDataset._normalize() para dataset_v4_ms_sx200."""
    X = img[:22].copy().astype(np.float32)  # (22, H, W)
    X[X == -9999] = 0.0
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    X[0]    = (X[0] - 2100.0) / 1000.0           # DEM
    X[1]    = X[1] / 90.0                         # Slope
    # Northness (2), Eastness (3): ya en [-1, 1]
    X[4]    = np.clip(X[4] / 9200.0, -1.0, 1.0)  # TPI
    X[5]    = (X[5] > 5).astype(np.float32)       # SCE -> binario
    X[6:14] = np.clip(X[6:14] / 90.0, -1.0, 1.0) # Sx_200m x8
    # Persistencia (14-16): ya en [0, 1]
    X[17]   = (X[17] - 2100.0) / 1000.0           # DEM_5m
    X[18]   = X[18] / 90.0                         # Slope_5m
    # Northness_5m (19), Eastness_5m (20): ya en [-1, 1]
    X[21]   = np.clip(X[21] / 9200.0, -1.0, 1.0)  # TPI_5m
    return X.reshape(22, -1).T                     # (H*W, 22)


def run_rf_v4_ms_sx200_optuna():
    label        = "rf_v4_ms_sx200_optuna"
    root_data    = _REPO / "dataset_v4_ms_sx200"
    root_out     = _REPO / "results/optuna_rf_v4_ms_sx200"
    model_path   = root_out / "rf_v4_ms_sx200_best.joblib"
    metrics_path = root_out / "rf_v4_ms_sx200_metrics.json"
    csv_path     = root_data / "dataset_v4_ms_sx200.csv"

    print(f"\n{'='*60}\n  {label}\n{'='*60}")

    for p in [model_path, csv_path, root_data / "images"]:
        if not p.exists():
            print(f"  [SKIP] No encontrado: {p}")
            return

    rf      = joblib.load(model_path)
    df      = pd.read_csv(csv_path)
    test_df = df[df["exp_temporal_split"] == "test"].reset_index(drop=True)
    print(f"  Test tiles: {len(test_df)}")

    spaef_vals = compute_spaef_for_rf(
        rf, test_df,
        img_dir=root_data / "images",
        msk_dir=root_data / "masks",
        n_channels=22,
        normalize_fn=normalize_v4_ms_sx200,
        label=label,
    )

    print_spaef(label, spaef_vals)
    if spaef_vals and metrics_path.exists():
        update_metrics_json(metrics_path, spaef_vals, nested_key="test_metrics")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("Calculando SPAEF para modelos RF sin esa metrica...")

    run_rf_v4_1m()
    run_rf_v5_optuna()
    run_rf_v6_optuna()
    run_rf_v4_ms_sx200_optuna()

    print("\nListo. Revisa los metrics.json actualizados en results/")
