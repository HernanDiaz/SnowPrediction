"""
Random Forest en dataset v4_17ch (1m de resolucion, 17 canales).
================================================================

Comparacion directa con RF v6 (mismos canales, diferente resolucion):
    RF v6 (5m, 17ch):  R2=0.2570  RMSE=0.636
    RF v4 (1m, 17ch):  ???

Canales (17):
    [0]  DEM           - Elevacion (metros)
    [1]  Slope         - Pendiente (grados)
    [2]  Northness     - cos(aspect) [-1, 1]
    [3]  Eastness      - sin(aspect) [-1, 1]
    [4]  TPI           - Topographic Position Index
    [5]  SCE           - Snow Cover Extent (codigos 0/10/11)
    [6]  Sx_100m_0     - Wind Shelter Index, dir 0 (N)
    [7]  Sx_100m_45
    [8]  Sx_100m_90
    [9]  Sx_100m_135
    [10] Sx_100m_180
    [11] Sx_100m_225
    [12] Sx_100m_270
    [13] Sx_100m_315
    [14] Persistencia_15d - fraccion dias con nieve en ventana de 15d previos
    [15] Persistencia_30d
    [16] Persistencia_60d

Normalizacion identica a SnowDataset._normalize() en data/dataset.py.
Split temporal: train=2021-2023, val=2024, test=2025.

Salidas:
    Modelo  : results/rf_v4_17ch/rf_v4_17ch_best.joblib
    Metricas: results/rf_v4_17ch/rf_v4_17ch_metrics.json
"""

import json
import time
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import joblib

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
_REPO     = Path(__file__).resolve().parent.parent
ROOT_DATA = _REPO / "dataset_v4_17ch"
ROOT_OUT  = _REPO / "results/rf_v4_17ch"
ROOT_OUT.mkdir(parents=True, exist_ok=True)

CSV     = ROOT_DATA / "dataset_v4_17ch.csv"
IMG_DIR = ROOT_DATA / "images"
MSK_DIR = ROOT_DATA / "masks"

CHANNEL_NAMES = [
    "DEM", "Slope", "Northness", "Eastness", "TPI", "SCE",
    "Sx_0", "Sx_45", "Sx_90", "Sx_135", "Sx_180", "Sx_225", "Sx_270", "Sx_315",
    "Pers_15d", "Pers_30d", "Pers_60d",
]
CHANNEL_IDX = list(range(17))

# Normalizacion (replica SnowDataset._normalize en data/dataset.py)
DEM_MEAN  = 2100.0
DEM_STD   = 1000.0
SLOPE_MAX =   90.0
TPI_MAX   = 9200.0
SX_MAX    =   90.0


def normalize(X: np.ndarray) -> np.ndarray:
    """
    Normaliza columnas en el mismo orden y escala que SnowDataset._normalize().
    X: (N, 17)
    """
    X = X.copy().astype(np.float32)
    X[:, 0] = (X[:, 0] - DEM_MEAN) / DEM_STD                     # DEM
    X[:, 1] = X[:, 1] / SLOPE_MAX                                 # Slope
    # Northness (2) y Eastness (3) ya en [-1, 1]
    X[:, 4] = np.clip(X[:, 4] / TPI_MAX, -1.0, 1.0)              # TPI
    X[:, 5] = (X[:, 5] > 5).astype(np.float32)                   # SCE -> binario
    X[:, 6:14] = np.clip(X[:, 6:14] / SX_MAX, -1.0, 1.0)        # Sx x8
    # Persistencia (14-16) ya en [0, 1]
    return X


def load_split_pixels(df: pd.DataFrame):
    """
    Carga todos los tiles del split y devuelve X (N, 17), y (N,).
    Solo incluye pixels con mascara valida (> -100).
    """
    X_list, y_list = [], []
    n_tiles = len(df)

    for i, row in enumerate(df.itertuples(), 1):
        if i % 500 == 0:
            print(f"  Cargando tile {i}/{n_tiles}...", flush=True)

        img_path = IMG_DIR / row.tile_id
        msk_path = MSK_DIR / row.tile_id

        try:
            img  = np.load(img_path).astype(np.float32)   # (17, 256, 256)
            mask = np.load(msk_path).astype(np.float32)   # (256, 256)
        except Exception as e:
            print(f"  Error cargando {row.tile_id}: {e}")
            continue

        features = img[CHANNEL_IDX, :, :]  # (17, 256, 256)

        # Mascara valida: valores de profundidad de nieve >= -100
        valid = mask > -100
        if valid.sum() == 0:
            continue

        X = features[:, valid].T   # (n_valid, 17)
        y = mask[valid]            # (n_valid,)

        # Limpiar nodata residual
        X[X == -9999] = 0.0
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

        X_list.append(X)
        y_list.append(y)

    if not X_list:
        raise RuntimeError("No se cargaron pixeles validos. Verifica el dataset.")

    return np.vstack(X_list), np.concatenate(y_list)


def compute_metrics(y_true, y_pred):
    return {
        "MAE":  round(float(mean_absolute_error(y_true, y_pred)), 4),
        "RMSE": round(float(np.sqrt(mean_squared_error(y_true, y_pred))), 4),
        "R2":   round(float(r2_score(y_true, y_pred)), 4),
        "NSE":  round(float(r2_score(y_true, y_pred)), 4),
        "Bias": round(float(np.mean(y_pred - y_true)), 4),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("=" * 60)
    print("  RF v4-17ch | 17 canales | split temporal")
    print("=" * 60)

    df = pd.read_csv(CSV)
    train_df = df[df["exp_temporal_split"] == "train"].reset_index(drop=True)
    val_df   = df[df["exp_temporal_split"] == "val"].reset_index(drop=True)
    test_df  = df[df["exp_temporal_split"] == "test"].reset_index(drop=True)

    print(f"\nSplit: train={len(train_df)}  val={len(val_df)}  test={len(test_df)}")

    # --- Cargar train ---
    print("\nCargando train...", flush=True)
    t0 = time.time()
    X_train, y_train = load_split_pixels(train_df)
    print(f"  Train pixels: {len(y_train):,}  ({(time.time()-t0)/60:.1f} min)")

    print("Normalizando...", flush=True)
    X_train = normalize(X_train)

    # Submuestrear si es necesario (RF no escala bien >5M pixeles)
    MAX_PIXELS = 2_000_000
    if len(y_train) > MAX_PIXELS:
        rng = np.random.RandomState(42)
        idx = rng.choice(len(y_train), MAX_PIXELS, replace=False)
        X_train = X_train[idx]
        y_train = y_train[idx]
        print(f"  Submuestreado a {MAX_PIXELS:,} pixels")

    # --- Entrenar RF (mismos hiperparametros que RF v5/v6 mejor trial) ---
    print("\nEntrenando RF...", flush=True)
    rf = RandomForestRegressor(
        n_estimators=500,
        max_depth=20,
        min_samples_leaf=1,
        max_features=0.3,
        min_samples_split=5,
        n_jobs=-1,
        random_state=42,
    )
    t0 = time.time()
    rf.fit(X_train, y_train)
    print(f"  Entrenamiento: {(time.time()-t0)/60:.1f} min")

    # --- Evaluacion en validation ---
    print("\nCargando val...", flush=True)
    X_val, y_val = load_split_pixels(val_df)
    X_val = normalize(X_val)
    y_val_pred = rf.predict(X_val)
    val_metrics = compute_metrics(y_val, y_val_pred)
    print(f"  Val  R2={val_metrics['R2']:.4f}  RMSE={val_metrics['RMSE']:.4f}")

    # --- Evaluacion en test ---
    print("\nCargando test...", flush=True)
    X_test, y_test = load_split_pixels(test_df)
    X_test = normalize(X_test)
    y_test_pred = rf.predict(X_test)
    test_metrics = compute_metrics(y_test, y_test_pred)
    print(f"\n  Test R2  : {test_metrics['R2']:.4f}")
    print(f"  Test RMSE: {test_metrics['RMSE']:.4f}")
    print(f"  Test MAE : {test_metrics['MAE']:.4f}")
    print(f"  Test Bias: {test_metrics['Bias']:.4f}")

    # --- Feature importance ---
    fi = dict(zip(CHANNEL_NAMES, rf.feature_importances_.tolist()))
    print("\nFeature importance:")
    for k, v in sorted(fi.items(), key=lambda x: -x[1]):
        print(f"  {k:15s}: {v:.4f}")

    # --- Guardar modelo ---
    model_path = ROOT_OUT / "rf_v4_17ch_best.joblib"
    joblib.dump(rf, model_path)
    print(f"\nModelo guardado: {model_path}")

    result = {
        "experiment":    "rf_v4_17ch",
        "dataset":       "dataset_v4_17ch (1m, 17ch)",
        "channels":      CHANNEL_NAMES,
        "n_train_tiles": len(train_df),
        "n_train_pixels_used": len(y_train),
        "rf_params": {
            "n_estimators": 500,
            "max_depth": 20,
            "max_features": 0.3,
        },
        "val_metrics":  {k: round(float(v), 4) for k, v in val_metrics.items()},
        "test_metrics": {k: round(float(v), 4) for k, v in test_metrics.items()},
        "feature_importance": {k: round(float(v), 4) for k, v in fi.items()},
    }
    metrics_path = ROOT_OUT / "rf_v4_17ch_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"Metricas guardadas: {metrics_path}")

    print("\n" + "=" * 60)
    print(f"  RF v4-17ch TEST R2   = {test_metrics['R2']:.4f}")
    print(f"  RF v4-17ch TEST RMSE = {test_metrics['RMSE']:.4f}")
    print("=" * 60)
