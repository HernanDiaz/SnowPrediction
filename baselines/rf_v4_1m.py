"""
Random Forest en dataset v4_fisico (1m de resolucion, 5 canales topograficos).
===============================================================================

Comparacion directa con RF v5 (mismo modelo, mismos canales, diferente resolucion):
    RF v5 (5m, 5 topo):  Train~60 tiles (~983K pixels)  -> R2=0.2555
    RF v4 (1m, 5 topo):  Train~3134 tiles (~205M pixels) -> ???

Canales: DEM, Slope, Northness, Eastness, TPI  (mismos que RF v5)
Split  : temporal (train=2021-2022, val=2023, test=2024-2025)

Salidas:
    Modelo  : results/rf_v4_1m/rf_v4_1m_best.joblib
    Metricas: results/rf_v4_1m/rf_v4_1m_metrics.json
"""

import os
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
ROOT_DATA = Path("E:/PycharmProjects/SnowPrediction/dataset_v4_fisico")
ROOT_OUT  = Path("E:/PycharmProjects/SnowPrediction/results/rf_v4_1m")
ROOT_OUT.mkdir(parents=True, exist_ok=True)

CSV = ROOT_DATA / "dataset_v4_fisico.csv"
IMG_DIR = ROOT_DATA / "images"
MSK_DIR = ROOT_DATA / "masks"

# Canales: DEM, Slope, Northness, Eastness, TPI  (indices 0-4)
CHANNEL_NAMES = ["DEM", "Slope", "Northness", "Eastness", "TPI"]
CHANNEL_IDX   = [0, 1, 2, 3, 4]

# Normalizacion (misma que dataset.py)
DEM_MEAN  = 2100.0
DEM_STD   = 1000.0
SLOPE_MAX =   90.0
TPI_MAX   = 9200.0


def normalize(X: np.ndarray) -> np.ndarray:
    """Normaliza columnas: DEM, Slope, Northness, Eastness, TPI."""
    X = X.copy().astype(np.float32)
    X[:, 0] = (X[:, 0] - DEM_MEAN) / DEM_STD           # DEM
    X[:, 1] = X[:, 1] / SLOPE_MAX                       # Slope
    # Northness y Eastness ya en [-1, 1]
    X[:, 4] = np.clip(X[:, 4] / TPI_MAX, -1.0, 1.0)    # TPI
    return X


def load_split_pixels(df: pd.DataFrame):
    """
    Carga todos los tiles del split y devuelve X (N, 5), y (N,).
    Solo incluye pixels con mascara valida (no NaN).
    """
    X_list, y_list = [], []
    n_tiles = len(df)

    for i, row in enumerate(df.itertuples(), 1):
        if i % 500 == 0:
            print(f"  Cargando tile {i}/{n_tiles}...", flush=True)

        img_path = IMG_DIR / row.tile_id
        msk_path = MSK_DIR / row.tile_id

        try:
            img  = np.load(img_path).astype(np.float32)   # (6, 256, 256)
            mask = np.load(msk_path).astype(np.float32)   # (256, 256)
        except Exception as e:
            print(f"  Error cargando {row.tile_id}: {e}")
            continue

        # Solo canales topograficos [0-4]
        features = img[CHANNEL_IDX, :, :]  # (5, 256, 256)

        # Mascara valida
        valid = ~np.isnan(mask)
        if valid.sum() == 0:
            continue

        # Pilas y filtrado
        X = features[:, valid].T          # (n_valid, 5)
        y = mask[valid]                   # (n_valid,)

        # Limpiar valores extremos
        X[X == -9999] = 0.0
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

        X_list.append(X)
        y_list.append(y)

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
    print("  RF v4-1m | 5 canales topo | split temporal")
    print("=" * 60)

    df = pd.read_csv(CSV)
    train_df = df[df["exp_temporal_split"] == "train"].reset_index(drop=True)
    val_df   = df[df["exp_temporal_split"] == "val"].reset_index(drop=True)
    test_df  = df[df["exp_temporal_split"] == "test"].reset_index(drop=True)

    print(f"\nSplit: train={len(train_df)} val={len(val_df)} test={len(test_df)}")

    # --- Cargar pixels ---
    print("\nCargando train...", flush=True)
    t0 = time.time()
    X_train, y_train = load_split_pixels(train_df)
    print(f"  Train pixels: {len(y_train):,}  ({(time.time()-t0)/60:.1f} min)")

    print("\nNormalizando...", flush=True)
    X_train = normalize(X_train)

    # Submuestrear si es necesario (>5M pixels puede ser lento para RF)
    MAX_PIXELS = 2_000_000
    if len(y_train) > MAX_PIXELS:
        idx = np.random.RandomState(42).choice(len(y_train), MAX_PIXELS, replace=False)
        X_train = X_train[idx]
        y_train = y_train[idx]
        print(f"  Submuestreado a {MAX_PIXELS:,} pixels")

    # --- Entrenar RF (mismos parametros que RF v5 mejor trial) ---
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

    # --- Feature importance ---
    fi = dict(zip(CHANNEL_NAMES, rf.feature_importances_.tolist()))
    print("\nFeature importance:", {k: round(v, 4) for k, v in fi.items()})

    # --- Guardar ---
    model_path = ROOT_OUT / "rf_v4_1m_best.joblib"
    joblib.dump(rf, model_path)
    print(f"\nModelo guardado: {model_path}")

    result = {
        "experiment":    "rf_v4_1m",
        "dataset":       "dataset_v4_fisico (1m)",
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
    metrics_path = ROOT_OUT / "rf_v4_1m_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"Metricas guardadas: {metrics_path}")

    print("\n" + "=" * 60)
    print(f"  RF v4-1m TEST R2   = {test_metrics['R2']:.4f}")
    print(f"  RF v4-1m TEST RMSE = {test_metrics['RMSE']:.4f}")
    print("=" * 60)
