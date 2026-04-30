"""
Calcula SPAEF para el modelo RF v4-17ch sobre el test set.
==========================================================

Carga el modelo RF guardado, predice tile a tile (preservando
estructura espacial) y calcula SPAEF por tile + media.

Uso:
    .venv\\Scripts\\python.exe baselines/compute_spaef_rf.py
"""

import json
import numpy as np
import pandas as pd
import joblib
from pathlib import Path

_REPO     = Path(__file__).resolve().parent.parent
ROOT_DATA = _REPO / "dataset_v4_17ch"
ROOT_OUT  = _REPO / "results/rf_v4_17ch"
CSV       = ROOT_DATA / "dataset_v4_17ch.csv"
IMG_DIR   = ROOT_DATA / "images"
MSK_DIR   = ROOT_DATA / "masks"
MODEL_PATH = ROOT_OUT / "rf_v4_17ch_best.joblib"

# Normalizacion identica a rf_v4_17ch.py
DEM_MEAN  = 2100.0
DEM_STD   = 1000.0
SLOPE_MAX =   90.0
TPI_MAX   = 9200.0
SX_MAX    =   90.0


def normalize(X: np.ndarray) -> np.ndarray:
    X = X.copy().astype(np.float32)
    X[:, 0] = (X[:, 0] - DEM_MEAN) / DEM_STD
    X[:, 1] = X[:, 1] / SLOPE_MAX
    X[:, 4] = np.clip(X[:, 4] / TPI_MAX, -1.0, 1.0)
    X[:, 5] = (X[:, 5] > 5).astype(np.float32)
    X[:, 6:14] = np.clip(X[:, 6:14] / SX_MAX, -1.0, 1.0)
    return X


def compute_spaef(obs: np.ndarray, sim: np.ndarray, n_bins: int = 100) -> float:
    from utils.metrics import compute_spaef as _spaef
    return _spaef(obs, sim, n_bins)


def main():
    import sys
    sys.path.insert(0, str(_REPO))
    from utils.metrics import compute_spaef as _spaef

    print(f"Cargando modelo RF: {MODEL_PATH}")
    rf = joblib.load(MODEL_PATH)

    df  = pd.read_csv(CSV)
    test_df = df[
        (df['exp_temporal_split'] == 'test') &
        (df['source'] == 'lidar')
    ].reset_index(drop=True)
    print(f"Test tiles: {len(test_df)}")

    spaef_vals = []

    for _, row in test_df.iterrows():
        img_path = IMG_DIR / row['tile_id']
        msk_path = MSK_DIR / row['tile_id']
        if not img_path.exists() or not msk_path.exists():
            continue

        img  = np.load(img_path)   # (17, H, W)
        mask = np.load(msk_path)   # (H, W)

        C, H, W = img.shape
        X_tile  = img[:17].reshape(17, -1).T  # (H*W, 17)
        X_tile  = normalize(X_tile)

        y_pred  = rf.predict(X_tile)          # (H*W,)
        y_true  = mask.flatten()

        valid   = y_true > 0.01
        if valid.sum() < 10:
            continue

        spaef = _spaef(y_true[valid], y_pred[valid])
        if not np.isnan(spaef):
            spaef_vals.append(spaef)

    if spaef_vals:
        print(f"\nSPAEF RF v4-17ch:")
        print(f"  Media  : {np.mean(spaef_vals):.4f}")
        print(f"  Std    : {np.std(spaef_vals):.4f}")
        print(f"  N tiles: {len(spaef_vals)}")

        # Actualizar metrics.json
        metrics_path = ROOT_OUT / "rf_v4_17ch_metrics.json"
        if metrics_path.exists():
            with open(metrics_path) as f:
                data = json.load(f)
            data['test_metrics']['SPAEF']       = round(float(np.mean(spaef_vals)), 4)
            data['test_metrics']['SPAEF_std']   = round(float(np.std(spaef_vals)),  4)
            data['test_metrics']['SPAEF_n_tiles'] = len(spaef_vals)
            with open(metrics_path, 'w') as f:
                json.dump(data, f, indent=2)
            print(f"  Guardado en: {metrics_path}")
    else:
        print("No se pudo calcular SPAEF.")


if __name__ == '__main__':
    main()
