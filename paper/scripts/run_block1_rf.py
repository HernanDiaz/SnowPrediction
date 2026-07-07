"""
PAPER - Block 1: Random Forest baseline en dataset_v4_ms_sx200 (22ch).
=======================================================================
Entrena un Random Forest con los mismos 22 canales que el ResUNet++
(dataset_v4_ms_sx200, split temporal identico) para comparacion directa.

Canales (22):
    [0]  DEM           - Elevacion (metros)
    [1]  Slope         - Pendiente (grados)
    [2]  Northness     - cos(aspect) [-1, 1]
    [3]  Eastness      - sin(aspect) [-1, 1]
    [4]  TPI           - Topographic Position Index
    [5]  SCE           - Snow Cover Extent (0/10/11 -> binario)
    [6]  Sx_200m_0     - Wind Shelter Index, dir 0 (N)
    [7]  Sx_200m_45
    [8]  Sx_200m_90
    [9]  Sx_200m_135
    [10] Sx_200m_180
    [11] Sx_200m_225
    [12] Sx_200m_270
    [13] Sx_200m_315
    [14] Persistencia_15d
    [15] Persistencia_30d
    [16] Persistencia_60d
    [17] DEM_5m        - Elevacion a resolucion 5m (promedio del tile)
    [18] Slope_5m
    [19] Northness_5m
    [20] Eastness_5m
    [21] TPI_5m

Salidas:
    Modelo  : paper/results/block1/b1_rf/b1_rf_best.joblib
    Metricas: paper/results/block1/b1_rf/b1_rf_metrics.json

Uso:
    .venv\\Scripts\\python.exe paper/scripts/run_block1_rf.py
"""

import argparse
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
REPO      = Path(__file__).resolve().parent.parent.parent
ROOT_DATA = REPO / "dataset_v4_ms_sx200"
ROOT_OUT  = REPO / "paper/results/block1/b1_rf"
ROOT_OUT.mkdir(parents=True, exist_ok=True)

CSV     = ROOT_DATA / "dataset_v4_ms_sx200.csv"
IMG_DIR = ROOT_DATA / "images"
MSK_DIR = ROOT_DATA / "masks"

CHANNEL_NAMES = [
    "DEM", "Slope", "Northness", "Eastness", "TPI", "SCE",
    "Sx200_0", "Sx200_45", "Sx200_90", "Sx200_135",
    "Sx200_180", "Sx200_225", "Sx200_270", "Sx200_315",
    "Pers_15d", "Pers_30d", "Pers_60d",
    "DEM_5m", "Slope_5m", "Northness_5m", "Eastness_5m", "TPI_5m",
]
N_CHANNELS = 22

# Normalizacion identica a SnowDataset._normalize() en data/dataset.py
DEM_MEAN  = 2100.0
DEM_STD   = 1000.0
SLOPE_MAX =   90.0
TPI_MAX   = 9200.0
SX_MAX    =   90.0


def normalize(X: np.ndarray) -> np.ndarray:
    """
    Normaliza columnas en el mismo orden y escala que SnowDataset._normalize().
    X shape: (N, 22)
    """
    X = X.copy().astype(np.float32)
    X[:, 0] = (X[:, 0] - DEM_MEAN) / DEM_STD            # DEM
    X[:, 1] = X[:, 1] / SLOPE_MAX                        # Slope
    # Northness (2) y Eastness (3) ya en [-1, 1]
    X[:, 4] = np.clip(X[:, 4] / TPI_MAX, -1.0, 1.0)     # TPI
    X[:, 5] = (X[:, 5] > 5).astype(np.float32)           # SCE -> binario
    X[:, 6:14] = np.clip(X[:, 6:14] / SX_MAX, -1.0, 1.0) # Sx x8
    # Persistencia (14-16) ya en [0, 1] — sin cambios
    # DEM_5m (17): misma normalizacion que DEM
    X[:, 17] = (X[:, 17] - DEM_MEAN) / DEM_STD
    # Slope_5m (18)
    X[:, 18] = X[:, 18] / SLOPE_MAX
    # Northness_5m (19), Eastness_5m (20) ya en [-1, 1]
    # TPI_5m (21)
    X[:, 21] = np.clip(X[:, 21] / TPI_MAX, -1.0, 1.0)
    return X


def load_split_pixels(df: pd.DataFrame):
    """Carga todos los tiles del split y devuelve X (N, 22), y (N,) y tiles."""
    X_list, y_list, tiles = [], [], []
    n_tiles = len(df)

    for i, row in enumerate(df.itertuples(), 1):
        if i % 200 == 0:
            print(f"  Cargando tile {i}/{n_tiles}...", flush=True)

        img_path = IMG_DIR / row.tile_id
        msk_path = MSK_DIR / row.tile_id

        try:
            img  = np.load(img_path).astype(np.float32)
            mask = np.load(msk_path).astype(np.float32)
        except Exception as e:
            print(f"  Error cargando {row.tile_id}: {e}")
            continue

        valid = mask > 0.01
        if valid.sum() < 10:
            continue

        features = img[:N_CHANNELS, :, :]
        X = features[:, valid].T
        y = mask[valid]

        X[X == -9999] = 0.0
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

        X_list.append(X)
        y_list.append(y)
        tiles.append((X.copy(), y.copy()))

    if not X_list:
        raise RuntimeError("No se cargaron pixeles validos. Verifica el dataset.")

    return np.vstack(X_list), np.concatenate(y_list), tiles


def compute_spaef(obs, sim, n_bins=100):
    obs = np.asarray(obs, dtype=np.float64)
    sim = np.maximum(np.asarray(sim, dtype=np.float64), 0.0)
    if len(obs) < 10:
        return float('nan')
    rho = float(np.corrcoef(obs, sim)[0, 1])
    if np.isnan(rho):
        return float('nan')
    m_obs, m_sim = float(np.mean(obs)), float(np.mean(sim))
    if m_obs == 0.0 or m_sim == 0.0:
        return float('nan')
    cv_obs = float(np.std(obs)) / m_obs
    cv_sim = float(np.std(sim)) / m_sim
    if cv_obs == 0.0:
        return float('nan')
    alpha = cv_sim / cv_obs
    lo, hi = min(obs.min(), sim.min()), max(obs.max(), sim.max())
    if hi <= lo:
        return float('nan')
    bins = np.linspace(lo, hi, n_bins + 1)
    h_obs, _ = np.histogram(obs, bins=bins)
    h_sim, _ = np.histogram(sim, bins=bins)
    h_obs = h_obs / (h_obs.sum() + 1e-10)
    h_sim = h_sim / (h_sim.sum() + 1e-10)
    beta = float(np.sum(np.minimum(h_obs, h_sim)))
    return float(1.0 - np.sqrt((rho - 1.0)**2 + (alpha - 1.0)**2 + (beta - 1.0)**2))


def compute_mspaef(obs, sim):
    obs = np.asarray(obs, dtype=np.float64)
    sim = np.maximum(np.asarray(sim, dtype=np.float64), 0.0)
    if len(obs) < 10:
        return float('nan')
    iqr = float(np.percentile(obs, 75) - np.percentile(obs, 25))
    if iqr == 0.0:
        return float('nan')
    alpha = float(np.corrcoef(obs, sim)[0, 1])
    if np.isnan(alpha):
        return float('nan')
    beta  = float(np.sqrt(np.mean((sim - obs)**2))) / iqr
    gamma = abs(float(np.mean(sim)) - float(np.mean(obs))) / iqr
    std_obs = float(np.std(obs))
    if std_obs == 0.0:
        return float('nan')
    sigma = float(np.std(sim)) / std_obs
    delta = (sigma - 1.0) / 2.0 + abs(sigma - 1.0) / (sigma + 2.0)
    return float(1.0 - 0.25 * ((alpha - 1.0)**2 + beta**2 + gamma**2 + delta**2))


def compute_spatial_metrics(rf_model, tiles):
    """Calcula SPAEF y MSPAEF tile a tile sobre pixeles con nieve (>0.01m)."""
    spaef_vals, mspaef_vals = [], []
    for X_tile, y_tile in tiles:
        X_norm = normalize(X_tile.copy())
        pred = np.maximum(rf_model.predict(X_norm), 0.0)
        s  = compute_spaef(y_tile, pred)
        ms = compute_mspaef(y_tile, pred)
        if not np.isnan(s):
            spaef_vals.append(s)
        if not np.isnan(ms):
            mspaef_vals.append(ms)
    return spaef_vals, mspaef_vals


def compute_metrics(y_true, y_pred):
    return {
        "R2":   round(float(r2_score(y_true, y_pred)), 4),
        "RMSE": round(float(np.sqrt(mean_squared_error(y_true, y_pred))), 4),
        "MAE":  round(float(mean_absolute_error(y_true, y_pred)), 4),
        "Bias": round(float(np.mean(y_pred - y_true)), 4),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval-only", action="store_true",
                        help="Solo evaluar: carga modelo guardado, calcula SPAEF/MSPAEF")
    args = parser.parse_args()

    print("=" * 60)
    print("  PAPER Block 1 — RF baseline | 22ch | dataset_v4_ms_sx200")
    print("=" * 60)

    if not CSV.exists():
        print(f"\nERROR: Dataset no encontrado: {CSV}")
        raise SystemExit(1)

    df = pd.read_csv(CSV)
    train_df = df[df["exp_temporal_split"] == "train"].reset_index(drop=True)
    val_df   = df[df["exp_temporal_split"] == "val"].reset_index(drop=True)
    test_df  = df[df["exp_temporal_split"] == "test"].reset_index(drop=True)
    print(f"\nSplit: train={len(train_df)}  val={len(val_df)}  test={len(test_df)}")

    model_path   = ROOT_OUT / "b1_rf_best.joblib"
    metrics_path = ROOT_OUT / "b1_rf_metrics.json"

    if args.eval_only:
        # --- Modo evaluacion: cargar modelo ya entrenado ---
        if not model_path.exists():
            raise SystemExit(f"Modelo no encontrado: {model_path}. Ejecuta sin --eval-only primero.")
        print(f"\nCargando modelo desde: {model_path}")
        rf = joblib.load(model_path)
        # Cargar metricas existentes para conservarlas
        with open(metrics_path) as f:
            result = json.load(f)
        val_metrics = result.get("val_metrics", {})
        fi = result.get("feature_importance", {})
        n_train_tiles = result.get("n_train_tiles", len(train_df))
        n_train_pixels = result.get("n_train_pixels_used", 0)
    else:
        # --- Modo entrenamiento completo ---
        print("\nCargando train...", flush=True)
        t0 = time.time()
        X_train, y_train, _ = load_split_pixels(train_df)
        print(f"  Train pixels: {len(y_train):,}  ({(time.time()-t0)/60:.1f} min)")
        X_train = normalize(X_train)

        MAX_PIXELS = 2_000_000
        if len(y_train) > MAX_PIXELS:
            rng = np.random.RandomState(42)
            idx = rng.choice(len(y_train), MAX_PIXELS, replace=False)
            X_train, y_train = X_train[idx], y_train[idx]
            print(f"  Submuestreado a {MAX_PIXELS:,} pixels")

        print("\nEntrenando RF...", flush=True)
        rf = RandomForestRegressor(
            n_estimators=500, max_depth=20, min_samples_leaf=1,
            max_features=0.3, min_samples_split=5, n_jobs=-1, random_state=42,
        )
        t0 = time.time()
        rf.fit(X_train, y_train)
        print(f"  Entrenamiento: {(time.time()-t0)/60:.1f} min")

        print("\nCargando val...", flush=True)
        X_val, y_val, _ = load_split_pixels(val_df)
        X_val = normalize(X_val)
        val_metrics = compute_metrics(y_val, rf.predict(X_val))
        print(f"  Val  R2={val_metrics['R2']:.4f}  RMSE={val_metrics['RMSE']:.4f}")

        ROOT_OUT.mkdir(parents=True, exist_ok=True)
        joblib.dump(rf, model_path)
        print(f"\nModelo guardado: {model_path}")

        fi = dict(zip(CHANNEL_NAMES, rf.feature_importances_.tolist()))
        n_train_tiles  = len(train_df)
        n_train_pixels = int(len(y_train))
        result = {}

    # --- Test (siempre) ---
    print("\nCargando test...", flush=True)
    X_test, y_test, test_tiles = load_split_pixels(test_df)
    X_test_norm = normalize(X_test.copy())
    y_test_pred = rf.predict(X_test_norm)
    test_metrics = compute_metrics(y_test, y_test_pred)

    print(f"\n  Test R2  : {test_metrics['R2']:.4f}")
    print(f"  Test RMSE: {test_metrics['RMSE']:.4f}")
    print(f"  Test MAE : {test_metrics['MAE']:.4f}")
    print(f"  Test Bias: {test_metrics['Bias']:.4f}")

    # --- SPAEF y MSPAEF por tile ---
    print("\nCalculando SPAEF y MSPAEF por tile...", flush=True)
    spaef_vals, mspaef_vals = compute_spatial_metrics(rf, test_tiles)
    test_metrics['SPAEF']         = round(float(np.mean(spaef_vals)),  4) if spaef_vals  else float('nan')
    test_metrics['SPAEF_std']     = round(float(np.std(spaef_vals)),   4) if spaef_vals  else float('nan')
    test_metrics['SPAEF_n_tiles'] = len(spaef_vals)
    test_metrics['MSPAEF']        = round(float(np.mean(mspaef_vals)), 4) if mspaef_vals else float('nan')
    test_metrics['MSPAEF_std']    = round(float(np.std(mspaef_vals)),  4) if mspaef_vals else float('nan')
    test_metrics['MSPAEF_n_tiles'] = len(mspaef_vals)
    print(f"  SPAEF : {test_metrics['SPAEF']:.4f}  (std={test_metrics['SPAEF_std']:.4f}, n={test_metrics['SPAEF_n_tiles']})")
    print(f"  MSPAEF: {test_metrics['MSPAEF']:.4f}  (std={test_metrics['MSPAEF_std']:.4f}, n={test_metrics['MSPAEF_n_tiles']})")

    if not args.eval_only:
        print("\nFeature importance:")
        for k, v in sorted(fi.items(), key=lambda x: -x[1]):
            print(f"  {k:15s}: {v:.4f}")

    result.update({
        "experiment":          "b1_rf",
        "dataset":             "dataset_v4_ms_sx200 (1m, 22ch)",
        "channels":            CHANNEL_NAMES,
        "n_train_tiles":       n_train_tiles,
        "n_train_pixels_used": n_train_pixels,
        "rf_params": {"n_estimators": 500, "max_depth": 20, "max_features": 0.3,
                      "min_samples_leaf": 1, "min_samples_split": 5, "random_state": 42},
        "val_metrics":         {k: round(float(v), 4) for k, v in val_metrics.items()},
        "test_metrics":        {k: round(float(v), 4) for k, v in test_metrics.items()},
        "feature_importance":  {k: round(float(v), 4) for k, v in fi.items()},
    })
    with open(metrics_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nMetricas guardadas: {metrics_path}")

    print("\n" + "=" * 60)
    print(f"  PAPER b1_rf  TEST R2    = {test_metrics['R2']:.4f}")
    print(f"  PAPER b1_rf  TEST RMSE  = {test_metrics['RMSE']:.4f}")
    print(f"  PAPER b1_rf  TEST SPAEF = {test_metrics['SPAEF']:.4f}")
    print("=" * 60)
