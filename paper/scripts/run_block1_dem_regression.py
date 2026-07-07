"""
PAPER - Block 1: Baseline de regresion lineal DEM -> snow depth.
=================================================================
Ajusta una regresion lineal simple (DEM -> profundidad de nieve) sobre el
conjunto de entrenamiento y evalua sobre el test set.

Es el baseline fisico minimo exigible en un paper de snow depth prediction:
la correlacion elevacion-nieve es el predictor mas basico del dominio y
sirve como referencia interpretable por encima del predictor constante.

Variantes evaluadas:
    1. Linear (DEM)         : y = a*DEM + b
    2. Linear (DEM + Slope) : y = a*DEM + b*Slope + c

Calcula R2, RMSE, MAE, Bias, SPAEF y MSPAEF (tile a tile) para comparacion
directa con RF y CNNs en la misma metrica espacial.

Salidas:
    paper/results/block1/b1_dem_regression/b1_dem_regression_metrics.json

Uso:
    .venv\\Scripts\\python.exe paper/scripts/run_block1_dem_regression.py
"""

import json
import time
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

REPO      = Path(__file__).resolve().parent.parent.parent
ROOT_DATA = REPO / "dataset_v4_ms_sx200"
ROOT_OUT  = REPO / "paper/results/block1/b1_dem_regression"
ROOT_OUT.mkdir(parents=True, exist_ok=True)

CSV     = ROOT_DATA / "dataset_v4_ms_sx200.csv"
IMG_DIR = ROOT_DATA / "images"
MSK_DIR = ROOT_DATA / "masks"

# Normalizacion identica a SnowDataset._normalize()
DEM_MEAN  = 2100.0
DEM_STD   = 1000.0
SLOPE_MAX =   90.0


# ---------------------------------------------------------------------------
# SPAEF y MSPAEF (mismo calculo que en utils/metrics.py)
# ---------------------------------------------------------------------------

def compute_spaef(obs, sim, n_bins=100):
    obs = np.asarray(obs, dtype=np.float64)
    sim = np.asarray(sim, dtype=np.float64)
    sim = np.maximum(sim, 0.0)
    if len(obs) < 10:
        return float('nan')
    rho = float(np.corrcoef(obs, sim)[0, 1])
    if np.isnan(rho):
        return float('nan')
    mean_obs = float(np.mean(obs))
    mean_sim = float(np.mean(sim))
    if mean_obs == 0.0 or mean_sim == 0.0:
        return float('nan')
    cv_obs = float(np.std(obs)) / mean_obs
    cv_sim = float(np.std(sim)) / mean_sim
    if cv_obs == 0.0:
        return float('nan')
    alpha = cv_sim / cv_obs
    lo = min(float(obs.min()), float(sim.min()))
    hi = max(float(obs.max()), float(sim.max()))
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
    sim = np.asarray(sim, dtype=np.float64)
    sim = np.maximum(sim, 0.0)
    if len(obs) < 10:
        return float('nan')
    iqr_obs = float(np.percentile(obs, 75) - np.percentile(obs, 25))
    if iqr_obs == 0.0:
        return float('nan')
    alpha = float(np.corrcoef(obs, sim)[0, 1])
    if np.isnan(alpha):
        return float('nan')
    rmse  = float(np.sqrt(np.mean((sim - obs) ** 2)))
    beta  = rmse / iqr_obs
    gamma = abs(float(np.mean(sim)) - float(np.mean(obs))) / iqr_obs
    std_obs = float(np.std(obs))
    if std_obs == 0.0:
        return float('nan')
    sigma = float(np.std(sim)) / std_obs
    delta = (sigma - 1.0) / 2.0 + abs(sigma - 1.0) / (sigma + 2.0)
    return float(1.0 - 0.25 * ((alpha - 1.0)**2 + beta**2 + gamma**2 + delta**2))


# ---------------------------------------------------------------------------
# Carga de datos
# ---------------------------------------------------------------------------

def load_split(df: pd.DataFrame, channels: list):
    """
    Carga tiles y devuelve X (N, len(channels)), y (N,) y lista de tiles
    con sus pixeles validos para calculo tile-by-tile de SPAEF.
    """
    X_list, y_list = [], []
    tiles = []   # lista de (x_tile, y_tile) para SPAEF por tile

    for row in df.itertuples():
        img_path = IMG_DIR / row.tile_id
        msk_path = MSK_DIR / row.tile_id
        try:
            img  = np.load(img_path).astype(np.float32)
            mask = np.load(msk_path).astype(np.float32)
        except Exception:
            continue

        valid = mask > 0.01
        if valid.sum() < 10:
            continue

        feats = img[channels, :, :]           # (n_ch, 256, 256)
        x_tile = feats[:, valid].T            # (n_valid, n_ch)
        y_tile = mask[valid]

        # Limpiar nodata
        x_tile[x_tile == -9999] = 0.0
        x_tile = np.nan_to_num(x_tile, nan=0.0, posinf=0.0, neginf=0.0)

        X_list.append(x_tile)
        y_list.append(y_tile)
        tiles.append((x_tile, y_tile))

    return np.vstack(X_list), np.concatenate(y_list), tiles


def normalize_features(X: np.ndarray, channels: list) -> np.ndarray:
    X = X.copy()
    for i, ch in enumerate(channels):
        if ch == 0:   # DEM
            X[:, i] = (X[:, i] - DEM_MEAN) / DEM_STD
        elif ch == 1: # Slope
            X[:, i] = X[:, i] / SLOPE_MAX
    return X


# ---------------------------------------------------------------------------
# Evaluacion de una variante
# ---------------------------------------------------------------------------

def evaluate_variant(name, channels, channel_names, model, X_test, y_test, test_tiles):
    y_pred = np.maximum(model.predict(X_test), 0.0)

    r2   = float(r2_score(y_test, y_pred))
    rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
    mae  = float(mean_absolute_error(y_test, y_pred))
    bias = float(np.mean(y_pred - y_test))

    # SPAEF y MSPAEF por tile
    spaef_vals, mspaef_vals = [], []
    for x_tile, y_tile in test_tiles:
        x_norm = normalize_features(x_tile, channels)
        pred_tile = np.maximum(model.predict(x_norm), 0.0)
        s  = compute_spaef(y_tile, pred_tile)
        ms = compute_mspaef(y_tile, pred_tile)
        if not np.isnan(s):
            spaef_vals.append(s)
        if not np.isnan(ms):
            mspaef_vals.append(ms)

    coef = {channel_names[i]: round(float(model.coef_[i]), 6) for i in range(len(channels))}
    coef['intercept'] = round(float(model.intercept_), 6)

    return {
        "variant":     name,
        "channels":    channel_names,
        "coefficients": coef,
        "test_metrics": {
            "R2":           round(r2,   4),
            "RMSE":         round(rmse, 4),
            "MAE":          round(mae,  4),
            "Bias":         round(bias, 4),
            "SPAEF":        round(float(np.mean(spaef_vals)),  4) if spaef_vals  else float('nan'),
            "SPAEF_std":    round(float(np.std(spaef_vals)),   4) if spaef_vals  else float('nan'),
            "SPAEF_n_tiles": len(spaef_vals),
            "MSPAEF":       round(float(np.mean(mspaef_vals)), 4) if mspaef_vals else float('nan'),
            "MSPAEF_std":   round(float(np.std(mspaef_vals)),  4) if mspaef_vals else float('nan'),
            "MSPAEF_n_tiles": len(mspaef_vals),
        }
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 60)
    print("  PAPER Block 1 — DEM regression baseline")
    print("=" * 60)

    if not CSV.exists():
        raise SystemExit(f"Dataset no encontrado: {CSV}")

    df = pd.read_csv(CSV)
    train_df = df[df["exp_temporal_split"] == "train"].reset_index(drop=True)
    test_df  = df[df["exp_temporal_split"] == "test"].reset_index(drop=True)
    print(f"\nSplit: train={len(train_df)}  test={len(test_df)}")

    variants = [
        ("DEM only",        [0],    ["DEM"]),
        ("DEM + Slope",     [0, 1], ["DEM", "Slope"]),
    ]

    results = []

    for name, channels, ch_names in variants:
        print(f"\n--- Variante: {name} ---")

        t0 = time.time()
        print("  Cargando train...", flush=True)
        X_tr, y_tr, _ = load_split(train_df, channels)
        X_tr = normalize_features(X_tr, channels)
        print(f"  Train pixels: {len(y_tr):,}  ({(time.time()-t0)/60:.1f} min)")

        # Submuestrear si es necesario
        MAX_PIX = 2_000_000
        if len(y_tr) > MAX_PIX:
            rng = np.random.RandomState(42)
            idx = rng.choice(len(y_tr), MAX_PIX, replace=False)
            X_tr, y_tr = X_tr[idx], y_tr[idx]
            print(f"  Submuestreado a {MAX_PIX:,} pixels")

        model = LinearRegression()
        model.fit(X_tr, y_tr)
        print(f"  Coeficientes: {dict(zip(ch_names, model.coef_.round(4)))}  intercept={model.intercept_:.4f}")

        print("  Cargando test...", flush=True)
        X_te, y_te, test_tiles = load_split(test_df, channels)
        X_te = normalize_features(X_te, channels)

        res = evaluate_variant(name, channels, ch_names, model, X_te, y_te, test_tiles)
        m = res["test_metrics"]
        print(f"  R2={m['R2']:.4f}  RMSE={m['RMSE']:.4f}  MAE={m['MAE']:.4f}  "
              f"Bias={m['Bias']:.4f}  SPAEF={m['SPAEF']:.4f}  MSPAEF={m['MSPAEF']:.4f}")
        results.append(res)

    # Guardar JSON (variante principal = DEM only para compile_tables)
    best = next(r for r in results if r["variant"] == "DEM only")
    output = {
        "experiment":    "b1_dem_regression",
        "dataset":       "dataset_v4_ms_sx200 (1m, 22ch)",
        "description":   "Linear regression DEM -> snow depth (physical baseline)",
        "test_metrics":  best["test_metrics"],
        "all_variants":  results,
    }
    out_path = ROOT_OUT / "b1_dem_regression_metrics.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nGuardado: {out_path}")

    print("\n" + "=" * 60)
    print("  Resumen (test 2025):")
    for r in results:
        m = r["test_metrics"]
        print(f"  {r['variant']:<20s}: R2={m['R2']:.4f}  RMSE={m['RMSE']:.4f}  SPAEF={m['SPAEF']:.4f}")
    print("=" * 60)
