"""
Busqueda de hiperparametros con Optuna - Random Forest / dataset_v4_ms_sx200.
==============================================================================

Dataset  : dataset_v4_ms_sx200 (1m, 22 canales, Sx_200m)
Split    : train=2021-2023, val=2024, test=2025
Objetivo : maximizar R2 en validacion

Canales (22):
  [0]  DEM          [1]  Slope        [2]  Northness    [3]  Eastness
  [4]  TPI          [5]  SCE          [6-13] Sx_200m (8 dirs)
  [14] Pers_15d     [15] Pers_30d     [16] Pers_60d
  [17] DEM_5m       [18] Slope_5m     [19] Northness_5m
  [20] Eastness_5m  [21] TPI_5m

Espacio de busqueda:
  - n_estimators    : 100, 200, 300, 500
  - max_depth       : 10, 15, 20, 30, None
  - min_samples_leaf: 1, 5, 10, 20
  - max_features    : sqrt, log2, 0.3, 0.5
  - min_samples_split: 2, 5, 10

Salidas:
  - Modelo   : results/optuna_rf_v4_ms_sx200/rf_v4_ms_sx200_best.joblib
  - Metricas : results/optuna_rf_v4_ms_sx200/rf_v4_ms_sx200_metrics.json
  - Ranking  : results/optuna_rf_v4_ms_sx200/ranking_rf_v4_ms_sx200.json
  - BD Optuna: results/optuna_rf_v4_ms_sx200/optuna_rf_v4_ms_sx200.db

Uso:
    .venv/Scripts/python.exe baselines/optuna_rf_v4_ms_sx200.py
    .venv/Scripts/python.exe baselines/optuna_rf_v4_ms_sx200.py --trials 50
    .venv/Scripts/python.exe baselines/optuna_rf_v4_ms_sx200.py --resume
"""

import os
import sys
import json
import time
import argparse
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import joblib
import optuna
from optuna.samplers import TPESampler
from sklearn.ensemble import RandomForestRegressor

_REPO = __import__('pathlib').Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO))

from data.dataset import load_splits
from utils.metrics import compute_metrics

# ---------------------------------------------------------------------------
# Configuracion
# ---------------------------------------------------------------------------
DATA_ROOT   = str(_REPO / 'dataset_v4_ms_sx200')
CSV_FILE    = os.path.join(DATA_ROOT, 'dataset_v4_ms_sx200.csv')
IMGS_DIR    = os.path.join(DATA_ROOT, 'images')
MASKS_DIR   = os.path.join(DATA_ROOT, 'masks')
RESULTS_DIR = str(_REPO / 'results/optuna_rf_v4_ms_sx200')
DB_PATH     = f'sqlite:///{RESULTS_DIR}/optuna_rf_v4_ms_sx200.db'
STUDY_NAME  = 'rf_v4_ms_sx200_hpo_v1'

N_TRIALS   = 30
MAX_PIXELS = 2_000_000
SEED       = 42
N_CH       = 22

CHANNEL_NAMES = [
    'DEM', 'Slope', 'Northness', 'Eastness', 'TPI', 'SCE',
    'Sx_200m_0', 'Sx_200m_45', 'Sx_200m_90', 'Sx_200m_135',
    'Sx_200m_180', 'Sx_200m_225', 'Sx_200m_270', 'Sx_200m_315',
    'Pers_15d', 'Pers_30d', 'Pers_60d',
    'DEM_5m', 'Slope_5m', 'Northness_5m', 'Eastness_5m', 'TPI_5m',
]

# Normalizacion identica a SnowDataset._normalize() en data/dataset.py
DEM_MEAN  = 2100.0
DEM_STD   = 1000.0
SLOPE_MAX =   90.0
TPI_MAX   = 9200.0
SX_MAX    =   90.0

os.makedirs(RESULTS_DIR, exist_ok=True)


def normalize(X: np.ndarray) -> np.ndarray:
    """X: (N, 22) — replica SnowDataset._normalize()"""
    X = X.copy().astype(np.float32)
    X[:, 0] = (X[:, 0] - DEM_MEAN) / DEM_STD          # DEM
    X[:, 1] = X[:, 1] / SLOPE_MAX                      # Slope
    # Northness (2), Eastness (3): ya en [-1, 1]
    X[:, 4] = np.clip(X[:, 4] / TPI_MAX, -1.0, 1.0)   # TPI
    X[:, 5] = (X[:, 5] > 5).astype(np.float32)         # SCE -> binario
    X[:, 6:14] = np.clip(X[:, 6:14] / SX_MAX, -1.0, 1.0)  # Sx_200m x8
    # Persistencia (14-16): ya en [0, 1]
    X[:, 17] = (X[:, 17] - DEM_MEAN) / DEM_STD        # DEM_5m
    X[:, 18] = X[:, 18] / SLOPE_MAX                    # Slope_5m
    # Northness_5m (19), Eastness_5m (20): ya en [-1, 1]
    X[:, 21] = np.clip(X[:, 21] / TPI_MAX, -1.0, 1.0) # TPI_5m
    return X


def load_pixels(df, split_name='', max_pixels=None):
    X_list, y_list = [], []
    for row in df.itertuples():
        img_path  = os.path.join(IMGS_DIR,  row.tile_id)
        mask_path = os.path.join(MASKS_DIR, row.tile_id)
        if not os.path.exists(img_path) or not os.path.exists(mask_path):
            continue
        img  = np.load(img_path).astype(np.float32)[:N_CH]   # (22, 256, 256)
        mask = np.load(mask_path).astype(np.float32)          # (256, 256)
        mask = np.nan_to_num(mask, nan=0.0)
        mask[mask <= -100] = 0.0
        valid = mask > 0.01
        if valid.sum() == 0:
            continue
        img[img == -9999] = 0.0
        img = np.nan_to_num(img, nan=0.0, posinf=0.0, neginf=0.0)
        X_list.append(img[:, valid].T)   # (n_valid, 22)
        y_list.append(mask[valid])
    X = np.vstack(X_list)
    y = np.concatenate(y_list)
    print(f"  {split_name}: {X.shape[0]:,} pixeles")
    if max_pixels and X.shape[0] > max_pixels:
        idx = np.random.RandomState(SEED).choice(X.shape[0], max_pixels, replace=False)
        X, y = X[idx], y[idx]
        print(f"  -> Submuestreo a {max_pixels:,} pixeles")
    return X, y


# Cargar datos una sola vez
print("\nCargando dataset_v4_ms_sx200...")
TRAIN_DF, VAL_DF, TEST_DF = load_splits(CSV_FILE, source='lidar', split_type='temporal')
print(f"  Train: {len(TRAIN_DF)} | Val: {len(VAL_DF)} | Test: {len(TEST_DF)} tiles")

X_train, y_train = load_pixels(TRAIN_DF, 'Train', max_pixels=MAX_PIXELS)
X_val,   y_val   = load_pixels(VAL_DF,   'Val')
X_test,  y_test  = load_pixels(TEST_DF,  'Test')

X_train = normalize(X_train)
X_val   = normalize(X_val)
X_test  = normalize(X_test)
print("  Normalizacion aplicada.\n")


def objective(trial):
    n_est  = trial.suggest_categorical('n_estimators',       [100, 200, 300, 500])
    depth  = trial.suggest_categorical('max_depth',          [10, 15, 20, 30, 'None'])
    msl    = trial.suggest_categorical('min_samples_leaf',   [1, 5, 10, 20])
    mf_raw = trial.suggest_categorical('max_features',       ['sqrt', 'log2', '0.3', '0.5'])
    mss    = trial.suggest_categorical('min_samples_split',  [2, 5, 10])

    max_depth    = None if depth == 'None' else int(depth)
    max_features = float(mf_raw) if mf_raw not in ('sqrt', 'log2') else mf_raw

    rf = RandomForestRegressor(
        n_estimators=n_est, max_depth=max_depth, min_samples_leaf=msl,
        max_features=max_features, min_samples_split=mss,
        n_jobs=-1, random_state=SEED,
    )
    rf.fit(X_train, y_train)
    y_pred = np.maximum(rf.predict(X_val), 0)
    val_r2 = compute_metrics(y_val, y_pred)['R2']

    print(f"  Trial {trial.number:03d} | n_est={n_est} depth={depth} "
          f"msl={msl} mf={mf_raw} mss={mss} | val_R2={val_r2:.4f}")
    return val_r2


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--trials',  type=int, default=N_TRIALS)
    parser.add_argument('--resume',  action='store_true')
    args = parser.parse_args()

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    sampler = TPESampler(seed=SEED, n_startup_trials=10)

    if args.resume:
        study = optuna.load_study(study_name=STUDY_NAME, storage=DB_PATH, sampler=sampler)
    else:
        study = optuna.create_study(study_name=STUDY_NAME, storage=DB_PATH,
                                    direction='maximize', sampler=sampler,
                                    load_if_exists=True)

    print(f"\n{'='*60}")
    print(f"  Optuna RF v4_ms_sx200 | {args.trials} trials | 22 canales")
    print(f"{'='*60}\n")

    study.optimize(objective, n_trials=args.trials, show_progress_bar=False)

    best = study.best_trial
    print(f"\n{'='*60}")
    print(f"  MEJOR TRIAL: #{best.number} | val_R2 = {best.value:.4f}")
    print(f"  Params: {best.params}")
    print(f"{'='*60}")

    # Reentrenar con mejores params en train+val
    print("\nReentrenando modelo final en train+val...")
    depth_f = None if best.params['max_depth'] == 'None' else int(best.params['max_depth'])
    mf_f    = (float(best.params['max_features'])
               if best.params['max_features'] not in ('sqrt', 'log2')
               else best.params['max_features'])

    rf_final = RandomForestRegressor(
        n_estimators=best.params['n_estimators'],
        max_depth=depth_f,
        min_samples_leaf=best.params['min_samples_leaf'],
        max_features=mf_f,
        min_samples_split=best.params['min_samples_split'],
        n_jobs=-1, random_state=SEED,
    )
    X_tv = np.concatenate([X_train, X_val])
    y_tv = np.concatenate([y_train, y_val])
    t0 = time.time()
    rf_final.fit(X_tv, y_tv)
    print(f"  Entrenamiento final: {(time.time()-t0)/60:.1f} min")

    # Evaluar en test
    y_pred_test = np.maximum(rf_final.predict(X_test), 0)
    test_metrics = compute_metrics(y_test, y_pred_test)

    print(f"\n  Test R2   : {test_metrics['R2']:.4f}")
    print(f"  Test MAE  : {test_metrics['MAE']:.4f}")
    print(f"  Test RMSE : {test_metrics['RMSE']:.4f}")
    print(f"  Test Bias : {test_metrics['Bias']:.4f}")

    # Feature importance
    fi = {n: round(float(v), 4) for n, v in zip(CHANNEL_NAMES, rf_final.feature_importances_)}
    print("\n  Feature importance (top 5):")
    for k, v in sorted(fi.items(), key=lambda x: -x[1])[:5]:
        print(f"    {k:<20s}: {v:.4f}")

    # Guardar modelo
    model_path = os.path.join(RESULTS_DIR, 'rf_v4_ms_sx200_best.joblib')
    joblib.dump(rf_final, model_path)
    print(f"\n  Modelo guardado: {model_path}")

    # Guardar metricas
    result = {
        'experiment':   'rf_v4_ms_sx200_optuna',
        'dataset':      'dataset_v4_ms_sx200 (1m, 22ch)',
        'channels':     CHANNEL_NAMES,
        'best_trial':   best.number,
        'val_R2':       round(best.value, 4),
        'best_params':  best.params,
        'test_metrics': {k: round(float(v), 4) for k, v in test_metrics.items()},
        'feature_importance': fi,
    }
    metrics_path = os.path.join(RESULTS_DIR, 'rf_v4_ms_sx200_metrics.json')
    with open(metrics_path, 'w') as f:
        json.dump(result, f, indent=2)
    print(f"  Metricas guardadas: {metrics_path}")

    # Ranking
    ranking = sorted(
        [{'trial': t.number, 'val_R2': round(t.value, 4), 'params': t.params}
         for t in study.trials if t.state.name == 'COMPLETE'],
        key=lambda x: -x['val_R2']
    )
    ranking_path = os.path.join(RESULTS_DIR, 'ranking_rf_v4_ms_sx200.json')
    with open(ranking_path, 'w') as f:
        json.dump(ranking, f, indent=2)
    print(f"  Ranking guardado:   {ranking_path}")

    print(f"\n{'='*60}")
    print(f"  RF v4_ms_sx200 TEST R2   = {test_metrics['R2']:.4f}")
    print(f"  RF v4_ms_sx200 TEST RMSE = {test_metrics['RMSE']:.4f}")
    print(f"{'='*60}\n")


if __name__ == '__main__':
    main()
