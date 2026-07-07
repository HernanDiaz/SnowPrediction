"""
Evalua RF con los mejores hiperparametros de optuna_rf_v6 para 3 seeds.
=======================================================================
Mejores params (Trial 90):
  n_estimators=300, max_depth=15, min_samples_leaf=1,
  max_features='log2', min_samples_split=2

Uso:
    .venv/Scripts/python.exe baselines/eval_rf_v6_seeds.py --seed 42
    .venv/Scripts/python.exe baselines/eval_rf_v6_seeds.py --seed 123
    .venv/Scripts/python.exe baselines/eval_rf_v6_seeds.py --seed 7
"""

import os
import sys
import json
import time
import argparse
import warnings
warnings.filterwarnings('ignore')

import numpy as np
from sklearn.ensemble import RandomForestRegressor

_REPO = __import__('pathlib').Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO))

from data.dataset import load_splits
from utils.metrics import compute_metrics

DATA_ROOT = str(_REPO / 'dataset_v4_ms_sx200')
CSV_FILE  = os.path.join(DATA_ROOT, 'dataset_v4_ms_sx200.csv')
IMGS_DIR  = os.path.join(DATA_ROOT, 'images')
MASKS_DIR = os.path.join(DATA_ROOT, 'masks')

N_CH       = 22
MAX_PIXELS = 2_000_000

# Mejores hiperparametros de Optuna v6 (Trial 90)
BEST_PARAMS = {
    'n_estimators':      300,
    'max_depth':         15,
    'min_samples_leaf':  1,
    'max_features':      'log2',
    'min_samples_split': 2,
}

DEM_MEAN  = 2100.0
DEM_STD   = 1000.0
SLOPE_MAX =   90.0
TPI_MAX   = 9200.0
SX_MAX    =   90.0


def normalize(X):
    X = X.copy().astype(np.float32)
    X[:, 0] = (X[:, 0] - DEM_MEAN) / DEM_STD
    X[:, 1] = X[:, 1] / SLOPE_MAX
    X[:, 4] = np.clip(X[:, 4] / TPI_MAX, -1.0, 1.0)
    X[:, 5] = (X[:, 5] > 5).astype(np.float32)
    X[:, 6:14] = np.clip(X[:, 6:14] / SX_MAX, -1.0, 1.0)
    X[:, 17] = (X[:, 17] - DEM_MEAN) / DEM_STD
    X[:, 18] = X[:, 18] / SLOPE_MAX
    X[:, 21] = np.clip(X[:, 21] / TPI_MAX, -1.0, 1.0)
    return X


def load_pixels(df, split_name, seed, max_pixels=None):
    X_list, y_list = [], []
    for row in df.itertuples():
        img_path  = os.path.join(IMGS_DIR,  row.tile_id)
        mask_path = os.path.join(MASKS_DIR, row.tile_id)
        if not os.path.exists(img_path) or not os.path.exists(mask_path):
            continue
        img  = np.load(img_path).astype(np.float32)[:N_CH]
        mask = np.load(mask_path).astype(np.float32)
        mask = np.nan_to_num(mask, nan=0.0)
        mask[mask <= -100] = 0.0
        valid = mask > 0.01
        if valid.sum() == 0:
            continue
        img[img == -9999] = 0.0
        img = np.nan_to_num(img, nan=0.0, posinf=0.0, neginf=0.0)
        X_list.append(img[:, valid].T)
        y_list.append(mask[valid])
    X = np.vstack(X_list)
    y = np.concatenate(y_list)
    print(f"  {split_name}: {X.shape[0]:,} pixeles")
    if max_pixels and X.shape[0] > max_pixels:
        idx = np.random.RandomState(seed).choice(X.shape[0], max_pixels, replace=False)
        X, y = X[idx], y[idx]
        print(f"  -> Submuestreo a {max_pixels:,} pixeles")
    return X, y


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, required=True, help='Seed (42, 123 o 7)')
    args = parser.parse_args()

    SEED = args.seed
    EXP_NAME    = f'rf_v6_s{SEED}'
    RESULTS_DIR = str(_REPO / f'results/{EXP_NAME}')
    os.makedirs(RESULTS_DIR, exist_ok=True)

    print(f"\nRF 22ch | seed={SEED} | params Optuna v6 (Trial 90)")
    print(f"Params: {BEST_PARAMS}\n")

    print("Cargando dataset_v4_ms_sx200...")
    TRAIN_DF, VAL_DF, TEST_DF = load_splits(CSV_FILE, source='lidar', split_type='temporal')
    print(f"  Train: {len(TRAIN_DF)} | Val: {len(VAL_DF)} | Test: {len(TEST_DF)} tiles")

    X_train, y_train = load_pixels(TRAIN_DF, 'Train', SEED, max_pixels=MAX_PIXELS)
    X_val,   y_val   = load_pixels(VAL_DF,   'Val',   SEED)
    X_test,  y_test  = load_pixels(TEST_DF,  'Test',  SEED)

    X_train = normalize(X_train)
    X_val   = normalize(X_val)
    X_test  = normalize(X_test)

    # Entrenar en train+val (metodologia paper)
    print("\nEntrenando en train+val...")
    X_tv = np.concatenate([X_train, X_val])
    y_tv = np.concatenate([y_train, y_val])
    rf = RandomForestRegressor(
        **BEST_PARAMS,
        n_jobs=-1,
        random_state=SEED,
    )
    t0 = time.time()
    rf.fit(X_tv, y_tv)
    print(f"  Entrenamiento: {(time.time()-t0)/60:.1f} min")

    # Evaluar en test
    y_pred = np.maximum(rf.predict(X_test), 0)
    metrics = compute_metrics(y_test, y_pred)

    print(f"\n  Test R2    : {metrics['R2']:.4f}")
    print(f"  Test MAE   : {metrics['MAE']:.4f}")
    print(f"  Test RMSE  : {metrics['RMSE']:.4f}")
    print(f"  Test SPAEF : {metrics.get('SPAEF', float('nan')):.4f}")
    print(f"  Test Bias  : {metrics['Bias']:.4f}")

    out = {
        'experiment': EXP_NAME,
        'seed': SEED,
        'best_params': BEST_PARAMS,
        'test_metrics': {k: float(v) for k, v in metrics.items()
                         if not isinstance(v, dict)},
    }
    out_path = os.path.join(RESULTS_DIR, f'{EXP_NAME}_metrics.json')
    with open(out_path, 'w') as f:
        json.dump(out, f, indent=2)
    print(f"\nGuardado en: {out_path}")


if __name__ == '__main__':
    main()
