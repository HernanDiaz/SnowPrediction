"""
Busqueda de hiperparametros con Optuna - Random Forest / Dataset v6 (17 canales).
==================================================================================

Canales: DEM, Slope, Northness, Eastness, TPI, SCE, Sx x8, Pers_15d, Pers_30d, Pers_60d

Espacio de busqueda:
  - n_estimators   : 100, 200, 300, 500
  - max_depth      : 10, 15, 20, 30, None
  - min_samples_leaf : 1, 5, 10, 20, 50
  - max_features   : "sqrt", "log2", 0.3, 0.5
  - min_samples_split : 2, 5, 10

Metrica objetivo: R2 en validacion (2023).
Al finalizar: reentrena con mejores params en train+val y evalua en test (2024-2025).

Salidas:
  - Mejor modelo    : results/optuna_rf_v6/rf_v6_best.joblib
  - Ranking JSON    : results/optuna_rf_v6/ranking_rf_v6.json
  - Metricas test   : results/optuna_rf_v6/rf_v6_test_metrics.json
  - BD Optuna       : results/optuna_rf_v6/optuna_rf_v6.db
"""

import os
import sys
import json
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import joblib
import optuna
from optuna.samplers import TPESampler
from sklearn.ensemble import RandomForestRegressor

sys.path.insert(0, 'E:/PycharmProjects/SnowPrediction')
from data.dataset import load_splits
from utils.metrics import compute_metrics

# ---------------------------------------------------------------------------
# Configuracion fija
# ---------------------------------------------------------------------------
DATA_ROOT   = 'E:/PycharmProjects/SnowPrediction/Articulo 1/Data/processed/dataset_v6_5m'
CSV_FILE    = os.path.join(DATA_ROOT, 'dataset_v6_fisico.csv')
IMGS_DIR    = os.path.join(DATA_ROOT, 'images')
MASKS_DIR   = os.path.join(DATA_ROOT, 'masks')
RESULTS_DIR = 'E:/PycharmProjects/SnowPrediction/results/optuna_rf_v6'
DB_PATH     = f'sqlite:///{RESULTS_DIR}/optuna_rf_v6.db'
STUDY_NAME  = 'rf_v6_hpo_v1'
N_TRIALS    = 80
MAX_PIXELS  = 500_000
SEED        = 42

CHANNEL_INDICES = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 30, 31, 32]
CHANNEL_NAMES   = [
    'DEM', 'Slope', 'Northness', 'Eastness', 'TPI', 'SCE',
    'Sx_0', 'Sx_45', 'Sx_90', 'Sx_135', 'Sx_180', 'Sx_225', 'Sx_270', 'Sx_315',
    'Pers_15d', 'Pers_30d', 'Pers_60d'
]
NORM = {'dem_mean': 2100.0, 'dem_std': 1000.0, 'slope_max': 90.0,
        'tpi_max': 9200.0, 'sx_max': 90.0}

os.makedirs(RESULTS_DIR, exist_ok=True)


def normalize_v6(image):
    image = image.copy().astype(np.float32)
    image = image[CHANNEL_INDICES]   # (17, H, W)
    image[image == -9999] = 0
    image = np.nan_to_num(image, nan=0.0, posinf=0.0, neginf=0.0)
    image[0] = (image[0] - NORM['dem_mean']) / NORM['dem_std']
    image[1] = image[1] / NORM['slope_max']
    image[4] = np.clip(image[4] / NORM['tpi_max'], -1.0, 1.0)
    image[5] = (image[5] > 5).astype(np.float32)
    image[6:14] = np.clip(image[6:14] / NORM['sx_max'], -1.0, 1.0)
    # Pers [14:17] ya en [0,1]
    return image


def load_pixels(df, split_name='', max_pixels=None):
    X_list, y_list = [], []
    for _, row in df.iterrows():
        tid  = row['tile_id']
        img  = os.path.join(IMGS_DIR,  tid)
        mask = os.path.join(MASKS_DIR, tid)
        if not os.path.exists(img) or not os.path.exists(mask):
            continue
        image = normalize_v6(np.load(img))   # (17, H, W)
        m     = np.load(mask).astype(np.float32)
        m[m <= -100] = np.nan
        m = np.nan_to_num(m, nan=0.0)
        valid = m > 0.01
        if valid.sum() == 0:
            continue
        X_list.append(image.reshape(17, -1).T[valid.flatten()])
        y_list.append(m.flatten()[valid.flatten()])
    X = np.concatenate(X_list)
    y = np.concatenate(y_list)
    print(f"  {split_name}: {X.shape[0]:,} pixeles validos")
    if max_pixels and X.shape[0] > max_pixels:
        idx = np.random.RandomState(SEED).choice(X.shape[0], size=max_pixels, replace=False)
        X, y = X[idx], y[idx]
        print(f"  -> Submuestreo a {max_pixels:,} pixeles")
    return X, y


# Cargar datos una sola vez
print("\nCargando datos v6...")
TRAIN_DF, VAL_DF, TEST_DF = load_splits(CSV_FILE, source='lidar', split_type='temporal')
print(f"  Train: {len(TRAIN_DF)} | Val: {len(VAL_DF)} | Test: {len(TEST_DF)} tiles")

X_train, y_train = load_pixels(TRAIN_DF, 'Train', max_pixels=MAX_PIXELS)
X_val,   y_val   = load_pixels(VAL_DF,   'Val')
X_test,  y_test  = load_pixels(TEST_DF,  'Test')


def objective(trial):
    n_est  = trial.suggest_categorical('n_estimators',      [100, 200, 300, 500])
    depth  = trial.suggest_categorical('max_depth',         [10, 15, 20, 30, 'None'])
    msl    = trial.suggest_categorical('min_samples_leaf',  [1, 5, 10, 20, 50])
    mf_raw = trial.suggest_categorical('max_features',      ['sqrt', 'log2', '0.3', '0.5'])
    mss    = trial.suggest_categorical('min_samples_split', [2, 5, 10])

    max_depth    = None if depth == 'None' else int(depth)
    max_features = float(mf_raw) if mf_raw not in ('sqrt', 'log2') else mf_raw

    rf = RandomForestRegressor(
        n_estimators=n_est,
        max_depth=max_depth,
        min_samples_leaf=msl,
        max_features=max_features,
        min_samples_split=mss,
        n_jobs=-1,
        random_state=SEED,
    )
    rf.fit(X_train, y_train)
    y_pred_val = np.maximum(rf.predict(X_val), 0)
    metrics = compute_metrics(y_val, y_pred_val)
    val_r2  = metrics['R2']

    print(f"  Trial {trial.number:03d} | n_est={n_est} depth={depth} msl={msl} "
          f"mf={mf_raw} mss={mss} | val_R2={val_r2:.4f}")
    return val_r2


if __name__ == '__main__':
    print(f"\n{'='*60}")
    print(f"  Optuna RF v6  |  {N_TRIALS} trials")
    print(f"{'='*60}\n")

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    sampler = TPESampler(seed=SEED)
    study = optuna.create_study(
        study_name=STUDY_NAME,
        storage=DB_PATH,
        direction='maximize',
        sampler=sampler,
        load_if_exists=True,
    )
    study.optimize(objective, n_trials=N_TRIALS, show_progress_bar=False)

    best = study.best_trial
    print(f"\n{'='*60}")
    print(f"  MEJOR TRIAL: #{best.number} | val_R2 = {best.value:.4f}")
    print(f"  Params: {best.params}")
    print(f"{'='*60}")

    # Reentrenar con mejores params en train completo y evaluar en test
    print("\nReentrenando con mejores params en train completo...")
    depth_final = None if best.params['max_depth'] == 'None' else int(best.params['max_depth'])
    mf_final    = (float(best.params['max_features'])
                   if best.params['max_features'] not in ('sqrt', 'log2')
                   else best.params['max_features'])

    rf_final = RandomForestRegressor(
        n_estimators=best.params['n_estimators'],
        max_depth=depth_final,
        min_samples_leaf=best.params['min_samples_leaf'],
        max_features=mf_final,
        min_samples_split=best.params['min_samples_split'],
        n_jobs=-1,
        random_state=SEED,
    )
    X_trainval = np.concatenate([X_train, X_val])
    y_trainval = np.concatenate([y_train, y_val])
    rf_final.fit(X_trainval, y_trainval)

    # Evaluar en test
    y_pred_test = np.maximum(rf_final.predict(X_test), 0)
    test_metrics = compute_metrics(y_test, y_pred_test)
    print(f"\n  Test R2:   {test_metrics['R2']:.4f}")
    print(f"  Test MAE:  {test_metrics['MAE']:.4f}")
    print(f"  Test RMSE: {test_metrics['RMSE']:.4f}")

    # Feature importance
    importances = rf_final.feature_importances_
    feat_imp = {n: round(float(v), 4) for n, v in zip(CHANNEL_NAMES, importances)}
    print(f"\n  Feature importance:")
    for k, v in sorted(feat_imp.items(), key=lambda x: -x[1]):
        print(f"    {k:<12} {v:.4f}")

    # Guardar modelo
    model_path = os.path.join(RESULTS_DIR, 'rf_v6_best.joblib')
    joblib.dump(rf_final, model_path)
    print(f"\n  Modelo guardado: {model_path}")

    # Guardar metricas
    result = {
        'experiment': 'rf_v6_optuna',
        'best_trial': best.number,
        'val_R2': round(best.value, 4),
        'best_params': best.params,
        'test_metrics': {k: round(float(v), 4) for k, v in test_metrics.items()},
        'feature_importance': feat_imp,
    }
    metrics_path = os.path.join(RESULTS_DIR, 'rf_v6_test_metrics.json')
    with open(metrics_path, 'w') as f:
        json.dump(result, f, indent=2)
    print(f"  Metricas guardadas: {metrics_path}")

    # Ranking de todos los trials
    ranking = sorted(
        [{'trial': t.number, 'val_R2': round(t.value, 4), 'params': t.params}
         for t in study.trials if t.state.name == 'COMPLETE'],
        key=lambda x: -x['val_R2']
    )
    ranking_path = os.path.join(RESULTS_DIR, 'ranking_rf_v6.json')
    with open(ranking_path, 'w') as f:
        json.dump(ranking, f, indent=2)
    print(f"  Ranking guardado:  {ranking_path}")

    print(f"\nOptuna RF v6 completado.")
