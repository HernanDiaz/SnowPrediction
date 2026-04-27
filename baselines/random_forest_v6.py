"""
Baseline: Random Forest con dataset v6 (17 canales: topo + SCE + Sx + persistencia).

Canales seleccionados de los .npy de 33 canales:
  [0]  DEM         - Elevacion (normalizado: (dem-2100)/1000)
  [1]  Slope       - Pendiente (/ 90)
  [2]  Northness   - cos(aspect), ya en [-1,1]
  [3]  Eastness    - sin(aspect), ya en [-1,1]
  [4]  TPI         - Posicion topografica (clip / 9200)
  [5]  SCE         - Snow Cover Extent (binarizado: >5 -> 1)
  [6]  Sx_0        - Wind Shelter Index, azimut   0deg, radio 100m (/ 90)
  [7]  Sx_45       - Wind Shelter Index, azimut  45deg, radio 100m (/ 90)
  [8]  Sx_90       - Wind Shelter Index, azimut  90deg, radio 100m (/ 90)
  [9]  Sx_135      - Wind Shelter Index, azimut 135deg, radio 100m (/ 90)
  [10] Sx_180      - Wind Shelter Index, azimut 180deg, radio 100m (/ 90)
  [11] Sx_225      - Wind Shelter Index, azimut 225deg, radio 100m (/ 90)
  [12] Sx_270      - Wind Shelter Index, azimut 270deg, radio 100m (/ 90)
  [13] Sx_315      - Wind Shelter Index, azimut 315deg, radio 100m (/ 90)
  [30] Pers_15d    - Fraccion dias nevados ultimos 15 dias (ya en [0,1])
  [31] Pers_30d    - Fraccion dias nevados ultimos 30 dias (ya en [0,1])
  [32] Pers_60d    - Fraccion dias nevados ultimos 60 dias (ya en [0,1])

Los canales 14-29 son ceros (bug dataset v6) y se omiten.

Uso:
    python baselines/random_forest_v6.py --config configs/attention_unet_v6_5m.yaml
"""

import os
import sys
import argparse
import yaml
import numpy as np
import joblib
from sklearn.ensemble import RandomForestRegressor

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data.dataset import load_splits
from utils.metrics import compute_metrics, compute_naive_benchmark, print_metrics


# -------------------------------------------------------------------------
# Configuracion de canales v6
# -------------------------------------------------------------------------
CHANNEL_INDICES = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 30, 31, 32]

CHANNEL_NAMES = [
    'DEM', 'Slope', 'Northness', 'Eastness', 'TPI', 'SCE',
    'Sx_0', 'Sx_45', 'Sx_90', 'Sx_135',
    'Sx_180', 'Sx_225', 'Sx_270', 'Sx_315',
    'Pers_15d', 'Pers_30d', 'Pers_60d'
]

NORM = {
    'dem_mean': 2100.0,
    'dem_std':  1000.0,
    'slope_max':  90.0,
    'tpi_max':  9200.0,
    'sx_max':     90.0,
}


def normalize_v6(image: np.ndarray) -> np.ndarray:
    """
    Selecciona y normaliza los 17 canales utiles del dataset v6.
    Identica a SnowDataset._normalize con n_channels=17 y channel_indices activos.
    """
    image = image.copy().astype(np.float32)

    # Seleccionar canales utiles (omite los ceros 14-29)
    image = image[CHANNEL_INDICES]   # (17, H, W)

    # Limpiar nodata
    image[image == -9999] = 0
    image = np.nan_to_num(image, nan=0.0, posinf=0.0, neginf=0.0)

    # Normalizacion canal a canal
    image[0] = (image[0] - NORM['dem_mean']) / NORM['dem_std']   # DEM
    image[1] = image[1] / NORM['slope_max']                       # Slope
    # [2] Northness y [3] Eastness ya en [-1,1]
    image[4] = np.clip(image[4] / NORM['tpi_max'], -1.0, 1.0)    # TPI
    image[5] = (image[5] > 5).astype(np.float32)                  # SCE binario
    image[6:14] = np.clip(image[6:14] / NORM['sx_max'], -1.0, 1.0)  # Sx x8
    # [14-16] Persistencia ya en [0,1]

    return image


def load_pixels(df, images_dir: str, masks_dir: str,
                max_pixels: int = None, split_name: str = '') -> tuple:
    """
    Carga tiles del dataset v6 y extrae pixeles validos como (n_pixels, 17).
    """
    X_list, y_list = [], []
    errors = 0

    for _, row in df.iterrows():
        tile_id   = row['tile_id']
        img_path  = os.path.join(images_dir, tile_id)
        mask_path = os.path.join(masks_dir,  tile_id)

        if not os.path.exists(img_path) or not os.path.exists(mask_path):
            errors += 1
            continue

        image = normalize_v6(np.load(img_path))        # (17, 256, 256)
        mask  = np.load(mask_path).astype(np.float32)  # (256, 256)

        mask[mask <= -100] = np.nan
        mask = np.nan_to_num(mask, nan=0.0)

        valid = mask > 0.01
        if valid.sum() == 0:
            continue

        pixels = image.reshape(17, -1).T   # (65536, 17)
        labels = mask.flatten()

        X_list.append(pixels[valid.flatten()])
        y_list.append(labels[valid.flatten()])

    X = np.concatenate(X_list, axis=0)
    y = np.concatenate(y_list, axis=0)

    print(f"  {split_name}: {len(df)} tiles | {X.shape[0]:,} pixeles validos "
          f"({errors} tiles con error)")

    if max_pixels and X.shape[0] > max_pixels:
        idx = np.random.choice(X.shape[0], size=max_pixels, replace=False)
        X, y = X[idx], y[idx]
        print(f"  -> Submuestreo a {max_pixels:,} pixeles para entrenamiento RF")

    return X, y


def train_rf(X_train: np.ndarray, y_train: np.ndarray,
             n_estimators: int = 200) -> RandomForestRegressor:
    print(f"\nEntrenando Random Forest v6 "
          f"({n_estimators} arboles, {X_train.shape[1]} features)...")
    rf = RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=20,
        min_samples_leaf=10,
        n_jobs=-1,
        random_state=42,
        verbose=1
    )
    rf.fit(X_train, y_train)
    return rf


def evaluate_rf(rf, X_test: np.ndarray, y_test: np.ndarray,
                exp_name: str, results_dir: str) -> dict:
    print("\nEvaluando en Test Set...")
    y_pred = np.maximum(rf.predict(X_test), 0)

    metrics = compute_metrics(y_test, y_pred)
    print_metrics(metrics, title=f"Random Forest v6 - {exp_name}")

    print("\nImportancia de variables (17 canales):")
    importances = rf.feature_importances_
    for name, imp in sorted(zip(CHANNEL_NAMES, importances), key=lambda x: -x[1]):
        bar = '#' * int(imp * 50)
        print(f"  {name:<12} {imp:.4f}  {bar}")

    os.makedirs(results_dir, exist_ok=True)
    results_path = os.path.join(results_dir, f"{exp_name}_rf_v6_metrics.txt")
    with open(results_path, 'w') as f:
        f.write(f"Random Forest v6 - {exp_name}\n")
        f.write("="*40 + "\n")
        f.write(f"Canales: {CHANNEL_NAMES}\n\n")
        for k, v in metrics.items():
            f.write(f"{k}: {v}\n")
        f.write("\nImportancia de variables:\n")
        for name, imp in sorted(zip(CHANNEL_NAMES, importances), key=lambda x: -x[1]):
            f.write(f"  {name}: {imp:.4f}\n")

    print(f"\nResultados guardados en: {results_path}")
    return metrics


def main():
    parser = argparse.ArgumentParser(
        description='Baseline Random Forest v6 - Snow Depth (17 canales)'
    )
    parser.add_argument('--config',       type=str, required=True)
    parser.add_argument('--n_estimators', type=int, default=200)
    parser.add_argument('--max_pixels',   type=int, default=500_000)
    args = parser.parse_args()

    with open(args.config, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    cfg_data   = config['data']
    cfg_output = config['output']
    exp_name   = config['experiment']['name']
    results_dir = cfg_output['results_dir']

    data_root  = cfg_data['root']
    csv_path   = os.path.join(data_root, cfg_data['csv_file'])
    images_dir = os.path.join(data_root, cfg_data['images_dir'])
    masks_dir  = os.path.join(data_root, cfg_data['masks_dir'])

    print(f"\n{'='*60}")
    print(f"  Baseline Random Forest v6 (17 canales)")
    print(f"  Config : {args.config}")
    print(f"  Canales: {len(CHANNEL_NAMES)} ({CHANNEL_NAMES})")
    print(f"  Arboles: {args.n_estimators}")
    print(f"{'='*60}\n")

    train_df, val_df, test_df = load_splits(
        csv_path,
        source=cfg_data['source'],
        split_type=cfg_data['split_type']
    )

    print("\nCargando pixeles de entrenamiento...")
    X_train, y_train = load_pixels(train_df, images_dir, masks_dir,
                                   max_pixels=args.max_pixels, split_name='Train')

    print("\nCargando pixeles de test...")
    X_test, y_test = load_pixels(test_df, images_dir, masks_dir,
                                 max_pixels=None, split_name='Test')

    print("\nCalculando benchmark naive...")
    naive_metrics = compute_naive_benchmark(y_train, y_test)
    print_metrics(naive_metrics, title="Naive (Media Train)")

    rf = train_rf(X_train, y_train, n_estimators=args.n_estimators)
    rf_metrics = evaluate_rf(rf, X_test, y_test, exp_name, results_dir)

    model_path = os.path.join(cfg_output['models_dir'], f"{exp_name}_rf_v6.joblib")
    joblib.dump(rf, model_path)
    print(f"Modelo RF v6 guardado en: {model_path}")

    print(f"\n{'='*60}")
    print(f"  RESUMEN COMPARATIVO")
    print(f"{'='*60}")
    print(f"  {'Modelo':<25} {'MAE':>8} {'RMSE':>8} {'R2':>8}")
    print(f"  {'-'*52}")
    print(f"  {'Naive (Media)':<25} {naive_metrics['MAE']:>8.4f} "
          f"{naive_metrics['RMSE']:>8.4f} {naive_metrics['R2']:>8.4f}")
    print(f"  {'RF v6 (17 canales)':<25} {rf_metrics['MAE']:>8.4f} "
          f"{rf_metrics['RMSE']:>8.4f} {rf_metrics['R2']:>8.4f}")
    print(f"  {'RF v5 ref (5 canales)':<25} {'0.4833':>8} {'0.6343':>8} {'0.2570':>8}")
    print(f"{'='*60}\n")


if __name__ == '__main__':
    main()
