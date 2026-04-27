"""
Baseline: Random Forest para prediccion de profundidad de nieve.

Funciona pixel a pixel: cada pixel es una muestra con sus N canales como features.
Se entrenan dos versiones:
  - RF sin SCE (5 canales topograficos)
  - RF con SCE (5 topograficos + 1 satelite)

Uso:
    python baselines/random_forest.py --config configs/unet_v5_5m_mae.yaml
    python baselines/random_forest.py --config configs/unet_v5_5m_mae_sce.yaml
"""

import os
import sys
import argparse
import yaml
import numpy as np
import joblib
from sklearn.ensemble import RandomForestRegressor

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data.dataset import load_splits, SnowDataset
from utils.metrics import compute_metrics, compute_naive_benchmark, print_metrics


# -------------------------------------------------------------------------
# Constantes de normalizacion (mismas que SnowDataset)
# -------------------------------------------------------------------------
NORM = {
    'dem_mean': 2100.0,
    'dem_std':  1000.0,
    'slope_max':  90.0,
    'tpi_max':  9200.0,
}

CHANNEL_NAMES = ['DEM', 'Slope', 'Northness', 'Eastness', 'TPI', 'SCE']


def normalize(image: np.ndarray, use_sce: bool) -> np.ndarray:
    """Misma normalizacion que SnowDataset._normalize."""
    image = image.copy().astype(np.float32)
    n_ch = 6 if use_sce else 5
    image = image[:n_ch]

    image[image == -9999] = 0
    image = np.nan_to_num(image, nan=0.0, posinf=0.0, neginf=0.0)

    image[0] = (image[0] - NORM['dem_mean']) / NORM['dem_std']
    image[1] = image[1] / NORM['slope_max']
    image[4] = np.clip(image[4] / NORM['tpi_max'], -1.0, 1.0)
    if use_sce:
        image[5] = (image[5] > 5).astype(np.float32)

    return image


def load_pixels(df, images_dir: str, masks_dir: str,
                use_sce: bool, max_pixels: int = None,
                split_name: str = '') -> tuple:
    """
    Carga todos los tiles y extrae pixeles validos como filas (n_pixels, n_features).

    Args:
        df:          DataFrame con metadatos de tiles
        images_dir:  Directorio de imagenes .npy
        masks_dir:   Directorio de mascaras .npy
        use_sce:     Si True, usa 6 canales
        max_pixels:  Limite de pixeles (submuestreo aleatorio para RF)
        split_name:  Nombre del split para log

    Returns:
        X (n_pixels, n_features), y (n_pixels,)
    """
    X_list, y_list = [], []
    n_ch = 6 if use_sce else 5
    errors = 0

    for _, row in df.iterrows():
        tile_id = row['tile_id']
        img_path  = os.path.join(images_dir, tile_id)
        mask_path = os.path.join(masks_dir,  tile_id)

        if not os.path.exists(img_path) or not os.path.exists(mask_path):
            errors += 1
            continue

        image = normalize(np.load(img_path), use_sce)   # (n_ch, 256, 256)
        mask  = np.load(mask_path).astype(np.float32)    # (256, 256)

        mask[mask <= -100] = np.nan
        mask = np.nan_to_num(mask, nan=0.0)

        # Pixeles validos: nieve real > 0
        valid = mask > 0.01
        if valid.sum() == 0:
            continue

        # Transponer a (H*W, n_ch) y filtrar validos
        pixels = image.reshape(n_ch, -1).T   # (65536, n_ch)
        labels = mask.flatten()              # (65536,)

        X_list.append(pixels[valid.flatten()])
        y_list.append(labels[valid.flatten()])

    X = np.concatenate(X_list, axis=0)
    y = np.concatenate(y_list, axis=0)

    print(f"  {split_name}: {len(df)} tiles | {X.shape[0]:,} pixeles validos "
          f"({errors} tiles con error)")

    # Submuestreo aleatorio si hay demasiados pixeles
    if max_pixels and X.shape[0] > max_pixels:
        idx = np.random.choice(X.shape[0], size=max_pixels, replace=False)
        X, y = X[idx], y[idx]
        print(f"  -> Submuestreo a {max_pixels:,} pixeles para entrenamiento RF")

    return X, y


def train_rf(X_train: np.ndarray, y_train: np.ndarray, n_estimators: int = 200) -> RandomForestRegressor:
    """Entrena un Random Forest con los pixeles de entrenamiento."""
    print(f"\nEntrenando Random Forest ({n_estimators} arboles, {X_train.shape[1]} features)...")
    rf = RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=20,
        min_samples_leaf=10,
        n_jobs=-1,           # Todos los nucleos CPU
        random_state=42,
        verbose=1
    )
    rf.fit(X_train, y_train)
    return rf


def evaluate_rf(rf, X_test: np.ndarray, y_test: np.ndarray, exp_name: str, results_dir: str) -> dict:
    """Evalua el RF y guarda los resultados."""
    print("\nEvaluando en Test Set...")
    y_pred = np.maximum(rf.predict(X_test), 0)

    metrics = compute_metrics(y_test, y_pred)
    print_metrics(metrics, title=f"Random Forest - {exp_name}")

    # Feature importance
    channel_names = CHANNEL_NAMES[:X_test.shape[1]]
    print("\nImportancia de variables:")
    importances = rf.feature_importances_
    for name, imp in sorted(zip(channel_names, importances), key=lambda x: -x[1]):
        bar = '#' * int(imp * 50)
        print(f"  {name:<12} {imp:.4f}  {bar}")

    # Guardar resultados
    os.makedirs(results_dir, exist_ok=True)
    results_path = os.path.join(results_dir, f"{exp_name}_rf_metrics.txt")
    with open(results_path, 'w') as f:
        f.write(f"Random Forest - {exp_name}\n")
        f.write("="*40 + "\n")
        for k, v in metrics.items():
            f.write(f"{k}: {v}\n")
        f.write("\nImportancia de variables:\n")
        for name, imp in sorted(zip(channel_names, importances), key=lambda x: -x[1]):
            f.write(f"  {name}: {imp:.4f}\n")

    print(f"\nResultados guardados en: {results_path}")
    return metrics


def main():
    parser = argparse.ArgumentParser(description='Baseline Random Forest - Snow Depth')
    parser.add_argument('--config',       type=str, required=True,  help='Config YAML del experimento')
    parser.add_argument('--n_estimators', type=int, default=200,    help='Numero de arboles RF')
    parser.add_argument('--max_pixels',   type=int, default=500_000, help='Max pixeles de entrenamiento')
    args = parser.parse_args()

    with open(args.config, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    cfg_data   = config['data']
    cfg_output = config['output']
    use_sce    = cfg_data.get('use_sce', False)
    exp_name   = config['experiment']['name'] + '_rf'
    results_dir = cfg_output['results_dir']

    data_root  = cfg_data['root']
    csv_path   = os.path.join(data_root, cfg_data['csv_file'])
    images_dir = os.path.join(data_root, cfg_data['images_dir'])
    masks_dir  = os.path.join(data_root, cfg_data['masks_dir'])

    print(f"\n{'='*60}")
    print(f"  Baseline Random Forest")
    print(f"  Config : {args.config}")
    print(f"  SCE    : {'SI' if use_sce else 'NO'}")
    print(f"  Arboles: {args.n_estimators}")
    print(f"{'='*60}\n")

    # Cargar splits
    train_df, val_df, test_df = load_splits(
        csv_path,
        source=cfg_data['source'],
        split_type=cfg_data['split_type']
    )

    # Cargar pixeles
    print("\nCargando pixeles de entrenamiento...")
    X_train, y_train = load_pixels(train_df, images_dir, masks_dir, use_sce,
                                   max_pixels=args.max_pixels, split_name='Train')

    print("\nCargando pixeles de test...")
    X_test,  y_test  = load_pixels(test_df,  images_dir, masks_dir, use_sce,
                                   max_pixels=None, split_name='Test')

    # Benchmark naive
    print("\nCalculando benchmark naive...")
    naive_metrics = compute_naive_benchmark(y_train, y_test)
    print_metrics(naive_metrics, title="Naive (Media Train)")

    # Entrenar RF
    rf = train_rf(X_train, y_train, n_estimators=args.n_estimators)

    # Evaluar
    rf_metrics = evaluate_rf(rf, X_test, y_test, exp_name, results_dir)

    # Guardar modelo RF
    model_path = os.path.join(cfg_output['models_dir'], f"{exp_name}.joblib")
    joblib.dump(rf, model_path)
    print(f"Modelo RF guardado en: {model_path}")

    # Resumen comparativo
    print(f"\n{'='*60}")
    print(f"  RESUMEN COMPARATIVO")
    print(f"{'='*60}")
    print(f"  {'Modelo':<20} {'MAE':>8} {'RMSE':>8} {'R2':>8} {'NSE':>8}")
    print(f"  {'-'*52}")
    print(f"  {'Naive (Media)':<20} {naive_metrics['MAE']:>8.4f} {naive_metrics['RMSE']:>8.4f} "
          f"{naive_metrics['R2']:>8.4f} {naive_metrics['NSE']:>8.4f}")
    print(f"  {'Random Forest':<20} {rf_metrics['MAE']:>8.4f} {rf_metrics['RMSE']:>8.4f} "
          f"{rf_metrics['R2']:>8.4f} {rf_metrics['NSE']:>8.4f}")
    print(f"{'='*60}\n")


if __name__ == '__main__':
    main()
