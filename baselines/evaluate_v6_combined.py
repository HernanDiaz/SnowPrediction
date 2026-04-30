"""
Evalua el modelo ResUNet++ v6_improved sobre val (2024) + test (2025) combinados.
===================================================================================

El dataset v6_improved tiene test=2025 que solo genera ~6 tiles (ficheros LiDAR
pequenos ese anno). Este script evalua sobre la union val+test para tener mas
tiles evaluados, especialmente relevante para el SPAEF que necesita suficientes
muestras por tile.

Metricas calculadas:
  - R2, RMSE, MAE, NSE, Bias  (sobre todos los pixeles validos)
  - SPAEF por tile + media/std  (sobre cada tile individualmente)

Uso:
    .venv\\Scripts\\python.exe baselines/evaluate_v6_combined.py
"""

import sys
import json
import numpy as np
import pandas as pd
import torch
from pathlib import Path
from torch.utils.data import DataLoader
from tqdm import tqdm

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO))

from data.dataset import SnowDatasetEval, load_splits
from models.unet import build_model
from training.train import get_device
from utils.metrics import compute_metrics, compute_spaef, print_metrics
import yaml

# ---------------------------------------------------------------------------
# Configuracion
# ---------------------------------------------------------------------------
CONFIG_PATH = _REPO / "configs/resunetpp_v6_improved.yaml"
OUT_PATH    = _REPO / "results/resunetpp_v6_improved/resunetpp_v6_improved_metrics_combined.json"


def load_config(path):
    with open(path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def resolve(config, key_path, base):
    """Resuelve rutas relativas desde la raiz del repo."""
    parts = key_path.split('.')
    d = config
    for p in parts[:-1]:
        d = d[p]
    val = d[parts[-1]]
    p = Path(val)
    return str(p if p.is_absolute() else base / p)


def main():
    config = load_config(CONFIG_PATH)

    # Resolver rutas
    data_root  = str(_REPO / config['data']['root'])
    csv_path   = str(Path(data_root) / config['data']['csv_file'])
    imgs_dir   = str(Path(data_root) / config['data']['images_dir'])
    masks_dir  = str(Path(data_root) / config['data']['masks_dir'])
    model_path = str(_REPO / config['output']['models_dir']
                     / f"{config['output']['model_name']}.pth")

    print(f"Config     : {CONFIG_PATH}")
    print(f"Modelo     : {model_path}")
    print(f"CSV        : {csv_path}")

    # Cargar CSV y combinar val + test
    df = pd.read_csv(csv_path)
    df_eval = df[
        (df['source'] == 'lidar') &
        (df['exp_temporal_split'].isin(['val', 'test']))
    ].reset_index(drop=True)

    print(f"\nTiles para evaluacion combinada (val+test):")
    for sp, cnt in df_eval['exp_temporal_split'].value_counts().items():
        print(f"  {sp}: {cnt}")
    print(f"  Total: {len(df_eval)}")

    # Dataset
    eval_ds = SnowDatasetEval(df_eval, imgs_dir, masks_dir,
                               use_sce=False,
                               n_channels=config['model']['in_channels'])
    eval_loader = DataLoader(
        eval_ds,
        batch_size=config['training']['batch_size'],
        shuffle=False,
        num_workers=0,
    )

    # Cargar modelo
    device = get_device(config['training']['device'])
    model  = build_model(config).to(device)

    if not Path(model_path).exists():
        print(f"\nERROR: Modelo no encontrado: {model_path}")
        sys.exit(1)

    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    print(f"\nPesos cargados desde: {model_path}")

    # Prediccion
    all_preds, all_targets = [], []
    spaef_per_tile = []

    with torch.no_grad():
        for batch in tqdm(eval_loader, desc="  Evaluando val+test"):
            if len(batch) == 3:
                images, masks, ids = batch
            else:
                images, masks = batch

            outputs = model(images.to(device)).cpu().numpy()
            targets = masks.cpu().numpy()

            tgt_flat = targets.squeeze(1).flatten()
            out_flat = outputs.squeeze(1).flatten()
            valid    = tgt_flat > 0.01

            if valid.sum() > 0:
                all_preds.extend(out_flat[valid].tolist())
                all_targets.extend(tgt_flat[valid].tolist())

            for b in range(targets.shape[0]):
                tgt_tile = targets[b, 0].flatten()
                out_tile = outputs[b, 0].flatten()
                v = tgt_tile > 0.01
                if v.sum() >= 10:
                    spaef_val = compute_spaef(tgt_tile[v], out_tile[v])
                    if not np.isnan(spaef_val):
                        spaef_per_tile.append(spaef_val)

    y_pred = np.array(all_preds)
    y_true = np.array(all_targets)

    metrics = compute_metrics(y_true, y_pred)

    if spaef_per_tile:
        metrics['SPAEF']         = round(float(np.mean(spaef_per_tile)), 4)
        metrics['SPAEF_std']     = round(float(np.std(spaef_per_tile)),  4)
        metrics['SPAEF_n_tiles'] = len(spaef_per_tile)
    else:
        metrics['SPAEF'] = float('nan')

    print_metrics(metrics, title="ResUNet++ v6_improved — val+test (2024+2025)")

    # Guardar
    out = {
        'experiment': 'resunetpp_v6_improved_combined',
        'eval_splits': ['val', 'test'],
        'n_pixels': int(len(y_true)),
        **metrics,
    }
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_PATH, 'w') as f:
        json.dump(out, f, indent=2)
    print(f"\nGuardado en: {OUT_PATH}")


if __name__ == '__main__':
    main()
