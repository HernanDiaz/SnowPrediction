"""
Calcula SPAEF y MSPAEF para un trial guardado de Optuna.
Uso:
    .venv/Scripts/python.exe scripts/eval_trial_spaef.py \
        --pth "Articulo 1/Models/optuna_resunetpp_meteo/trial_001_b48_r20.2970.pth"
"""

import sys
import argparse
import numpy as np
import torch
from pathlib import Path
from torch.utils.data import DataLoader

PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
sys.path.insert(0, PROJECT_ROOT)

from data.dataset import SnowDatasetEval, load_splits
from models.unet import build_model
from training.train import get_device
from utils.metrics import compute_metrics, compute_spaef, compute_mspaef


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--pth', type=str, required=True, help='Ruta al .pth guardado por Optuna')
    args = parser.parse_args()

    pth_path = Path(PROJECT_ROOT) / args.pth
    ckpt = torch.load(pth_path, map_location='cpu')

    arch        = ckpt.get('architecture', 'resunetpp')
    features    = ckpt['features']
    in_channels = ckpt['in_channels']
    out_channels = ckpt.get('out_channels', 1)
    dropout_p   = ckpt.get('dropout_p', 0.0)
    num_groups  = ckpt.get('num_groups', 8)

    print(f"\nModelo: {arch} | features={features} | in_ch={in_channels} | dropout={dropout_p:.3f}")
    print(f"Trial params: {ckpt.get('params', {})}")

    # Dataset
    arch_key = arch.lower()
    if 'meteo2' in str(pth_path):
        data_root = Path(PROJECT_ROOT) / 'dataset_v4_ms_sx200_meteo2'
        csv_file  = data_root / 'dataset_v4_ms_sx200_meteo2.csv'
    elif 'meteo' in str(pth_path):
        data_root = Path(PROJECT_ROOT) / 'dataset_v4_ms_sx200_meteo'
        csv_file  = data_root / 'dataset_v4_ms_sx200_meteo.csv'
    else:
        data_root = Path(PROJECT_ROOT) / 'dataset_v4_ms_sx200'
        csv_file  = data_root / 'dataset_v4_ms_sx200.csv'

    imgs_dir  = str(data_root / 'images')
    masks_dir = str(data_root / 'masks')

    _, _, test_df = load_splits(str(csv_file), source='lidar', split_type='temporal')

    test_ds = SnowDatasetEval(test_df, imgs_dir, masks_dir, use_sce=False, n_channels=in_channels)
    test_loader = DataLoader(test_ds, batch_size=4, shuffle=False, num_workers=0)

    device = get_device('auto')

    # Construir modelo
    cfg = {
        'model': {
            'architecture': arch,
            'in_channels':  in_channels,
            'out_channels': out_channels,
            'features':     features,
            'dropout_p':    dropout_p,
            'num_groups':   num_groups,
        }
    }
    model = build_model(cfg).to(device)
    model.load_state_dict(ckpt['state_dict'])
    model.eval()

    # Inferencia tile a tile para SPAEF
    all_preds, all_targets = [], []
    spaef_list, mspaef_list = [], []

    with torch.no_grad():
        for images, masks, tile_ids in test_loader:
            outputs = model(images.to(device)).cpu().numpy()  # (B,1,H,W)
            targets = masks.numpy()                            # (B,1,H,W)

            for i in range(len(tile_ids)):
                tgt = targets[i, 0].flatten()
                pred = outputs[i, 0].flatten()
                valid = tgt > 0.01
                if valid.sum() < 10:
                    continue
                t_v = tgt[valid]
                p_v = np.maximum(pred[valid], 0.0)
                all_targets.extend(t_v.tolist())
                all_preds.extend(p_v.tolist())
                spaef_list.append(compute_spaef(t_v, p_v))
                mspaef_list.append(compute_mspaef(t_v, p_v))

    all_targets = np.array(all_targets)
    all_preds   = np.array(all_preds)

    metrics = compute_metrics(all_targets, all_preds)
    spaef_vals  = [v for v in spaef_list  if not np.isnan(v)]
    mspaef_vals = [v for v in mspaef_list if not np.isnan(v)]

    print(f"\n{'='*55}")
    print(f"  Resultados: {pth_path.name}")
    print(f"{'='*55}")
    print(f"  R2         : {metrics['R2']:.4f}")
    print(f"  MAE        : {metrics['MAE']:.4f} m")
    print(f"  RMSE       : {metrics['RMSE']:.4f} m")
    print(f"  Bias       : {metrics['Bias']:.4f} m")
    print(f"  SPAEF      : {np.mean(spaef_vals):.4f}  (std={np.std(spaef_vals):.4f}, n={len(spaef_vals)})")
    print(f"  MSPAEF     : {np.mean(mspaef_vals):.4f}  (std={np.std(mspaef_vals):.4f}, n={len(mspaef_vals)})")
    print(f"{'='*55}\n")


if __name__ == '__main__':
    main()
