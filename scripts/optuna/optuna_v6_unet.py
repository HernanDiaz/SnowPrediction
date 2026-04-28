"""
Busqueda de hiperparametros con Optuna - UNet / Dataset v6 (17 canales).
=========================================================================

Arquitectura fija: UNet
Dataset v6: 6 topo+SCE + 8 Sx_100m + 3 persistencia nival
  channel_indices = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,30,31,32]

Espacio de busqueda:
  - base_channels : 32 | 48 | 64
  - dropout_p     : [0.0, 0.20]
  - lr            : log-uniform [3e-5, 5e-4]
  - weight_decay  : log-uniform [5e-6, 6e-4]
  - batch_size    : 8 | 16
  - optimizer     : adam | adamw
  - grad_clip     : 0.0 | 1.0

Fijado: loss=MSE, epochs=150, in_channels=17, sin augmentation.
Pruning: MedianPruner (warm-up 10 trials, reporte cada 10 epocas desde ep20).
Storage: SQLite (resumible si se interrumpe).

Salidas:
  - Pesos por trial: Articulo 1/Models/optuna_v6_unet/trial_NNN_b<base>_r2X.XXXX.pth
  - Ranking JSON:    results/optuna_v6_unet/ranking_unet_v6.json
  - BD Optuna:       results/optuna_v6_unet/optuna_unet_v6.db

Uso:
    python optuna_v6_unet.py               # 60 trials
    python optuna_v6_unet.py --trials 40   # N trials
    python optuna_v6_unet.py --resume      # Reanudar
"""

import os
import sys
import copy
import argparse
import json
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import optuna
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler

PROJECT_ROOT = str(__import__('pathlib').Path(__file__).resolve().parent.parent.parent)
sys.path.insert(0, PROJECT_ROOT)

from data.dataset import SnowDataset, SnowDatasetEval, load_splits
from models.unet import UNet
from training.train import get_device
from utils.metrics import compute_metrics

# -- Configuracion fija --------------------------------------------------------
DATA_ROOT   = str(__import__('pathlib').Path(PROJECT_ROOT) / 'Articulo 1/Data/processed/dataset_v6_5m')
CSV_FILE    = os.path.join(DATA_ROOT, 'dataset_v6_fisico.csv')
IMGS_DIR    = os.path.join(DATA_ROOT, 'images')
MASKS_DIR   = os.path.join(DATA_ROOT, 'masks')
MODELS_DIR  = str(__import__('pathlib').Path(PROJECT_ROOT) / 'Articulo 1/Models/optuna_v6_unet')
RESULTS_DIR = str(__import__('pathlib').Path(PROJECT_ROOT) / 'results/optuna_v6_unet')
DB_PATH     = f'sqlite:///{RESULTS_DIR}/optuna_unet_v6.db'
STUDY_NAME  = 'unet_v6_hpo_v1'

EPOCHS          = 150
IN_CH           = 17
OUT_CH          = 1
SEED            = 42
PRUNE_AFTER     = 20
REPORT_EVERY    = 10
CHANNEL_INDICES = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 30, 31, 32]

os.makedirs(MODELS_DIR,  exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

DEVICE = get_device('auto')

print("\nCargando splits de datos v6 (UNet)...")
TRAIN_DF, VAL_DF, TEST_DF = load_splits(CSV_FILE, source='lidar', split_type='temporal')
print(f"  Train: {len(TRAIN_DF)} | Val: {len(VAL_DF)} | Test: {len(TEST_DF)} tiles\n")


def compute_test_metrics(model, test_loader):
    model.eval()
    all_preds, all_targets = [], []
    with torch.no_grad():
        for batch in test_loader:
            images, masks = batch[0], batch[1]
            outputs = model(images.to(DEVICE)).cpu().numpy()
            targets = masks.numpy()
            tgt_flat = targets.squeeze(1).flatten()
            out_flat = outputs.squeeze(1).flatten()
            valid = tgt_flat > 0.01
            if valid.sum() > 0:
                all_preds.extend(out_flat[valid].tolist())
                all_targets.extend(tgt_flat[valid].tolist())
    return compute_metrics(np.array(all_targets), np.array(all_preds))


def _make_loaders(bs):
    train_ds = SnowDataset(TRAIN_DF, IMGS_DIR, MASKS_DIR,
                           use_sce=False, augment=False, n_channels=IN_CH)
    train_ds.channel_indices = CHANNEL_INDICES
    val_ds = SnowDataset(VAL_DF, IMGS_DIR, MASKS_DIR,
                         use_sce=False, augment=False, n_channels=IN_CH)
    val_ds.channel_indices = CHANNEL_INDICES
    test_ds = SnowDatasetEval(TEST_DF, IMGS_DIR, MASKS_DIR,
                              use_sce=False, n_channels=IN_CH)
    test_ds.channel_indices = CHANNEL_INDICES

    return (
        DataLoader(train_ds, batch_size=bs, shuffle=True,  num_workers=0, pin_memory=True),
        DataLoader(val_ds,   batch_size=bs, shuffle=False, num_workers=0, pin_memory=True),
        DataLoader(test_ds,  batch_size=bs, shuffle=False, num_workers=0),
    )


def objective(trial):
    base_ch   = trial.suggest_categorical('base_channels', [32, 48, 64])
    dropout_p = trial.suggest_float('dropout_p',    0.0,  0.20)
    lr        = trial.suggest_float('lr',           3e-5, 5e-4, log=True)
    wd        = trial.suggest_float('weight_decay', 5e-6, 6e-4, log=True)
    bs        = trial.suggest_categorical('batch_size', [8, 16])
    opt_name  = trial.suggest_categorical('optimizer',   ['adam', 'adamw'])
    grad_clip = trial.suggest_categorical('grad_clip',   [0.0, 1.0])

    features = [base_ch, base_ch * 2, base_ch * 4, base_ch * 8]
    if bs == 16 and base_ch >= 64:
        bs = 8

    torch.manual_seed(SEED)
    np.random.seed(SEED)

    train_loader, val_loader, test_loader = _make_loaders(bs)

    model = UNet(in_channels=IN_CH, out_channels=OUT_CH,
                 features=features, dropout_p=dropout_p).to(DEVICE)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    trial.set_user_attr('n_params', n_params)

    criterion = nn.MSELoss()
    optimizer = (optim.AdamW if opt_name == 'adamw' else optim.Adam)(
        model.parameters(), lr=lr, weight_decay=wd)

    print(f"\n{'-'*65}", flush=True)
    print(f"[Trial {trial.number:03d}] UNet | base={base_ch} | "
          f"lr={lr:.1e} | wd={wd:.1e} | bs={bs} | "
          f"opt={opt_name} | gc={grad_clip} | dp={dropout_p:.2f}", flush=True)
    print(f"  features={features} | params={n_params:,}", flush=True)

    best_val_loss    = float('inf')
    best_model_state = None

    for epoch in range(1, EPOCHS + 1):
        model.train()
        for images, masks in train_loader:
            images, masks = images.to(DEVICE), masks.to(DEVICE)
            optimizer.zero_grad()
            loss = criterion(model(images), masks)
            loss.backward()
            if grad_clip > 0:
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()

        model.eval()
        val_total = 0.0
        with torch.no_grad():
            for images, masks in val_loader:
                images, masks = images.to(DEVICE), masks.to(DEVICE)
                val_total += criterion(model(images), masks).item()
        val_loss = val_total / len(val_loader)

        if val_loss < best_val_loss:
            best_val_loss    = val_loss
            best_model_state = copy.deepcopy(model.state_dict())

        if epoch >= PRUNE_AFTER and epoch % REPORT_EVERY == 0:
            trial.report(val_loss, epoch)
            if trial.should_prune():
                print(f"  -> Pruned en epoch {epoch} (val={val_loss:.4f})", flush=True)
                raise optuna.exceptions.TrialPruned()

        if epoch % 50 == 0 or epoch == EPOCHS:
            print(f"  Epoch {epoch:03d}/{EPOCHS} | "
                  f"val={val_loss:.4f} | best={best_val_loss:.4f}", flush=True)

    model.load_state_dict(best_model_state)
    metrics = compute_test_metrics(model, test_loader)

    trial.set_user_attr('test_R2',       metrics['R2'])
    trial.set_user_attr('test_MAE',      metrics['MAE'])
    trial.set_user_attr('test_RMSE',     metrics['RMSE'])
    trial.set_user_attr('test_Bias',     metrics['Bias'])
    trial.set_user_attr('best_val_loss', round(best_val_loss, 6))

    model_filename = f"trial_{trial.number:03d}_b{base_ch}_r2{metrics['R2']:.4f}.pth"
    torch.save({
        'trial':           trial.number,
        'architecture':    'unet',
        'features':        features,
        'in_channels':     IN_CH,
        'out_channels':    OUT_CH,
        'dropout_p':       dropout_p,
        'channel_indices': CHANNEL_INDICES,
        'state_dict':      best_model_state,
        'val_loss':        round(best_val_loss, 6),
        'test_R2':         metrics['R2'],
        'test_MAE':        metrics['MAE'],
        'params':          trial.params,
    }, os.path.join(MODELS_DIR, model_filename))

    print(f"  -> COMPLETO | val_best={best_val_loss:.4f} | "
          f"test_R2={metrics['R2']:.4f} | test_MAE={metrics['MAE']:.4f} m", flush=True)
    return best_val_loss


def print_ranking(study, trial):
    if trial.state != optuna.trial.TrialState.COMPLETE:
        return
    completed = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    ranked = sorted(completed, key=lambda t: t.user_attrs.get('test_R2', -999), reverse=True)

    print(f"\n{'='*65}", flush=True)
    print(f"  RANKING UNet v6 | {len(completed)} trials completados", flush=True)
    print(f"{'='*65}", flush=True)
    print(f"  {'#':<4} {'Trial':<6} {'val_loss':<10} {'R2':<8} {'MAE':<8} "
          f"{'base':<6} {'lr':<8} {'wd':<8} {'bs':<4} {'opt':<6}", flush=True)
    for i, t in enumerate(ranked[:10], 1):
        p = t.params
        print(f"  {i:<4} {t.number:<6} "
              f"{t.user_attrs.get('best_val_loss', t.value):<10.4f} "
              f"{t.user_attrs.get('test_R2', -9):<8.4f} "
              f"{t.user_attrs.get('test_MAE', -9):<8.4f} "
              f"{p.get('base_channels', '?'):<6} "
              f"{p.get('lr', 0):<8.1e} "
              f"{p.get('weight_decay', 0):<8.1e} "
              f"{p.get('batch_size', '?'):<4} "
              f"{p.get('optimizer', '?'):<6}", flush=True)
    print(f"{'='*65}\n", flush=True)

    ranking_data = [
        {'rank': i, 'trial': t.number,
         'val_loss': t.user_attrs.get('best_val_loss', t.value),
         'test_R2': t.user_attrs.get('test_R2'), 'test_MAE': t.user_attrs.get('test_MAE'),
         'test_RMSE': t.user_attrs.get('test_RMSE'), 'test_Bias': t.user_attrs.get('test_Bias'),
         'n_params': t.user_attrs.get('n_params'), 'params': t.params}
        for i, t in enumerate(ranked, 1)
    ]
    with open(os.path.join(RESULTS_DIR, 'ranking_unet_v6.json'), 'w', encoding='utf-8') as f:
        json.dump(ranking_data, f, indent=2)


def main():
    parser = argparse.ArgumentParser(description='Optuna HPO UNet v6 (17 canales)')
    parser.add_argument('--trials',  type=int, default=60)
    parser.add_argument('--resume',  action='store_true')
    parser.add_argument('--timeout', type=int, default=None)
    args = parser.parse_args()

    sampler = TPESampler(seed=SEED, n_startup_trials=10)
    pruner  = MedianPruner(n_startup_trials=10, n_warmup_steps=20, interval_steps=10)

    if args.resume:
        study = optuna.load_study(study_name=STUDY_NAME, storage=DB_PATH,
                                  sampler=sampler, pruner=pruner)
    else:
        study = optuna.create_study(study_name=STUDY_NAME, storage=DB_PATH,
                                    direction='minimize', sampler=sampler,
                                    pruner=pruner, load_if_exists=True)

    n_existing = len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])
    print(f"\nEstudio: {STUDY_NAME}")
    print(f"Arquitectura: UNet | Dataset: v6 (17 canales)")
    print(f"Trials completados: {n_existing} | Nuevos: {args.trials}")
    print(f"Modelos: {MODELS_DIR}")
    print(f"Resultados: {RESULTS_DIR}\n")

    study.optimize(objective, n_trials=args.trials, timeout=args.timeout,
                   callbacks=[print_ranking], catch=(RuntimeError,))

    completed = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    pruned    = [t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED]
    print(f"\n{'='*65}")
    print(f"  UNet v6 HPO COMPLETADO | {len(completed)} completados | {len(pruned)} pruned")
    if completed:
        best = max(completed, key=lambda t: t.user_attrs.get('test_R2', -999))
        print(f"  Mejor test_R2: {best.user_attrs.get('test_R2'):.4f} (Trial #{best.number})")
        print(f"  Params: {best.params}")
    print(f"{'='*65}\n")


if __name__ == '__main__':
    main()
