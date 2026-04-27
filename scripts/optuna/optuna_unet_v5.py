"""
Optimizacion de hiperparametros con Optuna para U-Net / Attention U-Net (dataset 5m).
======================================================================================
Espacio de busqueda:
  - architecture  : unet | attention_unet
  - base_channels : 32 | 48 | 64
  - dropout_p     : [0.0, 0.20]
  - lr            : log-uniform [3e-5, 5e-4]
  - weight_decay  : log-uniform [5e-6, 6e-4]
  - batch_size    : 8 | 16
  - optimizer     : adam | adamw
  - grad_clip     : 0.0 | 1.0

Fijado: loss=MSE, epochs=150, sin SCE, sin augmentation, dataset 5m temporal split.
"""

import os, sys, copy, argparse, json, warnings
warnings.filterwarnings('ignore')

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import optuna
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler

PROJECT_ROOT = 'E:/PycharmProjects/SnowPrediction'
sys.path.insert(0, PROJECT_ROOT)

from data.dataset import SnowDataset, SnowDatasetEval, load_splits
from models.unet import UNet, AttentionUNet
from training.train import get_device
from utils.metrics import compute_metrics

# -- Configuracion -------------------------------------------------------------
DATA_ROOT   = 'E:/PycharmProjects/SnowPrediction/Articulo 1/Data/processed/dataset_v5_5m'
CSV_FILE    = os.path.join(DATA_ROOT, 'dataset_v4_fisico.csv')
IMGS_DIR    = os.path.join(DATA_ROOT, 'images')
MASKS_DIR   = os.path.join(DATA_ROOT, 'masks')
MODELS_DIR  = 'E:/PycharmProjects/SnowPrediction/Articulo 1/Models/optuna_unet'
RESULTS_DIR = 'E:/PycharmProjects/SnowPrediction/results/optuna_unet'
DB_PATH     = f'sqlite:///{RESULTS_DIR}/optuna_unet_5m.db'
STUDY_NAME  = 'unet_5m_hpo_v3'

EPOCHS       = 150
IN_CH        = 5
OUT_CH       = 1
SEED         = 42
PRUNE_AFTER  = 20
REPORT_EVERY = 10

os.makedirs(MODELS_DIR,  exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

DEVICE = get_device('auto')

print("\nCargando splits de datos...")
TRAIN_DF, VAL_DF, TEST_DF = load_splits(CSV_FILE, source='lidar', split_type='temporal')
print(f"  Train: {len(TRAIN_DF)} tiles | Val: {len(VAL_DF)} tiles | Test: {len(TEST_DF)} tiles")


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


def objective(trial):
    arch      = trial.suggest_categorical('architecture',  ['unet', 'attention_unet'])
    base_ch   = trial.suggest_categorical('base_channels', [32, 48, 64])
    dropout_p = trial.suggest_float('dropout_p',    0.0,  0.20)
    lr        = trial.suggest_float('lr',           3e-5, 5e-4, log=True)
    wd        = trial.suggest_float('weight_decay', 5e-6, 6e-4, log=True)
    bs        = trial.suggest_categorical('batch_size', [8, 16])
    opt_name  = trial.suggest_categorical('optimizer',   ['adam', 'adamw'])
    grad_clip = trial.suggest_categorical('grad_clip',   [0.0, 1.0])

    features = [base_ch, base_ch*2, base_ch*4, base_ch*8]
    if bs == 16 and base_ch >= 64:
        bs = 8

    torch.manual_seed(SEED)
    np.random.seed(SEED)

    train_ds = SnowDataset(TRAIN_DF, IMGS_DIR, MASKS_DIR, use_sce=False, augment=False)
    val_ds   = SnowDataset(VAL_DF,   IMGS_DIR, MASKS_DIR, use_sce=False, augment=False)
    test_ds  = SnowDatasetEval(TEST_DF, IMGS_DIR, MASKS_DIR, use_sce=False)
    train_loader = DataLoader(train_ds, batch_size=bs, shuffle=True,  num_workers=0, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=bs, shuffle=False, num_workers=0, pin_memory=True)
    test_loader  = DataLoader(test_ds,  batch_size=bs, shuffle=False, num_workers=0)

    if arch == 'unet':
        model = UNet(in_channels=IN_CH, out_channels=OUT_CH, features=features, dropout_p=dropout_p).to(DEVICE)
    else:
        model = AttentionUNet(in_channels=IN_CH, out_channels=OUT_CH, features=features, dropout_p=dropout_p).to(DEVICE)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    trial.set_user_attr('n_params', n_params)

    criterion = nn.MSELoss()
    optimizer = (optim.AdamW if opt_name == 'adamw' else optim.Adam)(
        model.parameters(), lr=lr, weight_decay=wd)

    print(f"\n{'-'*65}", flush=True)
    print(f"[Trial {trial.number:03d}] {arch} | base={base_ch} | "
          f"lr={lr:.1e} | wd={wd:.1e} | bs={bs} | opt={opt_name} | "
          f"gc={grad_clip} | dp={dropout_p:.2f}", flush=True)
    print(f"  features={features} | params={n_params:,}", flush=True)

    best_val_loss, best_state = float('inf'), None

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
            best_val_loss = val_loss
            best_state    = copy.deepcopy(model.state_dict())

        if epoch >= PRUNE_AFTER and epoch % REPORT_EVERY == 0:
            trial.report(val_loss, epoch)
            if trial.should_prune():
                print(f"  -> Pruned en epoch {epoch} (val={val_loss:.4f})", flush=True)
                raise optuna.exceptions.TrialPruned()

        if epoch % 50 == 0 or epoch == EPOCHS:
            print(f"  Epoch {epoch:03d}/{EPOCHS} | val={val_loss:.4f} | best={best_val_loss:.4f}", flush=True)

    model.load_state_dict(best_state)
    metrics = compute_test_metrics(model, test_loader)
    trial.set_user_attr('test_R2',       metrics['R2'])
    trial.set_user_attr('test_MAE',      metrics['MAE'])
    trial.set_user_attr('test_RMSE',     metrics['RMSE'])
    trial.set_user_attr('test_Bias',     metrics['Bias'])
    trial.set_user_attr('best_val_loss', round(best_val_loss, 6))

    torch.save(best_state, os.path.join(MODELS_DIR, f'trial_{trial.number:03d}_r2{metrics["R2"]:.4f}.pth'))

    print(f"  -> COMPLETO | val_best={best_val_loss:.4f} | "
          f"test_R2={metrics['R2']:.4f} | test_MAE={metrics['MAE']:.4f} m", flush=True)
    return best_val_loss


def print_ranking(study, trial):
    if trial.state != optuna.trial.TrialState.COMPLETE:
        return
    completed = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    ranked = sorted(completed, key=lambda t: t.user_attrs.get('test_R2', -999), reverse=True)

    print(f"\n{'='*65}", flush=True)
    print(f"  RANKING U-Net tras {len(completed)} trials completados", flush=True)
    print(f"{'='*65}", flush=True)
    print(f"  {'#':<4} {'Trial':<6} {'val_loss':<10} {'test_R2':<10} "
          f"{'arch':<16} {'base':<6} {'lr':<8} {'wd':<8} {'bs':<4} {'opt':<6}", flush=True)
    print(f"  {'-'*4} {'-'*6} {'-'*10} {'-'*10} {'-'*16} {'-'*6} {'-'*8} {'-'*8} {'-'*4} {'-'*6}", flush=True)
    for i, t in enumerate(ranked[:10], 1):
        p = t.params
        print(f"  {i:<4} {t.number:<6} "
              f"{t.user_attrs.get('best_val_loss', t.value):<10.4f} "
              f"{t.user_attrs.get('test_R2', -9):<10.4f} "
              f"{p.get('architecture','?'):<16} "
              f"{p.get('base_channels','?'):<6} "
              f"{p.get('lr',0):<8.1e} "
              f"{p.get('weight_decay',0):<8.1e} "
              f"{p.get('batch_size','?'):<4} "
              f"{p.get('optimizer','?'):<6}", flush=True)
    print(f"{'='*65}\n", flush=True)

    ranking_data = [
        {'rank': i, 'trial': t.number,
         'val_loss': t.user_attrs.get('best_val_loss', t.value),
         'test_R2': t.user_attrs.get('test_R2'), 'test_MAE': t.user_attrs.get('test_MAE'),
         'test_RMSE': t.user_attrs.get('test_RMSE'), 'test_Bias': t.user_attrs.get('test_Bias'),
         'n_params': t.user_attrs.get('n_params'), 'params': t.params}
        for i, t in enumerate(ranked, 1)
    ]
    with open(os.path.join(RESULTS_DIR, 'ranking_unet.json'), 'w') as f:
        json.dump(ranking_data, f, indent=2)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--trials',  type=int, default=60)
    parser.add_argument('--timeout', type=int, default=None)
    args = parser.parse_args()

    sampler = TPESampler(seed=SEED, n_startup_trials=10)
    pruner  = MedianPruner(n_startup_trials=10, n_warmup_steps=20, interval_steps=10)

    study = optuna.create_study(
        study_name=STUDY_NAME, storage=DB_PATH, direction='minimize',
        sampler=sampler, pruner=pruner, load_if_exists=True,
    )

    n_existing = len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])
    print(f"\nEstudio: {STUDY_NAME}")
    print(f"Trials completados: {n_existing} | Nuevos: {args.trials}")
    print(f"Arquitecturas: unet, attention_unet")
    print(f"Objetivo: minimizar val_loss (MSE) | Metrica: test R2 (2024-2025)\n")

    study.optimize(objective, n_trials=args.trials, timeout=args.timeout,
                   callbacks=[print_ranking], catch=(RuntimeError,))

    completed = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    pruned    = [t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED]
    print(f"\n{'='*65}")
    print(f"  U-Net HPO COMPLETADO | Trials: {len(completed)} completados, {len(pruned)} pruned")
    if completed:
        best_r2 = max(completed, key=lambda t: t.user_attrs.get('test_R2', -999))
        print(f"  Mejor test_R2: {best_r2.user_attrs.get('test_R2'):.4f} (Trial #{best_r2.number})")
        print(f"  Params: {best_r2.params}")
    print(f"{'='*65}\n")


if __name__ == '__main__':
    main()
