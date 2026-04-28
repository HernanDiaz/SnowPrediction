"""
Busqueda de hiperparametros con Optuna - Dataset v6 (17 canales).
=================================================================

Dataset v6: 6 canales topo+SCE + 8 Sx_100m (Wind Shelter Index) + 3 persistencia nival.
  channel_indices = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,30,31,32]

Espacio de busqueda:
  - architecture  : unet | attention_unet | resunetpp
  - base_channels : 32 | 48 | 64
  - dropout_p     : [0.0, 0.20]
  - lr            : log-uniform [3e-5, 5e-4]
  - weight_decay  : log-uniform [5e-6, 6e-4]
  - batch_size    : 8 | 16
  - optimizer     : adam | adamw
  - grad_clip     : 0.0 | 1.0

Fijado: loss=MSE, epochs=150, in_channels=17, num_groups=8 (resunetpp),
        sin augmentation, dataset v6 5m temporal split.
Pruning: MedianPruner (warm-up 10 trials, reporte cada 10 epocas desde ep20).
Storage: SQLite (resumible si se interrumpe).

Salidas guardadas:
  - Pesos de cada trial: Articulo 1/Models/optuna_v6/trial_NNN_r2X.XXXX.pth
  - Ranking actualizado tras cada trial: results/optuna_v6/ranking_v6.json
  - Base de datos Optuna: results/optuna_v6/optuna_v6.db

Uso:
    python optuna_v6.py               # 60 trials (por defecto)
    python optuna_v6.py --trials 40   # N trials personalizados
    python optuna_v6.py --resume      # Reanuda estudio existente
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

# -- Ruta al proyecto ----------------------------------------------------------
PROJECT_ROOT = str(__import__('pathlib').Path(__file__).resolve().parent)
sys.path.insert(0, PROJECT_ROOT)

from data.dataset import SnowDataset, SnowDatasetEval, load_splits
from models.unet import UNet, AttentionUNet
from models.resunet import ResUNetPP
from training.train import get_device
from utils.metrics import compute_metrics

# -- Configuracion fija --------------------------------------------------------
DATA_ROOT   = str(__import__('pathlib').Path(PROJECT_ROOT) / 'Articulo 1/Data/processed/dataset_v6_5m')
CSV_FILE    = os.path.join(DATA_ROOT, 'dataset_v6_fisico.csv')
IMGS_DIR    = os.path.join(DATA_ROOT, 'images')
MASKS_DIR   = os.path.join(DATA_ROOT, 'masks')
MODELS_DIR  = str(__import__('pathlib').Path(PROJECT_ROOT) / 'Articulo 1/Models/optuna_v6')
RESULTS_DIR = str(__import__('pathlib').Path(PROJECT_ROOT) / 'results/optuna_v6')
DB_PATH     = f'sqlite:///{RESULTS_DIR}/optuna_v6.db'
STUDY_NAME  = 'snow_v6_hpo_v1'

EPOCHS       = 150
IN_CH        = 17          # 6 topo+SCE + 8 Sx_100m + 3 persistencia
OUT_CH       = 1
NUM_GROUPS   = 8           # GroupNorm para ResUNet++ (fijo)
SEED         = 42
PRUNE_AFTER  = 20
REPORT_EVERY = 10

# Canales utiles del dataset v6 (omitir 16 canales vacios en posiciones 14-29)
CHANNEL_INDICES = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 30, 31, 32]

os.makedirs(MODELS_DIR,  exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

# -- Device (una sola vez) -----------------------------------------------------
DEVICE = get_device('auto')

# -- Carga de datos (una sola vez para todos los trials) -----------------------
print("\nCargando splits de datos v6...")
TRAIN_DF, VAL_DF, TEST_DF = load_splits(CSV_FILE, source='lidar', split_type='temporal')
print(f"  Train: {len(TRAIN_DF)} tiles | Val: {len(VAL_DF)} tiles | Test: {len(TEST_DF)} tiles")
print(f"  In channels: {IN_CH} | Channel indices: {CHANNEL_INDICES}\n")


# -----------------------------------------------------------------------------
# Helper: metricas en test set
# -----------------------------------------------------------------------------
def compute_test_metrics(model: nn.Module, test_loader: DataLoader) -> dict:
    """Calcula R2/MAE/RMSE/Bias sobre test set (pixeles con target > 0.01 m)."""
    model.eval()
    all_preds, all_targets = [], []

    with torch.no_grad():
        for batch in test_loader:
            images, masks = batch[0], batch[1]
            outputs = model(images.to(DEVICE)).cpu().numpy()
            targets = masks.numpy()

            tgt_flat = targets.squeeze(1).flatten()
            out_flat = outputs.squeeze(1).flatten()
            valid    = tgt_flat > 0.01

            if valid.sum() > 0:
                all_preds.extend(out_flat[valid].tolist())
                all_targets.extend(tgt_flat[valid].tolist())

    y_pred = np.array(all_preds)
    y_true = np.array(all_targets)
    return compute_metrics(y_true, y_pred)


# -----------------------------------------------------------------------------
# Helper: construir datasets con channel_indices v6
# -----------------------------------------------------------------------------
def _make_datasets(bs: int):
    """Crea train/val/test datasets con los 17 canales v6."""
    train_ds = SnowDataset(TRAIN_DF, IMGS_DIR, MASKS_DIR,
                           use_sce=False, augment=False, n_channels=IN_CH)
    train_ds.channel_indices = CHANNEL_INDICES

    val_ds = SnowDataset(VAL_DF, IMGS_DIR, MASKS_DIR,
                         use_sce=False, augment=False, n_channels=IN_CH)
    val_ds.channel_indices = CHANNEL_INDICES

    test_ds = SnowDatasetEval(TEST_DF, IMGS_DIR, MASKS_DIR,
                              use_sce=False, n_channels=IN_CH)
    test_ds.channel_indices = CHANNEL_INDICES

    train_loader = DataLoader(train_ds, batch_size=bs, shuffle=True,
                              num_workers=0, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=bs, shuffle=False,
                              num_workers=0, pin_memory=True)
    test_loader  = DataLoader(test_ds,  batch_size=bs, shuffle=False,
                              num_workers=0)

    return train_loader, val_loader, test_loader


# -----------------------------------------------------------------------------
# Funcion objetivo de Optuna
# -----------------------------------------------------------------------------
def objective(trial: optuna.Trial) -> float:
    # -- 1. Muestrear hiperparametros -----------------------------------------
    arch      = trial.suggest_categorical('architecture',
                                          ['unet', 'attention_unet', 'resunetpp'])
    base_ch   = trial.suggest_categorical('base_channels', [32, 48, 64])
    dropout_p = trial.suggest_float('dropout_p',    0.0,  0.20)
    lr        = trial.suggest_float('lr',           3e-5, 5e-4, log=True)
    wd        = trial.suggest_float('weight_decay', 5e-6, 6e-4, log=True)
    bs        = trial.suggest_categorical('batch_size', [8, 16])
    opt_name  = trial.suggest_categorical('optimizer',   ['adam', 'adamw'])
    grad_clip = trial.suggest_categorical('grad_clip',   [0.0, 1.0])

    features = [base_ch, base_ch * 2, base_ch * 4, base_ch * 8]

    # Limitar VRAM: modelos grandes con bs=16 pueden saturar memoria
    if bs == 16 and base_ch >= 64:
        bs = 8

    # ResUNet++ tiene mas parametros: forzar bs<=8 con base>=48
    if arch == 'resunetpp' and bs == 16 and base_ch >= 48:
        bs = 8

    # -- 2. Semilla para reproducibilidad ------------------------------------
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    # -- 3. DataLoaders -------------------------------------------------------
    train_loader, val_loader, test_loader = _make_datasets(bs)

    # -- 4. Modelo ------------------------------------------------------------
    if arch == 'unet':
        model = UNet(in_channels=IN_CH, out_channels=OUT_CH,
                     features=features, dropout_p=dropout_p).to(DEVICE)
    elif arch == 'attention_unet':
        model = AttentionUNet(in_channels=IN_CH, out_channels=OUT_CH,
                              features=features, dropout_p=dropout_p).to(DEVICE)
    else:  # resunetpp
        model = ResUNetPP(in_channels=IN_CH, out_channels=OUT_CH,
                          features=features, dropout_p=dropout_p,
                          num_groups=NUM_GROUPS).to(DEVICE)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    trial.set_user_attr('n_params', n_params)

    # -- 5. Optimizador -------------------------------------------------------
    criterion = nn.MSELoss()
    if opt_name == 'adamw':
        optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    else:
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=wd)

    print(f"\n{'-'*70}", flush=True)
    print(f"[Trial {trial.number:03d}] {arch} | base={base_ch} | "
          f"lr={lr:.1e} | wd={wd:.1e} | bs={bs} | "
          f"opt={opt_name} | gc={grad_clip} | dp={dropout_p:.2f}", flush=True)
    print(f"  features={features} | params={n_params:,} | in_ch={IN_CH}", flush=True)

    # -- 6. Bucle de entrenamiento --------------------------------------------
    best_val_loss    = float('inf')
    best_model_state = None

    for epoch in range(1, EPOCHS + 1):
        # Entrenamiento
        model.train()
        for images, masks in train_loader:
            images, masks = images.to(DEVICE), masks.to(DEVICE)
            optimizer.zero_grad()
            loss = criterion(model(images), masks)
            loss.backward()
            if grad_clip > 0:
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()

        # Validacion
        model.eval()
        val_total = 0.0
        with torch.no_grad():
            for images, masks in val_loader:
                images, masks = images.to(DEVICE), masks.to(DEVICE)
                val_total += criterion(model(images), masks).item()
        val_loss = val_total / len(val_loader)

        # Guardar mejor checkpoint en memoria
        if val_loss < best_val_loss:
            best_val_loss    = val_loss
            best_model_state = copy.deepcopy(model.state_dict())

        # Pruning Optuna (solo desde PRUNE_AFTER, cada REPORT_EVERY epocas)
        if epoch >= PRUNE_AFTER and epoch % REPORT_EVERY == 0:
            trial.report(val_loss, epoch)
            if trial.should_prune():
                print(f"  -> Pruned en epoch {epoch} (val={val_loss:.4f})", flush=True)
                raise optuna.exceptions.TrialPruned()

        if epoch % 50 == 0 or epoch == EPOCHS:
            print(f"  Epoch {epoch:03d}/{EPOCHS} | "
                  f"val={val_loss:.4f} | best={best_val_loss:.4f}", flush=True)

    # -- 7. Evaluacion en test set --------------------------------------------
    model.load_state_dict(best_model_state)
    metrics = compute_test_metrics(model, test_loader)

    trial.set_user_attr('test_R2',        metrics['R2'])
    trial.set_user_attr('test_MAE',       metrics['MAE'])
    trial.set_user_attr('test_RMSE',      metrics['RMSE'])
    trial.set_user_attr('test_Bias',      metrics['Bias'])
    trial.set_user_attr('best_val_loss',  round(best_val_loss, 6))

    # -- 8. Guardar pesos del trial -------------------------------------------
    model_filename = (
        f"trial_{trial.number:03d}_{arch}_"
        f"b{base_ch}_r2{metrics['R2']:.4f}.pth"
    )
    model_path = os.path.join(MODELS_DIR, model_filename)
    torch.save({
        'trial':          trial.number,
        'architecture':   arch,
        'features':       features,
        'in_channels':    IN_CH,
        'out_channels':   OUT_CH,
        'num_groups':     NUM_GROUPS,
        'dropout_p':      dropout_p,
        'channel_indices': CHANNEL_INDICES,
        'state_dict':     best_model_state,
        'val_loss':       round(best_val_loss, 6),
        'test_R2':        metrics['R2'],
        'test_MAE':       metrics['MAE'],
        'params':         trial.params,
    }, model_path)

    print(f"  -> COMPLETO | val_best={best_val_loss:.4f} | "
          f"test_R2={metrics['R2']:.4f} | "
          f"test_MAE={metrics['MAE']:.4f} m | "
          f"model={model_filename}", flush=True)

    return best_val_loss


# -----------------------------------------------------------------------------
# Callback: ranking actualizado tras cada trial
# -----------------------------------------------------------------------------
def print_ranking(study: optuna.Study, trial: optuna.Trial):
    if trial.state != optuna.trial.TrialState.COMPLETE:
        return

    completed = [t for t in study.trials
                 if t.state == optuna.trial.TrialState.COMPLETE]
    ranked = sorted(completed,
                    key=lambda t: t.user_attrs.get('test_R2', -999),
                    reverse=True)

    print(f"\n{'='*75}", flush=True)
    print(f"  RANKING v6 tras {len(completed)} trials completados", flush=True)
    print(f"{'='*75}", flush=True)
    print(f"  {'#':<4} {'Trial':<6} {'val_loss':<10} {'R2':<8} {'MAE':<8} "
          f"{'arch':<16} {'base':<6} {'lr':<8} {'wd':<8} {'bs':<4} {'opt':<6}", flush=True)
    print(f"  {'-'*4} {'-'*6} {'-'*10} {'-'*8} {'-'*8} "
          f"{'-'*16} {'-'*6} {'-'*8} {'-'*8} {'-'*4} {'-'*6}", flush=True)

    for i, t in enumerate(ranked[:10], 1):
        p = t.params
        print(f"  {i:<4} {t.number:<6} "
              f"{t.user_attrs.get('best_val_loss', t.value):<10.4f} "
              f"{t.user_attrs.get('test_R2', -9):<8.4f} "
              f"{t.user_attrs.get('test_MAE', -9):<8.4f} "
              f"{p.get('architecture', '?'):<16} "
              f"{p.get('base_channels', '?'):<6} "
              f"{p.get('lr', 0):<8.1e} "
              f"{p.get('weight_decay', 0):<8.1e} "
              f"{p.get('batch_size', '?'):<4} "
              f"{p.get('optimizer', '?'):<6}", flush=True)
    print(f"{'='*75}\n", flush=True)

    # Guardar ranking completo en JSON
    ranking_path = os.path.join(RESULTS_DIR, 'ranking_v6.json')
    ranking_data = [
        {
            'rank':        i,
            'trial':       t.number,
            'val_loss':    t.user_attrs.get('best_val_loss', t.value),
            'test_R2':     t.user_attrs.get('test_R2'),
            'test_MAE':    t.user_attrs.get('test_MAE'),
            'test_RMSE':   t.user_attrs.get('test_RMSE'),
            'test_Bias':   t.user_attrs.get('test_Bias'),
            'n_params':    t.user_attrs.get('n_params'),
            'params':      t.params,
        }
        for i, t in enumerate(ranked, 1)
    ]
    with open(ranking_path, 'w', encoding='utf-8') as f:
        json.dump(ranking_data, f, indent=2)


# -----------------------------------------------------------------------------
# Entry point
# -----------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description='Optuna HPO para snow depth v6 (17 canales, 3 arquitecturas)'
    )
    parser.add_argument('--trials',  type=int,  default=60,
                        help='Numero de trials (default: 60)')
    parser.add_argument('--resume',  action='store_true',
                        help='Reanudar estudio existente')
    parser.add_argument('--timeout', type=int,  default=None,
                        help='Timeout en segundos (opcional)')
    args = parser.parse_args()

    sampler = TPESampler(seed=SEED, n_startup_trials=10)
    pruner  = MedianPruner(n_startup_trials=10, n_warmup_steps=20,
                           interval_steps=10)

    if args.resume:
        print(f"\nReanudando estudio: {STUDY_NAME}")
        study = optuna.load_study(
            study_name=STUDY_NAME, storage=DB_PATH,
            sampler=sampler, pruner=pruner,
        )
    else:
        print(f"\nCreando estudio: {STUDY_NAME}")
        study = optuna.create_study(
            study_name=STUDY_NAME,
            storage=DB_PATH,
            direction='minimize',
            sampler=sampler,
            pruner=pruner,
            load_if_exists=True,
        )

    n_existing = len([t for t in study.trials
                      if t.state == optuna.trial.TrialState.COMPLETE])
    print(f"Trials ya completados : {n_existing}")
    print(f"Trials nuevos         : {args.trials}")
    print(f"Dataset               : v6 (17 canales)")
    print(f"Arquitecturas         : unet | attention_unet | resunetpp")
    print(f"Objetivo              : minimizar val_loss (MSE, val=2023)")
    print(f"Metrica informativa   : test R2 (2024-2025)")
    print(f"Modelos guardados en  : {MODELS_DIR}")
    print(f"Resultados en         : {RESULTS_DIR}")
    print(f"Base de datos         : {DB_PATH}\n")

    study.optimize(
        objective,
        n_trials=args.trials,
        timeout=args.timeout,
        callbacks=[print_ranking],
        catch=(RuntimeError,),   # Capturar OOM u otros errores GPU sin abortar
    )

    # -- Resumen final --------------------------------------------------------
    completed = [t for t in study.trials
                 if t.state == optuna.trial.TrialState.COMPLETE]
    pruned    = [t for t in study.trials
                 if t.state == optuna.trial.TrialState.PRUNED]

    print(f"\n{'='*75}")
    print(f"  BUSQUEDA v6 COMPLETADA")
    print(f"  Trials completados: {len(completed)} | Pruned: {len(pruned)}")
    print(f"{'='*75}")

    if completed:
        best_by_val = study.best_trial
        best_by_r2  = max(completed,
                          key=lambda t: t.user_attrs.get('test_R2', -999))

        print(f"\n  Mejor por val_loss (criterio de seleccion Optuna):")
        print(f"    Trial #{best_by_val.number} | "
              f"val_loss={best_by_val.value:.4f} | "
              f"test_R2={best_by_val.user_attrs.get('test_R2', 'N/A')}")
        print(f"    Arch: {best_by_val.params.get('architecture')} | "
              f"Params: {best_by_val.params}")

        print(f"\n  Mejor por test_R2 (metrica objetivo del paper):")
        print(f"    Trial #{best_by_r2.number} | "
              f"val_loss={best_by_r2.value:.4f} | "
              f"test_R2={best_by_r2.user_attrs.get('test_R2'):.4f} | "
              f"test_MAE={best_by_r2.user_attrs.get('test_MAE'):.4f} m")
        print(f"    Arch: {best_by_r2.params.get('architecture')} | "
              f"Params: {best_by_r2.params}")

        print(f"\n  Ranking final: {RESULTS_DIR}/ranking_v6.json")
        print(f"  Modelos:       {MODELS_DIR}/")
        print(f"  Base de datos: {DB_PATH}")

    print(f"{'='*75}\n")


if __name__ == '__main__':
    main()
