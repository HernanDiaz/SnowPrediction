"""
Búsqueda de hiperparámetros con Optuna para U-Net snow depth (dataset 5m).
==========================================================================

Objetivo: minimizar val_loss (MSE, val=2023).
Registra también test R² (2024-2025) como atributo informativo por trial.

Espacio de búsqueda:
  - architecture  : unet | attention_unet
  - base_channels : 32 | 48 | 64 | 96  → features = [b, 2b, 4b, 8b]
  - dropout_p     : [0.0, 0.20]
  - lr            : log-uniform [3e-5, 5e-4]
  - weight_decay  : log-uniform [5e-6, 6e-4]
  - batch_size    : 8 | 16
  - optimizer     : adam | adamw
  - grad_clip     : 0.0 | 1.0

Fijado: loss=MSE, epochs=150, sin SCE, sin augmentation, dataset 5m temporal split.
Pruning: MedianPruner (warm-up 10 trials, reporte cada 10 épocas desde ep20).
Storage: SQLite (resumible si se interrumpe).

Uso:
    python optuna_search.py              # 60 trials (por defecto)
    python optuna_search.py --trials 30  # N trials personalizados
    python optuna_search.py --resume     # Reanuda estudio existente
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

# ── Ruta al proyecto ──────────────────────────────────────────────────────────
PROJECT_ROOT = str(__import__('pathlib').Path(__file__).resolve().parent.parent.parent)
sys.path.insert(0, PROJECT_ROOT)

from data.dataset import SnowDataset, SnowDatasetEval, load_splits
from models.unet import UNet, AttentionUNet
from training.train import get_device
from utils.metrics import compute_metrics

# ── Configuración fija ────────────────────────────────────────────────────────
DATA_ROOT   = str(__import__('pathlib').Path(PROJECT_ROOT) / 'Articulo 1/Data/processed/dataset_v5_5m')
CSV_FILE    = os.path.join(DATA_ROOT, 'dataset_v4_fisico.csv')
IMGS_DIR    = os.path.join(DATA_ROOT, 'images')
MASKS_DIR   = os.path.join(DATA_ROOT, 'masks')
MODELS_DIR  = str(__import__('pathlib').Path(PROJECT_ROOT) / 'Articulo 1/Models/optuna')
RESULTS_DIR = str(__import__('pathlib').Path(PROJECT_ROOT) / 'results/optuna')
DB_PATH     = f'sqlite:///{RESULTS_DIR}/optuna_5m.db'
STUDY_NAME  = 'unet_5m_hpo_v2'

EPOCHS       = 150
IN_CH        = 5
OUT_CH       = 1
SEED         = 42           # Semilla fija para reproducibilidad dentro de cada trial
PRUNE_AFTER  = 20           # No prunear antes de este epoch
REPORT_EVERY = 10           # Reportar val_loss a Optuna cada N epochs

os.makedirs(MODELS_DIR,  exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

# ── Device (una sola vez) ─────────────────────────────────────────────────────
DEVICE = get_device('auto')

# ── Carga de datos (una sola vez para todos los trials) ───────────────────────
print("\nCargando splits de datos...")
TRAIN_DF, VAL_DF, TEST_DF = load_splits(CSV_FILE, source='lidar', split_type='temporal')
print(f"  Train: {len(TRAIN_DF)} tiles | Val: {len(VAL_DF)} tiles | Test: {len(TEST_DF)} tiles")


# ─────────────────────────────────────────────────────────────────────────────
# Función auxiliar: test R² (solo pixeles con snow > 0.01 m, igual que evaluate.py)
# ─────────────────────────────────────────────────────────────────────────────
def compute_test_r2(model: nn.Module, test_loader: DataLoader) -> dict:
    """Calcula métricas sobre test set filtrando píxeles con target > 0.01 m."""
    model.eval()
    all_preds, all_targets = [], []

    with torch.no_grad():
        for batch in test_loader:
            images, masks = batch[0], batch[1]
            outputs = model(images.to(DEVICE)).cpu().numpy()   # (B,1,H,W)
            targets = masks.numpy()                             # (B,1,H,W)

            tgt_flat = targets.squeeze(1).flatten()
            out_flat = outputs.squeeze(1).flatten()
            valid    = tgt_flat > 0.01

            if valid.sum() > 0:
                all_preds.extend(out_flat[valid].tolist())
                all_targets.extend(tgt_flat[valid].tolist())

    y_pred = np.array(all_preds)
    y_true = np.array(all_targets)
    return compute_metrics(y_true, y_pred)


# ─────────────────────────────────────────────────────────────────────────────
# Función objetivo de Optuna
# ─────────────────────────────────────────────────────────────────────────────
def objective(trial: optuna.Trial) -> float:
    # ── 1. Muestrear hiperparámetros ─────────────────────────────────────────
    arch      = trial.suggest_categorical('architecture',  ['unet', 'attention_unet'])
    base_ch   = trial.suggest_categorical('base_channels', [32, 48, 64])
    dropout_p = trial.suggest_float('dropout_p',    0.0,  0.20)
    lr        = trial.suggest_float('lr',           3e-5, 5e-4, log=True)
    wd        = trial.suggest_float('weight_decay', 5e-6, 6e-4, log=True)
    bs        = trial.suggest_categorical('batch_size', [8, 16])
    opt_name  = trial.suggest_categorical('optimizer',   ['adam', 'adamw'])
    grad_clip = trial.suggest_categorical('grad_clip',   [0.0, 1.0])

    features = [base_ch, base_ch * 2, base_ch * 4, base_ch * 8]

    # Limitar combinaciones que saturan VRAM (>10M params con bs=16 es demasiado lento)
    n_params_est = base_ch * 64 * 1000   # estimacion aproximada
    if bs == 16 and base_ch >= 64:
        bs = 8   # forzar bs=8 para modelos grandes

    # ── 2. Semilla para reproducibilidad ─────────────────────────────────────
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    # ── 3. DataLoaders ───────────────────────────────────────────────────────
    train_ds = SnowDataset(TRAIN_DF, IMGS_DIR, MASKS_DIR, use_sce=False, augment=False)
    val_ds   = SnowDataset(VAL_DF,   IMGS_DIR, MASKS_DIR, use_sce=False, augment=False)
    test_ds  = SnowDatasetEval(TEST_DF, IMGS_DIR, MASKS_DIR, use_sce=False)

    train_loader = DataLoader(train_ds, batch_size=bs, shuffle=True,  num_workers=0, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=bs, shuffle=False, num_workers=0, pin_memory=True)
    test_loader  = DataLoader(test_ds,  batch_size=bs, shuffle=False, num_workers=0)

    # ── 4. Modelo ─────────────────────────────────────────────────────────────
    if arch == 'unet':
        model = UNet(in_channels=IN_CH, out_channels=OUT_CH,
                     features=features, dropout_p=dropout_p).to(DEVICE)
    else:
        model = AttentionUNet(in_channels=IN_CH, out_channels=OUT_CH,
                              features=features, dropout_p=dropout_p).to(DEVICE)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    trial.set_user_attr('n_params', n_params)

    # ── 5. Optimizador ────────────────────────────────────────────────────────
    criterion = nn.MSELoss()
    if opt_name == 'adamw':
        optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    else:
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=wd)

    print(f"\n{'-'*65}", flush=True)
    print(f"[Trial {trial.number:03d}] {arch} | base={base_ch} | "
          f"lr={lr:.1e} | wd={wd:.1e} | bs={bs} | opt={opt_name} | "
          f"gc={grad_clip} | dp={dropout_p:.2f}", flush=True)
    print(f"  features={features} | params={n_params:,}", flush=True)

    # ── 6. Bucle de entrenamiento ─────────────────────────────────────────────
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

        # Validación
        model.eval()
        val_total = 0.0
        with torch.no_grad():
            for images, masks in val_loader:
                images, masks = images.to(DEVICE), masks.to(DEVICE)
                val_total += criterion(model(images), masks).item()
        val_loss = val_total / len(val_loader)

        # Guardar mejor checkpoint
        if val_loss < best_val_loss:
            best_val_loss    = val_loss
            best_model_state = copy.deepcopy(model.state_dict())

        # Pruning Optuna (solo a partir de PRUNE_AFTER, cada REPORT_EVERY)
        if epoch >= PRUNE_AFTER and epoch % REPORT_EVERY == 0:
            trial.report(val_loss, epoch)
            if trial.should_prune():
                print(f"  -> Pruned en epoch {epoch} (val={val_loss:.4f})", flush=True)
                raise optuna.exceptions.TrialPruned()

        if epoch % 50 == 0 or epoch == EPOCHS:
            print(f"  Epoch {epoch:03d}/{EPOCHS} | val={val_loss:.4f} | best={best_val_loss:.4f}", flush=True)

    # -- 7. Evaluacion en test set (informativa, no usada para seleccion) ----
    model.load_state_dict(best_model_state)
    test_metrics = compute_test_r2(model, test_loader)
    trial.set_user_attr('test_R2',   test_metrics['R2'])
    trial.set_user_attr('test_MAE',  test_metrics['MAE'])
    trial.set_user_attr('test_RMSE', test_metrics['RMSE'])
    trial.set_user_attr('test_Bias', test_metrics['Bias'])
    trial.set_user_attr('best_val_loss', round(best_val_loss, 6))

    # Guardar modelo de cada trial
    model_path = os.path.join(MODELS_DIR, f'trial_{trial.number:03d}_r2{test_metrics["R2"]:.4f}.pth')
    torch.save(best_model_state, model_path)

    print(f"  -> COMPLETO | val_best={best_val_loss:.4f} | "
          f"test_R2={test_metrics['R2']:.4f} | "
          f"test_MAE={test_metrics['MAE']:.4f} m", flush=True)

    return best_val_loss


# ─────────────────────────────────────────────────────────────────────────────
# Callback: imprime ranking tras cada trial
# ─────────────────────────────────────────────────────────────────────────────
def print_ranking(study: optuna.Study, trial: optuna.Trial):
    if trial.state != optuna.trial.TrialState.COMPLETE:
        return

    completed = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    # Ordenar por test_R² descendente (usando val_loss si no hay test_R²)
    ranked = sorted(completed,
                    key=lambda t: t.user_attrs.get('test_R2', -999),
                    reverse=True)

    print(f"\n{'='*65}", flush=True)
    print(f"  RANKING tras {len(completed)} trials completados", flush=True)
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
              f"{p.get('lr', 0):<8.1e} "
              f"{p.get('weight_decay', 0):<8.1e} "
              f"{p.get('batch_size','?'):<4} "
              f"{p.get('optimizer','?'):<6}", flush=True)
    print(f"{'='*65}\n", flush=True)

    # Guardar ranking en JSON para análisis posterior
    ranking_path = os.path.join(RESULTS_DIR, 'ranking.json')
    ranking_data = [
        {
            'rank': i,
            'trial': t.number,
            'val_loss': t.user_attrs.get('best_val_loss', t.value),
            'test_R2': t.user_attrs.get('test_R2'),
            'test_MAE': t.user_attrs.get('test_MAE'),
            'test_RMSE': t.user_attrs.get('test_RMSE'),
            'test_Bias': t.user_attrs.get('test_Bias'),
            'n_params': t.user_attrs.get('n_params'),
            'params': t.params,
        }
        for i, t in enumerate(ranked, 1)
    ]
    with open(ranking_path, 'w') as f:
        json.dump(ranking_data, f, indent=2)


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description='Optuna HPO para U-Net snow depth 5m')
    parser.add_argument('--trials',  type=int, default=60,         help='Número de trials (default: 60)')
    parser.add_argument('--resume',  action='store_true',          help='Reanudar estudio existente')
    parser.add_argument('--timeout', type=int, default=None,       help='Timeout en segundos (opcional)')
    args = parser.parse_args()

    # Sampler TPE con semilla fija (reproducible)
    sampler = TPESampler(seed=SEED, n_startup_trials=10)

    # Pruner: MedianPruner
    #   n_startup_trials=10: no prunear hasta tener 10 trials completos de referencia
    #   n_warmup_steps=20:   no prunear antes del epoch 20
    #   interval_steps=10:   evaluar cada 10 epochs (igual que REPORT_EVERY)
    pruner = MedianPruner(n_startup_trials=10, n_warmup_steps=20, interval_steps=10)

    if args.resume:
        print(f"\nReanudando estudio: {STUDY_NAME}")
        study = optuna.load_study(study_name=STUDY_NAME, storage=DB_PATH,
                                  sampler=sampler, pruner=pruner)
    else:
        print(f"\nCreando estudio nuevo: {STUDY_NAME}")
        study = optuna.create_study(
            study_name=STUDY_NAME,
            storage=DB_PATH,
            direction='minimize',
            sampler=sampler,
            pruner=pruner,
            load_if_exists=True,   # No falla si ya existe
        )

    n_existing = len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])
    print(f"Trials ya completados: {n_existing}")
    print(f"Trials nuevos a lanzar: {args.trials}")
    print(f"Objetivo: minimizar val_loss (MSE, val=2023)")
    print(f"Métrica informativa: test R² (2024-2025)\n")

    study.optimize(
        objective,
        n_trials=args.trials,
        timeout=args.timeout,
        callbacks=[print_ranking],
        catch=(RuntimeError,),   # Capturar errores de GPU sin abortar el estudio
    )

    # ── Resumen final ─────────────────────────────────────────────────────────
    completed = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    pruned    = [t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED]

    print(f"\n{'='*65}")
    print(f"  BÚSQUEDA COMPLETADA")
    print(f"  Trials completados: {len(completed)} | Pruned: {len(pruned)}")
    print(f"{'='*65}")

    if completed:
        best_by_val = study.best_trial
        best_by_r2  = max(completed, key=lambda t: t.user_attrs.get('test_R2', -999))

        print(f"\n  Mejor por val_loss (criterio de selección):")
        print(f"    Trial #{best_by_val.number} | val_loss={best_by_val.value:.4f} | "
              f"test_R²={best_by_val.user_attrs.get('test_R2', 'N/A')}")
        print(f"    Params: {best_by_val.params}")

        print(f"\n  Mejor por test_R² (métrica objetivo del paper):")
        print(f"    Trial #{best_by_r2.number} | val_loss={best_by_r2.value:.4f} | "
              f"test_R²={best_by_r2.user_attrs.get('test_R2')}")
        print(f"    Params: {best_by_r2.params}")

        print(f"\n  Ranking final guardado en: {RESULTS_DIR}/ranking.json")
        print(f"  Modelos guardados en:       {MODELS_DIR}")
        print(f"  Base de datos Optuna:       {DB_PATH}")

    print(f"{'='*65}\n")


if __name__ == '__main__':
    main()
