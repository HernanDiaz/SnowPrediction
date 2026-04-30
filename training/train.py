import os
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR, CosineAnnealingWarmRestarts
from torch.optim.swa_utils import AveragedModel, SWALR, update_bn
from torch.utils.data import DataLoader
from tqdm import tqdm

from utils.visualization import plot_training_curves


# ---------------------------------------------------------------------------
# Loss espacial: MSE + lambda * (1 - Pearson)
# ---------------------------------------------------------------------------

class SpatialMSELoss(nn.Module):
    """
    Loss combinada: MSE pixel a pixel + penalizacion por baja correlacion espacial.

    L = MSE(pred, target) + lambda_pearson * (1 - rho(pred, target))

    donde rho es la correlacion de Pearson computada por tile sobre todos
    los pixeles (H*W), promediada sobre el batch.

    Propiedades:
        - MSE garantiza que los valores sean correctos en magnitud.
        - El termino Pearson penaliza predicciones que estan en los sitios
          equivocados aunque tengan el valor medio correcto.
        - Es diferenciable, por lo que es compatible con backpropagation.

    Args:
        lambda_pearson: peso del termino de correlacion (default: 0.5).
            lambda=0.0 -> MSE puro
            lambda=0.5 -> igual peso a ambos terminos
            lambda=1.0 -> mas peso al patron espacial
    """

    def __init__(self, lambda_pearson: float = 0.5):
        super().__init__()
        self.lambda_pearson = lambda_pearson

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # pred, target: (B, 1, H, W)
        mse = F.mse_loss(pred, target)

        if self.lambda_pearson == 0.0:
            return mse

        B = pred.shape[0]
        p = pred.view(B, -1)    # (B, H*W)
        t = target.view(B, -1)  # (B, H*W)

        p_mean = p.mean(dim=1, keepdim=True)
        t_mean = t.mean(dim=1, keepdim=True)
        p_c = p - p_mean
        t_c = t - t_mean

        num   = (p_c * t_c).sum(dim=1)
        denom = torch.sqrt((p_c ** 2).sum(dim=1) * (t_c ** 2).sum(dim=1) + 1e-8)
        rho   = num / denom       # (B,)  en [-1, 1]

        pearson_loss = (1.0 - rho).mean()
        return mse + self.lambda_pearson * pearson_loss


def get_loss_fn(loss_name: str, huber_delta: float = 0.5,
                lambda_pearson: float = 0.5) -> nn.Module:
    if loss_name == 'spatial_mse':
        return SpatialMSELoss(lambda_pearson=lambda_pearson)
    losses = {
        'mae':   nn.L1Loss(),
        'mse':   nn.MSELoss(),
        'huber': nn.HuberLoss(delta=huber_delta),
    }
    if loss_name not in losses:
        raise ValueError(
            f"Loss desconocida: '{loss_name}'. "
            f"Usa 'mae', 'mse', 'huber' o 'spatial_mse'."
        )
    return losses[loss_name]


def get_device(device_str: str) -> torch.device:
    if device_str == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(device_str)
    print(f"Dispositivo: {device}", end="")
    if device.type == 'cuda':
        print(f"  ({torch.cuda.get_device_name(0)})", end="")
    print()
    return device


# ----------------------------------------------------------------------
# Bucles de una epoca
# ----------------------------------------------------------------------

def _train_epoch(model, loader, optimizer, criterion, device, grad_clip=0.0) -> float:
    model.train()
    total = 0.0
    pbar  = tqdm(loader, desc="  Train", leave=False)
    for images, masks in pbar:
        images, masks = images.to(device), masks.to(device)
        optimizer.zero_grad()
        loss = criterion(model(images), masks)
        loss.backward()
        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        total += loss.item()
        pbar.set_postfix({'loss': f"{loss.item():.4f}"})
    return total / len(loader)


def _val_epoch(model, loader, criterion, device) -> float:
    model.eval()
    total = 0.0
    with torch.no_grad():
        for images, masks in tqdm(loader, desc="  Val  ", leave=False):
            images, masks = images.to(device), masks.to(device)
            total += criterion(model(images), masks).item()
    return total / len(loader)


# ----------------------------------------------------------------------
# Bucle completo de entrenamiento
# ----------------------------------------------------------------------

def train_model(model:        nn.Module,
                train_loader: DataLoader,
                val_loader:   DataLoader,
                config:       dict) -> dict:
    """
    Entrena el modelo y guarda el mejor checkpoint por val loss.

    Args:
        model:        Modelo PyTorch
        train_loader: DataLoader de entrenamiento
        val_loader:   DataLoader de validacion
        config:       Configuracion completa del YAML

    Returns:
        Historial de perdidas {'train_loss': [...], 'val_loss': [...]}
    """
    cfg_tr  = config['training']
    cfg_out = config['output']
    exp     = config['experiment']['name']

    device       = get_device(cfg_tr['device'])
    model        = model.to(device)
    weight_decay = cfg_tr.get('weight_decay', 0.0)
    grad_clip    = cfg_tr.get('grad_clip', 0.0)

    # Optimizer: 'adam' (default) or 'adamw'
    opt_name = cfg_tr.get('optimizer', 'adam').lower()
    if opt_name == 'adamw':
        optimizer = optim.AdamW(model.parameters(),
                                lr=cfg_tr['learning_rate'],
                                weight_decay=weight_decay)
    else:
        optimizer = optim.Adam(model.parameters(),
                               lr=cfg_tr['learning_rate'],
                               weight_decay=weight_decay)

    huber_delta     = cfg_tr.get('huber_delta', 0.5)
    lambda_pearson  = cfg_tr.get('lambda_pearson', 0.5)
    criterion       = get_loss_fn(cfg_tr['loss'],
                                  huber_delta=huber_delta,
                                  lambda_pearson=lambda_pearson)
    epochs       = cfg_tr['epochs']

    # LR scheduler — 'plateau' (ReduceLROnPlateau), 'cosine' (CosineAnnealingLR),
    #                 'cosine_wr' (CosineAnnealingWarmRestarts), or False
    sched_name = cfg_tr.get('lr_scheduler', False)
    # Backward compat: lr_scheduler: true → plateau
    if sched_name is True:
        sched_name = 'plateau'
    scheduler = None
    if sched_name == 'plateau':
        scheduler = ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=cfg_tr.get('lr_factor', 0.5),
            patience=cfg_tr.get('lr_patience', 15),
            min_lr=cfg_tr.get('lr_min', 1e-6),
        )
    elif sched_name == 'cosine':
        scheduler = CosineAnnealingLR(
            optimizer,
            T_max=cfg_tr.get('cosine_T_max', epochs),
            eta_min=cfg_tr.get('lr_min', 1e-6),
        )
    elif sched_name == 'cosine_wr':
        scheduler = CosineAnnealingWarmRestarts(
            optimizer,
            T_0=cfg_tr.get('cosine_T0', max(1, epochs // 3)),
            T_mult=cfg_tr.get('cosine_T_mult', 1),
            eta_min=cfg_tr.get('lr_min', 1e-6),
        )

    os.makedirs(cfg_out['models_dir'],  exist_ok=True)
    os.makedirs(cfg_out['results_dir'], exist_ok=True)

    model_path = os.path.join(cfg_out['models_dir'],
                              f"{cfg_out['model_name']}.pth")
    curve_path = os.path.join(cfg_out['results_dir'],
                              f"{exp}_training_curve.png")

    history       = {'train_loss': [], 'val_loss': []}
    best_val_loss = float('inf')
    es_patience   = cfg_tr.get('es_patience', 20)   # epocas sin mejora antes de parar
    es_counter    = 0

    # Stochastic Weight Averaging (SWA)
    swa_enabled = cfg_tr.get('swa', False)
    swa_start   = cfg_tr.get('swa_start', max(1, epochs // 2))
    swa_lr      = cfg_tr.get('swa_lr', cfg_tr['learning_rate'] * 0.1)
    swa_model   = None
    swa_sched   = None
    if swa_enabled:
        swa_model = AveragedModel(model)
        swa_sched = SWALR(optimizer, swa_lr=swa_lr, anneal_epochs=10)

    sched_label = {'plateau': 'ReduceLROnPlateau', 'cosine': 'CosineAnnealingLR',
                   'cosine_wr': 'CosineAnnealingWarmRestarts'}.get(sched_name, '')
    print(f"\nExperimento : {exp}")
    print(f"Loss        : {cfg_tr['loss'].upper()}"
          + (f" (delta={huber_delta})" if cfg_tr['loss'] == 'huber' else "")
          + (f" (lambda_pearson={lambda_pearson})" if cfg_tr['loss'] == 'spatial_mse' else ""))
    print(f"Optimizer   : {opt_name.upper()}"
          + (f"  |  GradClip: {grad_clip}" if grad_clip > 0 else "")
          + (f"  |  SWA desde ep{swa_start} (lr={swa_lr:.1e})" if swa_enabled else ""))
    print(f"Epochs      : {epochs}  |  "
          f"Batch: {cfg_tr['batch_size']}  |  "
          f"LR: {cfg_tr['learning_rate']}"
          + (f"  |  WD: {weight_decay}" if weight_decay > 0 else "")
          + (f"  |  LRS: {sched_label}" if sched_label else ""))
    print(f"Train tiles : {len(train_loader.dataset)}  |  "
          f"Val tiles: {len(val_loader.dataset)}\n")

    for epoch in range(1, epochs + 1):
        tr_loss  = _train_epoch(model, train_loader, optimizer, criterion, device, grad_clip)
        val_loss = _val_epoch(model, val_loader, criterion, device)

        history['train_loss'].append(tr_loss)
        history['val_loss'].append(val_loss)

        if swa_enabled and epoch >= swa_start:
            swa_model.update_parameters(model)
            swa_sched.step()
        elif scheduler is not None:
            if isinstance(scheduler, ReduceLROnPlateau):
                scheduler.step(val_loss)
            else:
                scheduler.step()

        early_stopping = cfg_tr.get('early_stopping', True)
        saved = ""
        if not early_stopping:
            # Sin early stopping: siempre guardar el ultimo checkpoint
            torch.save(model.state_dict(), model_path)
            saved = "  [guardado]"
        elif val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), model_path)
            saved = "  [GUARDADO]"
            es_counter = 0
        else:
            es_counter += 1

        current_lr = optimizer.param_groups[0]['lr']
        lr_str = f" | LR: {current_lr:.2e}" if sched_name else ""
        es_str = f" | ES: {es_counter}/{es_patience}" if early_stopping else ""
        print(f"Epoch {epoch:03d}/{epochs} | "
              f"Train: {tr_loss:.4f} | "
              f"Val: {val_loss:.4f}{lr_str}{es_str}{saved}")

        if early_stopping and es_counter >= es_patience:
            print(f"\nEarly stopping activado en epoca {epoch} "
                  f"(sin mejora en {es_patience} epocas consecutivas).")
            break

    # SWA: actualizar BatchNorm y guardar el modelo promediado
    if swa_enabled and swa_model is not None:
        print(f"\nActualizando BatchNorm para SWA...")
        update_bn(train_loader, swa_model, device=device)
        swa_path = model_path.replace('.pth', '_swa.pth')
        torch.save(swa_model.module.state_dict(), swa_path)
        print(f"Modelo SWA guardado: {swa_path}")
        # Sobreescribir el modelo principal con el SWA (para que evaluate lo use)
        torch.save(swa_model.module.state_dict(), model_path)
        print(f"Modelo principal reemplazado con SWA: {model_path}")

    print(f"\nMejor val loss : {best_val_loss:.4f}")
    print(f"Modelo guardado: {model_path}")

    plot_training_curves(history, save_path=curve_path,
                         title=f"Curva de Aprendizaje - {exp}")

    # Guardar historial numerico de perdidas y config usada
    history_path = os.path.join(cfg_out['results_dir'], f"{exp}_history.json")
    history_out  = {
        'experiment':   exp,
        'best_val_loss': round(best_val_loss, 6),
        'epochs_run':    len(history['train_loss']),
        'train_loss':    [round(v, 6) for v in history['train_loss']],
        'val_loss':      [round(v, 6) for v in history['val_loss']],
        'config': {
            'architecture':    config.get('model', {}).get('architecture', ''),
            'in_channels':     config.get('model', {}).get('in_channels', ''),
            'features':        config.get('model', {}).get('features', ''),
            'dropout_p':       config.get('model', {}).get('dropout_p', ''),
            'batch_size':      cfg_tr.get('batch_size', ''),
            'learning_rate':   cfg_tr.get('learning_rate', ''),
            'weight_decay':    cfg_tr.get('weight_decay', ''),
            'optimizer':       cfg_tr.get('optimizer', ''),
            'loss':            cfg_tr.get('loss', ''),
            'grad_clip':       cfg_tr.get('grad_clip', ''),
            'epochs':          cfg_tr.get('epochs', ''),
            'channel_indices': config.get('data', {}).get('channel_indices', ''),
        },
    }
    with open(history_path, 'w', encoding='utf-8') as f:
        json.dump(history_out, f, indent=2)
    print(f"Historial guardado: {history_path}")

    return history
