import os
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Sin ventanas emergentes, solo guarda a disco
import matplotlib.pyplot as plt


def plot_training_curves(history: dict,
                         save_path: str = None,
                         title: str = "Curva de Aprendizaje"):
    """
    Dibuja las curvas de perdida de entrenamiento y validacion.

    Args:
        history:   Diccionario con claves 'train_loss' y 'val_loss'
        save_path: Ruta donde guardar la figura (None = no guarda)
        title:     Titulo del grafico
    """
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(history['train_loss'], label='Train Loss', linewidth=2)
    ax.plot(history['val_loss'],   label='Val Loss',   linewidth=2, linestyle='--')
    ax.set_title(title)
    ax.set_xlabel('Epoca')
    ax.set_ylabel('Loss (metros)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150)
        print(f"Curva guardada en: {save_path}")
    plt.close()


def plot_scatter(y_true: np.ndarray,
                 y_pred: np.ndarray,
                 metrics: dict,
                 save_path: str = None,
                 max_points: int = 10_000):
    """
    Scatter plot de prediccion vs realidad con linea 1:1.

    Args:
        y_true:     Valores reales (array 1D)
        y_pred:     Valores predichos (array 1D)
        metrics:    Diccionario de metricas para el titulo
        save_path:  Ruta de guardado
        max_points: Maximo de puntos a dibujar (muestreo aleatorio)
    """
    idx = np.random.choice(len(y_true),
                           size=min(max_points, len(y_true)),
                           replace=False)
    yt = y_true[idx]
    yp = np.maximum(y_pred[idx], 0)
    max_val = max(float(yt.max()), float(yp.max()), 0.5)

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.scatter(yt, yp, alpha=0.15, s=2, c='steelblue', label='Predicciones')
    ax.plot([0, max_val], [0, max_val], 'r--', lw=2, label='Linea 1:1')
    ax.set_xlabel('Nieve Real (LiDAR) [m]')
    ax.set_ylabel('Nieve Predicha [m]')
    ax.set_title(
        f"Prediccion vs Realidad\n"
        f"MAE={metrics['MAE']:.3f} m  |  "
        f"RMSE={metrics['RMSE']:.3f} m  |  "
        f"R2={metrics['R2']:.3f}  |  "
        f"NSE={metrics['NSE']:.3f}"
    )
    ax.set_xlim(0, max_val)
    ax.set_ylim(0, max_val)
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150)
        print(f"Scatter guardado en: {save_path}")
    plt.close()


def plot_predictions(images:   np.ndarray,
                     masks:    np.ndarray,
                     preds:    np.ndarray,
                     tile_ids: list,
                     n_samples: int = 3,
                     save_path: str = None):
    """
    Visualiza n_samples tiles con 4 paneles cada uno:
      DEM | Nieve Real | Prediccion | Mapa de Error

    Args:
        images:    Array (N, C, H, W) con los tiles de entrada
        masks:     Array (N, H, W) o (N, 1, H, W) con el ground truth
        preds:     Array (N, H, W) o (N, 1, H, W) con las predicciones
        tile_ids:  Lista de nombres de tile
        n_samples: Numero de ejemplos a visualizar
        save_path: Ruta de guardado
    """
    n = min(n_samples, len(images))
    fig, axes = plt.subplots(n, 4, figsize=(20, 5 * n))
    if n == 1:
        axes = axes[np.newaxis, :]

    for i in range(n):
        dem   = images[i, 0] * 1000 + 2100   # Des-normalizar DEM
        mask  = masks[i, 0] if masks.ndim == 4 else masks[i]
        pred  = np.maximum(
            preds[i, 0] if preds.ndim == 4 else preds[i], 0
        )
        error = pred - mask

        panels = [
            (dem,   'Topografia (DEM)',          'terrain', None,   None),
            (mask,  'Nieve REAL (LiDAR)',         'Blues',   0,      3.0),
            (pred,  'Nieve PREDICHA',             'Blues',   0,      3.0),
            (error, 'Error [m]\n(Rojo=+, Azul=-)', 'bwr',   -1.5,   1.5),
        ]

        for j, (data, title, cmap, vmin, vmax) in enumerate(panels):
            ax = axes[i, j]
            im = ax.imshow(data, cmap=cmap, vmin=vmin, vmax=vmax)
            ax.set_title(f"[{tile_ids[i]}]\n{title}" if j == 0 else title,
                         fontsize=9)
            ax.axis('off')
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150)
        print(f"Mapas guardados en: {save_path}")
    plt.close()
