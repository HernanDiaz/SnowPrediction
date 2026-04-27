import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """
    Calcula el conjunto completo de metricas para prediccion de nieve.

    Args:
        y_true: Valores reales  (array 1D de pixeles validos)
        y_pred: Valores predichos (array 1D)

    Returns:
        Diccionario con MAE, RMSE, R2, NSE y Bias
    """
    y_pred = np.maximum(y_pred, 0.0)  # Restriccion fisica: nieve >= 0

    mae  = float(mean_absolute_error(y_true, y_pred))
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    r2   = float(r2_score(y_true, y_pred))
    bias = float(np.mean(y_pred - y_true))

    # Nash-Sutcliffe Efficiency
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    nse = float(1.0 - ss_res / ss_tot) if ss_tot > 0 else float('nan')

    return {
        'MAE':  round(mae,  4),
        'RMSE': round(rmse, 4),
        'R2':   round(r2,   4),
        'NSE':  round(nse,  4),
        'Bias': round(bias, 4),
    }


def compute_naive_benchmark(train_values: np.ndarray,
                             test_values:  np.ndarray) -> dict:
    """
    Benchmark naive: predice siempre la media de entrenamiento.
    Sirve como cota inferior de referencia para comparar modelos.

    Args:
        train_values: Pixeles validos del conjunto de entrenamiento
        test_values:  Pixeles validos del conjunto de test

    Returns:
        Diccionario con metricas del predictor naive + media usada
    """
    mean_train   = float(np.mean(train_values))
    y_pred_naive = np.full_like(test_values, fill_value=mean_train)
    metrics      = compute_metrics(test_values, y_pred_naive)
    metrics['mean_train'] = round(mean_train, 4)
    return metrics


def print_metrics(metrics: dict, title: str = "Resultados"):
    """Imprime las metricas con formato legible."""
    line = "=" * 48
    print(f"\n{line}")
    print(f"  {title}")
    print(line)
    unit_fields = {'MAE', 'RMSE', 'Bias', 'mean_train'}
    for k, v in metrics.items():
        unit = " m" if k in unit_fields else ""
        print(f"  {k:<12}: {v}{unit}")
    print(f"{line}\n")
