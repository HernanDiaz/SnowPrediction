import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def compute_spaef(obs: np.ndarray, sim: np.ndarray, n_bins: int = 100) -> float:
    """
    SPAEF - Spatial Efficiency (Koch et al., 2018).

    Mide la similitud espacial entre dos campos considerando:
        rho   = correlacion de Pearson (patron espacial)
        alpha = ratio de CV  (CV_sim / CV_obs)
        beta  = interseccion de histogramas normalizados

    SPAEF = 1 - sqrt((rho-1)^2 + (alpha-1)^2 + (beta-1)^2)

    Rango: (-inf, 1].  SPAEF=1: perfecto,  SPAEF=0: sin habilidad.
    Complementa R2: R2 mide varianza pixel a pixel; SPAEF mide similitud
    de patrones espaciales y distribucion de valores.

    Args:
        obs:    Valores observados  (1D, pixeles validos de un tile/cuenca)
        sim:    Valores simulados   (1D, mismos pixeles)
        n_bins: Bins para histograma (defecto: 100)

    Returns:
        Valor SPAEF (float), nan si datos insuficientes.

    Ref: Koch et al. (2018), J. Geophys. Res. Atmospheres.
    """
    obs = np.asarray(obs, dtype=np.float64)
    sim = np.asarray(sim, dtype=np.float64)
    sim = np.maximum(sim, 0.0)

    if len(obs) < 10:
        return float('nan')

    # rho: correlacion de Pearson
    rho = float(np.corrcoef(obs, sim)[0, 1])
    if np.isnan(rho):
        return float('nan')

    # alpha: ratio de coeficientes de variacion
    mean_obs = float(np.mean(obs))
    mean_sim = float(np.mean(sim))
    if mean_obs == 0.0 or mean_sim == 0.0:
        return float('nan')
    cv_obs = float(np.std(obs)) / mean_obs
    cv_sim = float(np.std(sim)) / mean_sim
    if cv_obs == 0.0:
        return float('nan')
    alpha = cv_sim / cv_obs

    # beta: interseccion de histogramas normalizados
    lo = min(float(obs.min()), float(sim.min()))
    hi = max(float(obs.max()), float(sim.max()))
    if hi <= lo:
        return float('nan')
    bins = np.linspace(lo, hi, n_bins + 1)
    h_obs, _ = np.histogram(obs, bins=bins)
    h_sim, _ = np.histogram(sim, bins=bins)
    h_obs = h_obs / (h_obs.sum() + 1e-10)
    h_sim = h_sim / (h_sim.sum() + 1e-10)
    beta = float(np.sum(np.minimum(h_obs, h_sim)))

    return float(1.0 - np.sqrt((rho - 1.0)**2 + (alpha - 1.0)**2 + (beta - 1.0)**2))


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
    unit_fields  = {'MAE', 'RMSE', 'Bias', 'mean_train'}
    skip_fields  = {'SPAEF_std', 'SPAEF_n_tiles'}   # se muestran junto a SPAEF
    for k, v in metrics.items():
        if k in skip_fields:
            continue
        unit = " m" if k in unit_fields else ""
        if k == 'SPAEF':
            n    = metrics.get('SPAEF_n_tiles', '?')
            std  = metrics.get('SPAEF_std', float('nan'))
            print(f"  {k:<12}: {v}  (std={std}, n={n} tiles)")
        else:
            print(f"  {k:<12}: {v}{unit}")
    print(f"{line}\n")
