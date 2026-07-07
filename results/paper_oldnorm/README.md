# paper_oldnorm — Resultados aislados con normalización VIEJA (norm_extended=False)

Conjunto consistente y **verificado reproducible** para el paper, todo con la
**misma normalización** (topografía 5m, canales 17-21, en CRUDO — comportamiento
anterior al parche de `_normalize`). Cada métrica se ha regenerado cargando el
peso correspondiente y evaluando con norm vieja, comparando con el valor guardado.

## Verificación (eval con norm vieja vs valor guardado)
- **Barrido λ (ResUNet++):** 22/24 reproducen EXACTO (dR²≈0.000).
  - Excepción: `sp00_s123`, `sp00_s7` → entrenados con norm NUEVA (NO incluidos).
  - λ=0.1…1.0: 3 seeds limpios cada uno. λ=0: solo 1 seed old (s42). λ=0.4: 3/3 ✅.
- **Ablación (ResUNet++):** 18/18 reproducen EXACTO ✅.
- **Comparativa temporal:** RF (norm-invariante), U-Net GN (old, optA), ResUNet++ λ=0.4.
- **Espacial:** ResUNet++ entrenado con norm vieja (optA).

## Resultados consolidados (3 seeds, media ± std ddof=1)

### Comparativa temporal (test 2025)
| Modelo | R² | SPAEF | MAE | RMSE | Bias |
|---|---|---|---|---|---|
| RF v6 | +0.140 ± 0.004 | +0.107 ± 0.003 | 0.515 | 0.675 | −0.307 |
| U-Net (GroupNorm) | +0.138 ± 0.101 | +0.254 ± 0.024 | 0.511 | 0.675 | −0.288 |
| ResUNet++ (λ=0.4) | +0.208 ± 0.027 | +0.297 ± 0.018 | 0.477 | 0.648 | −0.282 |

### Espacial (ResUNet++ λ=0.4, norm vieja)
R² = +0.103 ± 0.100 | SPAEF = +0.282 ± 0.026

## Estructura
- `lambda/{tag}_s{seed}/metrics.json` — barrido λ (22 verificados)
- `ablation/{group}_s{seed}/metrics.json` — ablación (18)
- `comparison/{model}_s{seed}/metrics.json` — comparativa (RF, U-Net GN, ResUNet++)
- `spatial/resunetpp_spatial_s{seed}/metrics.json` — espacial old
- `manifest_lambda.json`, `manifest_ablation.json` — peso de origen + repro por run
- `summary_oldnorm.json` — agregados

## Pendiente / notas
- λ=0 del barrido: solo 1 seed old (s123/s7 eran norm nueva). Reportar n=1 o documentar.
- Los pesos NO se duplican aquí (≈2 GB); el manifiesto apunta a su ruta en `Articulo 1/Models`.
- La normalización de cada experimento se controla con `data.norm_extended` (false = esta carpeta).
