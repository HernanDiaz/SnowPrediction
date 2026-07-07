# SnowPrediction — Estado del Proyecto

**Última actualización:** 25 de mayo de 2026
**Objetivo:** Predecir profundidad de nieve a 1 m de resolución en la cuenca de Izas (Pirineos) con deep learning. Contribución principal: introducir SPAEF como métrica de evaluación espacial y una función de pérdida espacial (SpatialMSELoss) que mejora simultáneamente precisión píxel y coherencia espacial.

---

## 1. Contexto del problema

- **Datos:** LiDAR de 27 fechas (2021–2025) a 1 m de resolución, cuenca de Izas (~1 km²)
- **Split temporal:** Train = 2021–2023 (15 fechas, 4241 tiles), Val = 2024 (7 fechas, 431 tiles), Test = 2025 (5 fechas, 220 tiles)
- **Tiles:** 256×256 píxeles (= 256×256 m), stride 128 (train) / 256 (val/test)
- **Dataset:** dataset_v4_ms_sx200 (22 canales, 1m, Sx_200m)

### Distribución de profundidad de nieve por split

| Split | Media (m) | Mediana (m) | P90 (m) | Cobertura |
|-------|----------:|------------:|--------:|-----------|
| Train (2021-23) | 0.804 | 0.639 | 1.728 | 80.9% |
| Val (2024) | 0.819 | 0.617 | 1.743 | 74.8% |
| **Test (2025)** | **1.160** | **1.095** | **2.109** | 82.8% |

**Nota:** 2025 es un año excepcionalmente nevado (+44% respecto a train). Esto explica la dificultad de generalización y el bajo R² en test.

### SPAEF

```
SPAEF = 1 - sqrt((rho-1)^2 + (alpha-1)^2 + (beta-1)^2)
```

Donde rho = correlación de Pearson espacial, alpha = ratio de CV, beta = intersección de histogramas normalizados. SPAEF=1 es perfecto; negativo es peor que la media espacial.

---

## 2. Resultados completos (test set 2025)

### Baselines

| Modelo | R² | RMSE (m) | MAE (m) | SPAEF | Notas |
|--------|-----|----------|---------|-------|-------|
| RF v4-17ch | 0.022 | 0.754 | 0.566 | 0.130 | 5 canales topo |
| RF Optuna 22ch | 0.024 | 0.719 | 0.556 | -0.113 | train+val, SPAEF peor |

### UNet HPO (dataset_v4_ms_sx200, 50 épocas, seed=42)

| Checkpoint | R² | RMSE | MAE | SPAEF |
|------------|-----|------|-----|-------|
| Best val | -0.039 | 0.742 | 0.568 | 0.177 |
| **Last epoch** | **0.362** | **0.582** | **0.432** | **0.346** |

- Config: `configs/unet_v4_ms_sx200_hpo.yaml`
- Params: base=48, lr=3.064e-5, adam, bs=16, dropout≈0, wd=2.149e-5
- **Usar last epoch** — best val no es buen proxy para test en UNet

### ResUNet++ HPO (dataset_v4_ms_sx200, 50 épocas, 3 seeds)

| Experimento | R² media | R² std | SPAEF media | SPAEF std |
|-------------|--------:|-------:|------------:|----------:|
| Full 22ch (seeds 42,123,7) | 0.132 | 0.112 | 0.251 | 0.051 |

- Config base: `configs/resunetpp_v4_ms_sx200_hpo.yaml`
- Params: base=64, lr=1.287e-4, adamw, bs=8, dropout=0.077, wd=1.239e-4
- **Varianza muy alta entre seeds** (R² std=0.112) — problema fundamental del dataset

### Barrido lambda — base=48/adam (experimento original, 1 seed)

| lambda | R² | RMSE | MAE | SPAEF |
|--------|-----|------|-----|-------|
| 0.00 (MSE) | 0.239 | 0.635 | 0.470 | 0.254 |
| 0.10 | 0.139 | 0.676 | 0.501 | 0.218 |
| 0.25 | 0.156 | 0.669 | 0.493 | 0.260 |
| 0.40 | 0.133 | 0.678 | 0.504 | 0.255 |
| 0.50 | 0.219 | 0.644 | 0.470 | 0.311 |
| **0.60** | **0.254** | **0.629** | **0.463** | **0.337** |
| 0.75 | 0.130 | 0.679 | 0.516 | 0.302 |
| 1.00 | -0.138 | 0.777 | 0.591 | 0.149 |

- Configs: `configs/resunetpp_v4_ms_sx200_sp*.yaml`
- Results: `results/resunetpp_v4_ms_sx200_sp*/`
- **Nota:** hiperparámetros NO optimizados con Optuna

### Barrido lambda — HPO (base=64/adamw, seed=42 + seeds 123,7 EN CURSO)

| lambda | R² (s42) | SPAEF (s42) | Estado |
|--------|--------:|------------:|--------|
| 0.00 | 0.244 | 0.298 | ✓ (3 seeds en ablación) |
| 0.10 | 0.177 | 0.244 | s42 ✓, s123+s7 en curso |
| 0.25 | 0.262 | 0.311 | s42 ✓, s123+s7 en curso |
| 0.40 | 0.186 | 0.294 | s42 ✓, s123+s7 en curso |
| 0.50 | 0.210 | 0.293 | s42 ✓, s123+s7 en curso |
| 0.60 | 0.083 | 0.191 | s42 ✓, s123+s7 en curso |
| 0.75 | 0.250 | 0.329 | s42 ✓, s123+s7 en curso |
| 1.00 | 0.169 | 0.263 | s42 ✓, s123+s7 en curso |

- Configs seeds adicionales: `configs/resunetpp_hpo_sp*_s{123,7}.yaml`
- Results: `results/lambda_sweep_hpo/`
- Script: `scripts/run_lambda_sweep_hpo_seeds.py`

---

## 3. Estudio de ablación de canales

ResUNet++ HPO (base=64, adamw, 50ep), leave-one-group-out, 3 seeds (42, 123, 7).

| Grupo eliminado | Canales | R² media | R² std | SPAEF media | SPAEF std |
|----------------|---------|--------:|-------:|------------:|----------:|
| Ninguno (full) | 22 | 0.132 | 0.112 | 0.251 | 0.051 |
| Topo_5m (17-21) | 17 | -0.114 | 0.119 | 0.076 | 0.080 |
| Persistencia (14-16) | 19 | -0.148 | 0.113 | 0.129 | 0.026 |
| Sx_200m (6-13) | 14 | 0.089 | 0.097 | 0.225 | 0.044 |
| Topo_1m (0-4) | 17 | 0.079 | 0.092 | 0.220 | 0.048 |
| SCE (5) | 21 | 0.132 | 0.027 | 0.242 | 0.022 |

**Conclusiones:**
- **Topo_5m y Persistencia son críticos** — su eliminación hunde el modelo
- **SCE apenas aporta** — eliminar el canal de cobertura de nieve no cambia nada
- **Sx y Topo_1m son importantes** pero el modelo se recupera parcialmente sin ellos

- Configs: `configs/resunetpp_ablation_*.yaml`
- Results: `results/ablation/`
- Script: `scripts/run_ablation.py`

---

## 4. Canales de entrada (22 canales, v4-ms-Sx200)

| Canal | Nombre | Descripcion | Res. |
|-------|--------|-------------|------|
| 0 | DEM | Elevacion (m) | 1 m |
| 1 | Slope | Pendiente (grados) | 1 m |
| 2 | Northness | cos(aspecto) | 1 m |
| 3 | Eastness | sin(aspecto) | 1 m |
| 4 | TPI | Indice de posicion topografica | 1 m |
| 5 | SCE | Snow Cover Extent (Sentinel-2) | 1 m |
| 6–13 | Sx_200m_θ | Shelter index, 8 direcciones, r=200m | 1 m |
| 14 | Pers_15d | Persistencia nieve 15 dias | 1 m |
| 15 | Pers_30d | Persistencia nieve 30 dias | 1 m |
| 16 | Pers_60d | Persistencia nieve 60 dias | 1 m |
| 17 | DEM_5m | Elevacion a 5m | 5 m |
| 18 | Slope_5m | Pendiente a 5m | 5 m |
| 19 | Northness_5m | cos(aspecto) a 5m | 5 m |
| 20 | Eastness_5m | sin(aspecto) a 5m | 5 m |
| 21 | TPI_5m | TPI a 5m | 5 m |

---

## 5. Optuna HPO (completados 13/05/2026)

### UNet Optuna (optuna_unet_v4ms)
- 29 completados + 2 pruned | dataset_v4_ms_sx200 | 50 épocas
- **Mejor: Trial 11** | R²=0.351, RMSE=0.587, MAE=0.436
- Params: base=48, lr=3.06e-5, adam, gc=1.0, bs=16, dropout≈0, wd≈1e-5
- DB: `results/optuna_unet_v4ms/`

### ResUNet++ Optuna (optuna_resunetpp_v4ms)
- 24 completados + 6 pruned | dataset_v4_ms_sx200 | 50 épocas
- **Mejor: Trial 17** | R²=0.2473, MAE=0.468, RMSE=0.632, Bias=-0.224
- Params: base=64, lr=1.287e-4, wd=1.239e-4, adamw, gc=1.0, **bs=8** (no 16), dropout=0.077
- DB: `results/optuna_resunetpp_v4ms/`

### RF Optuna (optuna_rf_v4_ms_sx200)
- 30 trials | dataset_v4_ms_sx200 | 22 canales
- **Mejor: Trial 18** | val_R²=-0.011
- Params: n_est=500, max_depth=10, min_samples_leaf=10, max_features=sqrt
- Test: R²=0.024, RMSE=0.719, SPAEF=-0.113
- **Nota:** entrenado en train+val (metodología RF estándar)
- Results: `results/optuna_rf_v4_ms_sx200/`

---

## 6. Estructura de código

```
main.py                          Entrada: --config, --mode (train/eval/both)
training/
  train.py                       Bucle entrenamiento + SpatialMSELoss
  evaluate.py                    Evaluacion SPAEF + metricas, guarda _metrics.json y _last_metrics.json
utils/
  metrics.py                     SPAEF, MSPAEF, compute_metrics
data/
  dataset.py                     SnowDataset — BUGFIX 23/05/2026: normalizar antes de channel_indices
baselines/
  optuna_rf_v4_ms_sx200.py       RF Optuna 22ch
  compute_spaef_rf_missing.py    Calcula SPAEF para modelos RF sin esa metrica
scripts/
  run_ablation.py                Ablacion canales (17 experimentos)
  run_ablation_rerun.py          Re-ejecucion de 9 experimentos afectados por bug
  run_resunetpp_hpo_spatial_loss.py   Barrido lambda HPO seed=42
  run_lambda_sweep_hpo_seeds.py       Barrido lambda HPO seeds 123+7 (EN CURSO)
```

**Para lanzar un experimento:**
```bash
.venv\Scripts\python.exe main.py --config configs/<nombre>.yaml --mode both
```

---

## 7. Bugs corregidos

### Bug normalización dataset.py (23/05/2026)
- **Problema:** `channel_indices` se aplicaba ANTES de `_normalize()`, causando que la normalización usara posiciones incorrectas cuando se eliminaban canales del inicio del array (ej. sin_topo1: SCE en posición 0 recibía normalización DEM → NaN en GroupNorm)
- **Fix:** Normalizar todos los canales ANTES de seleccionar subconjunto. También `_normalize` usa `image.shape[0]` en vez de `self.n_channels`
- **Experimentos afectados (re-ejecutados):** sin_sx, sin_sce, sin_topo1

### Bug batch_size ResUNet++ Optuna
- El script de Optuna tenía guardia OOM: `if bs==16 and base_ch>=48: bs=8`
- Trial 17 sugirió bs=16 pero entrenó con bs=8 → config HPO corregido a bs=8

---

## 8. Notas críticas

- **Val loss NO es buen proxy para test** — 2025 es un año atípico (+44% nieve vs train)
- **Solo 15 fechas de train** → varianza interanual enorme, R² std≈0.11 entre seeds
- **UNet: usar last epoch** (best val da R²=-0.039)
- **ResUNet++: usar best val** (last epoch peor en la mayoría de casos)
- **Seed fijada justo antes de build_model()** en main.py para reproducibilidad
- **SCE no aporta** al modelo CNN según ablación (aunque RF lo valora en 16%)
- **Topo_5m y Persistencia son los canales más importantes** según ablación

---

## 9. Próximos pasos

- [ ] Completar barrido lambda HPO seeds 123+7 (~14h, EN CURSO)
- [ ] Analizar resultados lambda sweep HPO con 3 seeds y determinar lambda óptimo
- [ ] Figuras del paper (curvas Pareto, mapas, ablación)
- [ ] Solicitar datos meteorológicos a Jesús Revuelto (SWE, Tª, precipitación 2020-2025)
- [ ] Subir datos/pesos a Zenodo o HuggingFace
- [ ] Escribir/actualizar borrador del paper con nuevos resultados

---

## 10. Commits recientes

```
3082b6e  Add PROJECT_STATUS.md with full experiment summary
e9921e0  Add multi-scale and spatial loss experiments
c4a5d6c  Add SPAEF metric, SpatialMSELoss, and seed control
633dad0  Add v4_17ch experiments
a6a7d8e  Cambiar split temporal v4
```
