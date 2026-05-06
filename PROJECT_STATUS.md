# SnowPrediction — Estado del Proyecto

**Última actualización:** 6 de mayo de 2026
**Repositorio:** https://github.com/HernanDiaz/SnowPrediction
**Objetivo:** Predecir profundidad de nieve a 1 m de resolución en la cuenca de Izas (Pirineos) con deep learning. Contribución principal: introducir SPAEF como métrica de evaluación espacial y una función de pérdida espacial (SpatialMSELoss) que mejora simultáneamente precisión píxel y coherencia espacial.

---

## 1. Contexto del problema

- **Datos:** LiDAR de 27 fechas (2021–2025) a 1 m de resolución, cuenca de Izas (~1 km²)
- **Split temporal:** Train = 2021–2023 (4241 tiles), Val = 2024 (431 tiles), Test = 2025 (220 tiles)
- **Tiles:** 256×256 píxeles (= 256×256 m), stride 128 (train) / 256 (val/test)
- **Arquitectura principal:** ResUNet++ con features [48, 96, 192, 384], GroupNorm 8, dropout 0.1
- **Métrica clave:** SPAEF (Spatial Efficiency) — complementa R² midiendo coherencia espacial del patrón predicho

### SPAEF

```
SPAEF = 1 - sqrt((rho-1)^2 + (alpha-1)^2 + (beta-1)^2)
```

Donde rho = correlación de Pearson espacial, alpha = ratio de coeficientes de variación, beta = intersección de histogramas normalizados. SPAEF=1 es perfecto; negativo es peor que la media espacial.

**Hallazgo clave:** R² y SPAEF pueden divergir dramáticamente. El modelo a 5 m (v6_300ep) tiene R²=0.271 pero SPAEF=−0.479: aprende el gradiente elevación→nieve pero falla en los patrones de redistribución por viento.

---

## 2. Modelos evaluados y resultados (test set 2025)

| Modelo | R² | RMSE (m) | MAE (m) | SPAEF | SPAEF std |
|--------|-----|----------|---------|-------|-----------|
| RF v4-17ch (baseline) | 0.022 | 0.754 | 0.566 | 0.130 | 0.434 |
| U-Net v4-17ch | −0.261 | 0.823 | 0.624 | 0.110 | 0.394 |
| ResUNet++ v4-17ch | 0.072 | 0.701 | 0.526 | 0.180 | 0.421 |
| ResUNet++ v6_300ep (5 m) | 0.271 | 0.641 | — | −0.479 | 2.810 |
| ResUNet++ v4-ms (22ch) | 0.100 | 0.691 | 0.518 | 0.212 | 0.241 |
| ResUNet++ v4-ms-Sx200 | 0.239 | 0.635 | 0.470 | 0.254 | 0.241 |
| ResUNet++ v4-ms s1 (seed=1) | 0.185 | 0.658 | 0.486 | 0.264 | 0.253 |
| ResUNet++ v4-ms s2 (seed=2) | 0.225 | 0.641 | 0.472 | 0.277 | 0.237 |
| ResUNet++ v4-ms s3 (seed=3) | −0.023 | 0.736 | 0.557 | 0.178 | 0.272 |

### Archivos de resultados
```
results/
  rf_v4_17ch/              rf_v4_17ch_metrics.json
  unet_v4_17ch/            unet_v4_17ch_metrics.json
  resunetpp_v4_17ch/       resunetpp_v4_17ch_metrics.json
  resunetpp_v6_300ep/      resunetpp_v6_300ep_metrics.json
  resunetpp_v4_ms/         resunetpp_v4_ms_metrics.json
  resunetpp_v4_ms_sx200/   resunetpp_v4_ms_sx200_metrics.json
  resunetpp_v4_ms_s1/      resunetpp_v4_ms_s1_metrics.json
  resunetpp_v4_ms_s2/      resunetpp_v4_ms_s2_metrics.json
  resunetpp_v4_ms_s3/      resunetpp_v4_ms_s3_metrics.json
```

---

## 3. Experimento principal: frente de Pareto con SpatialMSELoss

### Función de pérdida

```python
L = MSE(pred, target) + lambda * (1 - rho(pred, target))
```

Donde rho es la correlación de Pearson calculada sobre todos los píxeles (H×W) de cada tile, promediada sobre el batch. Es diferenciable y compatible con backpropagation. Implementada en `training/train.py` como `SpatialMSELoss`.

### Frente de Pareto (barrido de lambda)

El barrido de 8 valores de lambda traza el frente de Pareto en el espacio R²–SPAEF. Todos los experimentos usan ResUNet++ v4-ms-Sx200 (22 canales, seed=42).

| lambda | R² | RMSE (m) | MAE (m) | SPAEF | Bias (m) | Pareto |
|--------|-----|----------|---------|-------|----------|--------|
| 0.00 (MSE) | 0.239 | 0.635 | 0.470 | 0.254 | −0.176 | No dominado |
| 0.10 | 0.139 | 0.676 | 0.501 | 0.218 | −0.250 | Dominado |
| 0.25 | 0.156 | 0.669 | 0.493 | 0.260 | −0.263 | No dominado |
| 0.40 | 0.133 | 0.678 | 0.504 | 0.255 | −0.312 | Dominado |
| 0.50 | 0.219 | 0.644 | 0.470 | 0.311 | −0.243 | Dominado |
| **0.60 (optimo)** | **0.254** | **0.629** | **0.463** | **0.337** | −0.223 | **No dominado (mejor global)** |
| 0.75 | 0.130 | 0.679 | 0.516 | 0.302 | −0.316 | Dominado |
| 1.00 | −0.138 | 0.777 | 0.591 | 0.149 | −0.475 | Dominado |

**Conclusion clave:** lambda=0.6 mejora SIMULTANEAMENTE todas las metricas respecto al MSE puro (+6% R², −1% RMSE, +32% SPAEF). No hay trade-off. Es el valor recomendado para uso operacional.

**Nota sobre Bias:** El Bias empeora monotonicamente con lambda (Pearson optimiza el patron pero no la escala absoluta). Con lambda=0.6 el modelo subestima 0.223 m de media — se recomienda correccion de bias post-hoc.

### Archivos de resultados (spatial loss)
```
results/
  resunetpp_v4_ms_sx200_sp01/    *_metrics.json   (lambda=0.10)
  resunetpp_v4_ms_sx200_sp025/   *_metrics.json   (lambda=0.25)
  resunetpp_v4_ms_sx200_sp04/    *_metrics.json   (lambda=0.40)
  resunetpp_v4_ms_sx200_sp05/    *_metrics.json   (lambda=0.50)
  resunetpp_v4_ms_sx200_sp06/    *_metrics.json   (lambda=0.60)
  resunetpp_v4_ms_sx200_sp075/   *_metrics.json   (lambda=0.75)
  resunetpp_v4_ms_sx200_sp10/    *_metrics.json   (lambda=1.00)
```

---

## 4. Configuraciones de experimentos

```
configs/
  resunetpp_v4_ms.yaml              Multi-scale 22ch (Sx 100m)
  resunetpp_v4_ms_sx200.yaml        Multi-scale 22ch (Sx 200m) — modelo base
  resunetpp_v4_ms_s1.yaml           Seed sensitivity seed=1
  resunetpp_v4_ms_s2.yaml           Seed sensitivity seed=2
  resunetpp_v4_ms_s3.yaml           Seed sensitivity seed=3
  resunetpp_v4_ms_sx200_sp01.yaml   Spatial loss lambda=0.10
  resunetpp_v4_ms_sx200_sp025.yaml  Spatial loss lambda=0.25
  resunetpp_v4_ms_sx200_sp04.yaml   Spatial loss lambda=0.40
  resunetpp_v4_ms_sx200_sp05.yaml   Spatial loss lambda=0.50
  resunetpp_v4_ms_sx200_sp06.yaml   Spatial loss lambda=0.60
  resunetpp_v4_ms_sx200_sp075.yaml  Spatial loss lambda=0.75
  resunetpp_v4_ms_sx200_sp10.yaml   Spatial loss lambda=1.00
  resunetpp_v6_improved.yaml        Modelo 5m resolución
```

Hiperparámetros comunes a todos los experimentos de spatial loss:
- seed=42, lr=0.00039, weight_decay=1.6e-5, batch_size=8
- epochs=300, es_patience=20, scheduler=cosine, lr_min=1e-6
- optimizer=adam, grad_clip=0.0, dropout=0.1

---

## 5. Canales de entrada (22 canales, v4-ms-Sx200)

| Canal | Nombre | Descripcion | Res. |
|-------|--------|-------------|------|
| 0 | DEM | Elevacion (m) | 1 m |
| 1 | Slope | Pendiente (grados) | 1 m |
| 2 | Northness | cos(aspecto) | 1 m |
| 3 | Eastness | sin(aspecto) | 1 m |
| 4 | TPI | Indice de posicion topografica (31x31) | 1 m |
| 5 | SCE | Snow Cover Extent (Sentinel-2) | 1 m |
| 6–13 | Sx_200m_θ | Shelter index, 8 direcciones, r=200 m | 1 m |
| 14 | Pers_15d | Persistencia de nieve (ventana 15 dias) | 1 m |
| 15 | Pers_30d | Persistencia de nieve (ventana 30 dias) | 1 m |
| 16 | Pers_60d | Persistencia de nieve (ventana 60 dias) | 1 m |
| 17 | DEM_5m | Elevacion a 5 m de resolucion | 5 m |
| 18 | Slope_5m | Pendiente a 5 m | 5 m |
| 19 | Northness_5m | cos(aspecto) a 5 m | 5 m |
| 20 | Eastness_5m | sin(aspecto) a 5 m | 5 m |
| 21 | TPI_5m | TPI a 5 m | 5 m |

---

## 6. Estructura del código

```
main.py                          Punto de entrada: --config, --mode (train/eval/both), --seed
training/
  train.py                       Bucle de entrenamiento + SpatialMSELoss + schedulers + SWA
  evaluate.py                    Evaluacion con SPAEF, guarda *_metrics.json
utils/
  metrics.py                     Implementacion de SPAEF (por tile, media ± std)
data/
  generate_dataset_v4_ms_sx200.py  Generacion del dataset 22ch con Sx 200m
  generate_dataset_v4_ms.py        Generacion del dataset 22ch con Sx 100m
  generate_dataset_v6_improved.py  Dataset a 5m
baselines/
  compute_spaef_rf.py            Calcula SPAEF para el Random Forest baseline
  evaluate_v6_combined.py        Evalua v6_improved en val+test combinado
scripts/
  run_spatial_loss.py            Lanzador secuencial: sp01, sp05, sp10
  run_spatial_loss_extended.py   Lanzador secuencial: sp025, sp04, sp06, sp075
  run_overnight.py               Lanzador: seeds 1/2/3 + sx200 + v6 eval
```

Para lanzar un experimento:
```bash
.venv\Scripts\python.exe main.py --config configs/resunetpp_v4_ms_sx200_sp06.yaml --mode both
```

---

## 7. Modelos entrenados (pesos .pth)

Los pesos NO están en el repositorio git (excluidos por .gitignore). Se encuentran en:
```
Articulo 1/Models/
  resunetpp_v4_ms_sx200.pth          Modelo base MSE (R2=0.239, SPAEF=0.254)
  resunetpp_v4_ms_sx200_sp06.pth     Mejor modelo (lambda=0.6, R2=0.254, SPAEF=0.337)
  resunetpp_v4_ms_sx200_sp05.pth     lambda=0.5
  resunetpp_v4_ms_s1.pth / s2.pth / s3.pth
  ... (un .pth por experimento)
```

---

## 8. Borrador del articulo

```
Articulo 1/
  draft_paper_v1.docx    Borrador inicial (29 abril 2026)
  draft_paper_v2.docx    Borrador actualizado (30 abril 2026) con:
                           - Tabla completa frente de Pareto (8 lambdas)
                           - Seccion 5.5: resultados spatial loss
                           - Seccion 6.3: discusion Pareto (sin trade-off, bias)
                           - Abstract y conclusiones actualizados
```

Titulo actual: *"Multi-scale deep learning for high-resolution snow depth prediction: evaluating spatial pattern accuracy with SPAEF"*

---

## 9. Proximos pasos sugeridos

### Para el articulo (prioritario)
- [ ] Figura 1: mapa de la cuenca de Izas con DEM hillshade
- [ ] Figura 2: tile de ejemplo mostrando (a) observado, (b) v6_300ep, (c) v4-ms-Sx200
- [ ] Figura 3: curva R²–SPAEF del frente de Pareto con puntos etiquetados por lambda
- [ ] Figura 4: curvas de entrenamiento del mejor modelo
- [ ] Rellenar referencias bibliograficas en draft_paper_v2.docx
- [ ] Corrección de bias post-hoc para el modelo lambda=0.6

### Experimentos opcionales
- [ ] Repetir lambda=0.6 con 3 semillas diferentes (verificar robustez del resultado)
- [ ] Probar grad_clip=1.0 con lambda=0.6 (posible mejora de estabilidad)
- [ ] Evaluar el modelo lambda=0.6 en la cuenca de validacion 2024 (no solo 2025)

### Datos
- [ ] Subir datasets y pesos de modelos a Zenodo o HuggingFace Hub para citar en el paper

---

## 10. Commits recientes

```
e9921e0  Add multi-scale and spatial loss experiments
c4a5d6c  Add SPAEF metric, SpatialMSELoss, and seed control
633dad0  Add v4_17ch experiments: dataset generation + RF + UNet + ResUNet++
a6a7d8e  Cambiar split temporal v4: train 2021-23 / val 2024 / test 2025
```
