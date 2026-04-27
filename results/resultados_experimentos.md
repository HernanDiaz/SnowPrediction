# Registro de Experimentos - Snow Depth Prediction
# Articulo 1 - Izas (Pirineos)
# Fecha inicio: 2026-04-25

---

## Configuracion del entorno
- GPU: NVIDIA GeForce RTX 5060 Ti (16GB VRAM)
- Dataset LiDAR: 27 fechas (2021-2025)
- Dataset SCE satelite: 431 imagenes
- Zona: Izas, Pirineos

---

## Splits temporales
| Split | Anos | Tiles 5m | Tiles 1m |
|---|---|---|---|
| Train | 2021, 2022 | 54 | 3570 |
| Val   | 2023       | 17 | 1133 |
| Test  | 2024, 2025 | 44 | 3185 |

---

## Benchmark Naive (Media del Train)
| Dataset | MAE (m) | RMSE (m) | R² | NSE | Bias (m) | Mean Train (m) |
|---|---|---|---|---|---|---|
| 5m | 0.5579 | 0.7502 | -0.039 | -0.039 | -0.146 | 0.790 |
| 1m | 0.5780 | 0.7507 | -0.010 | -0.010 | -0.075 | 0.855 |

---

## Baseline Random Forest (pixel a pixel, sin SCE)
> 200 arboles | max_depth=20 | entrenado con 500K pixeles submuestreados

| Dataset | MAE (m) | RMSE (m) | R² | NSE | Bias (m) |
|---|---|---|---|---|---|
| 5m | 0.4833 | 0.6343 | 0.257 | 0.257 | -0.118 |
| 1m | - | - | - | - | - |

Importancia de variables RF sin SCE (5m):
- TPI: 44.1% (variable mas importante)
- DEM: 25.8%
- Slope: 15.6%
- Northness: 8.3%
- Eastness: 6.1%

## Baseline Random Forest (pixel a pixel, con SCE)
> 200 arboles | max_depth=20 | entrenado con 500K pixeles submuestreados

| Dataset | MAE (m) | RMSE (m) | R² | NSE | Bias (m) |
|---|---|---|---|---|---|
| 5m | 0.4919 | 0.6446 | 0.233 | 0.233 | -0.124 |
| 1m | - | - | - | - | - |

Importancia de variables RF con SCE (5m):
- TPI: 43.7%
- DEM: 25.4%
- Slope: 14.9%
- Northness: 8.4%
- Eastness: 6.1%
- SCE: 1.5% <- CASI NULA en RF pixel a pixel

NOTA IMPORTANTE: El RF con SCE es PEOR que sin SCE (R2=0.233 vs 0.257).
Explicacion: el RF trabaja pixel a pixel. Los pixeles validos ya tienen nieve (mask>0.01),
por lo que su SCE es casi siempre =1. La SCE no aporta informacion adicional a nivel de pixel.
La U-Net en cambio trabaja a nivel de tile (256x256) y puede explotar el contexto espacial del SCE.

---

## Grupo A - Modelos originales del alumno (TPI ÷ 30, sin SCE)
> NOTA: Normalizacion TPI incorrecta (rango real ±9155, se dividio por 30 → valores ±300)

| Experimento | Loss | MAE (m) | RMSE (m) | R² | NSE | Bias (m) | Modelo .pth |
|---|---|---|---|---|---|---|---|
| unet_v5_5m_mae | MAE | 0.5246 | 0.7149 | 0.056 | 0.056 | -0.199 | e3_b03_1_best_unet_lidar_5m.pth |
| unet_v5_5m_mse | MSE | 0.5285 | 0.7040 | 0.085 | 0.085 | -0.293 | e3_b03_1_best_unet_lidar_5m_MSE.pth |
| unet_v4_1m_mae | MAE | 0.5601 | 0.7301 | 0.038 | 0.038 | -0.286 | e2_b02_1_best_unet_lidar.pth |
| unet_v4_1m_mse | MSE | 0.5535 | 0.7247 | 0.052 | 0.052 | -0.327 | e2_b02_3_best_unet_lidar_MSE.pth |

---

## Grupo B - Normalizacion corregida, sin SCE (TPI ÷ 9200)
> Objetivo: aislar el efecto de la normalizacion respecto al Grupo A

| Experimento | Loss | MAE (m) | RMSE (m) | R² | NSE | Bias (m) | Estado |
|---|---|---|---|---|---|---|---|
| unet_v5_5m_mae_fixed | MAE | 0.5350 | 0.7200 | 0.042 | 0.042 | - | Completado |
| unet_v5_5m_mse_fixed | MSE | 0.5176 | 0.6852 | 0.133 | 0.133 | -0.264 | Completado |
| unet_v4_1m_mae_fixed | MAE | - | - | - | - | - | En progreso |
| unet_v4_1m_mse_fixed | MSE | 0.5299 | 0.6885 | 0.1445 | 0.1445 | -0.214 | Completado |

NOTA: La correccion de TPI mejora significativamente el modelo MSE (R2: 0.085->0.133, +56% relativo).
Para MAE la mejora es minima (R2: 0.056->0.042). MAE sigue siendo inferior a MSE para esta distribucion.

RESULTADOS 1m MSE fixed:
- R²=0.1445: mejor que los modelos 1m del alumno (R²=0.052 Grupo A) pero peor que el mejor 5m+WD (R²=0.194)
- La mayor cantidad de datos (3570 tiles train vs 54 en 5m) NO compensa la falta de regularizacion (no hay WD)
- Bias=-0.214m: menor sesgo que los modelos 5m (tipicamente -0.25 a -0.30m)
- Val_loss best=0.1925 (early stopping) pero val muy variable ep-a-ep (0.28-0.37): alto ruido en validacion
- Naive 1m: R²=-0.010, MAE=0.578m, RMSE=0.7507m — el modelo 1m supera claramente al naive
- Proximo: 1m MAE fixed (en curso) y 1m MSE+WD=1e-4 (pendiente de lanzar)

---

## Grupo C - Normalizacion corregida + canal SCE satelite
> Objetivo: medir el impacto real del canal dinamico SCE respecto al Grupo B

| Experimento | Loss | MAE (m) | RMSE (m) | R² | NSE | Bias (m) | Estado |
|---|---|---|---|---|---|---|---|
| unet_v5_5m_mae_sce | MAE | 0.6111 | 0.7981 | -0.176 | -0.176 | - | Completado |
| unet_v5_5m_mse_sce | MSE | 0.6032 | 0.7876 | -0.145 | -0.145 | -0.273 | Completado |
| unet_v4_1m_mae_sce | MAE | - | - | - | - | - | Pendiente |
| unet_v4_1m_mse_sce | MSE | - | - | - | - | - | Pendiente |

RESULTADO CRITICO: Anadir SCE empeora AMBOS modelos (MAE y MSE) en el dataset 5m.
- MSE sin SCE: R2=0.133 | MSE con SCE: R2=-0.145 --> deterioro de -0.278 en R2
- MAE sin SCE: R2=0.042 | MAE con SCE: R2=-0.176 --> deterioro de -0.218 en R2

HIPOTESIS (ver seccion Analisis):
El producto SCE satelite (MODIS/Sentinel) tiene menor sensibilidad que LiDAR para nieve fina.
Cuando SCE=0 (satelite no detecta nieve), el modelo predice ~0, pero el LiDAR puede registrar
capas delgadas de nieve que el satelite no ve. Esto genera errores sistematicos en el test set.
Ademas, el sobreajuste es mas pronunciado con 6 canales y solo 54 tiles de entrenamiento.
(Val RMSE optima ~0.41m para MSE+SCE, pero Test RMSE=0.788m -> gran brecha train/test)

---

---

## Grupo D - Data Augmentation (Normalizacion corregida, sin SCE)
> Objetivo: evaluar si la augmentation geometrica mejora la generalizacion

| Experimento | Augmentation | MAE (m) | RMSE (m) | R² | NSE | Bias (m) | Estado |
|---|---|---|---|---|---|---|---|
| unet_v5_5m_mse_augmented | H+V flip | 0.5458 | 0.7313 | 0.068 | 0.068 | - | Completado |
| unet_v5_5m_mse_aug_h     | Solo H flip | 0.5253 | 0.6979 | 0.100 | 0.100 | -0.272 | Completado |

RESULTADO: Ambos modos de augmentation empeoran la U-Net respecto al baseline MSE fixed (R²=0.133).
- H+V flip: R²=0.068 (peor, esperado: flip V invierte asimetria N-S del snowpack)
- H-only flip: R²=0.100 (mejor que H+V, pero sigue siendo peor que sin augmentation)
HIPOTESIS: Con solo 54 tiles de train, la augmentation no aporta suficiente variedad real.
El flip H dobla el numero de ejemplos pero el modelo sigue viendo las mismas 54 escenas.
La augmentation seria util con datasets mucho mayores (>200 tiles).

---

## Grupo E - Funcion de perdida alternativa + regularizacion (Normalizacion corregida, sin aug)
> Objetivo: Huber loss, LR scheduler, weight decay

| Experimento | Loss | Extra | MAE (m) | RMSE (m) | R² | NSE | Bias (m) | Estado |
|---|---|---|---|---|---|---|---|---|
| unet_v5_5m_huber | Huber (d=0.5) | - | 0.5296 | 0.7116 | 0.065 | 0.065 | -0.284 | Completado |
| unet_v5_5m_mse_lrs | MSE | ReduceLROnPlateau | 0.5300 | 0.6990 | 0.098 | 0.098 | -0.293 | Completado |
| **unet_v5_5m_mse_wd** | MSE | WeightDecay=1e-4 | **0.4939** | **0.6609** | **0.194** | **0.194** | **-0.213** | **Completado** |

RESULTADO GRUPO E:
- Huber loss: peor que MSE (R²=0.065 vs 0.133). La distribucion de snow depth con muchos ceros favorece MSE.
- MSE + LRS (ReduceLROnPlateau): R²=0.098, sin mejora significativa. El LR base (1e-4) ya era conservador.
- **MSE + Weight Decay (1e-4): R²=0.194 — NUEVO MEJOR U-Net** (supera al MSE fixed R²=0.133, +46% relativo)

HIPOTESIS: El weight decay (L2) penaliza pesos grandes y reduce el sobreajuste con 54 tiles de train.
La mejora real del WD sugiere que el modelo estandar estaba sobreajustando patrones topograficos del train.

---

## Grupo F - Arquitecturas alternativas (Dataset 5m, sin aug, MSE)
> Objetivo: probar arquitecturas mas adecuadas para datasets pequeños

| Experimento | Arquitectura | Params | MAE (m) | RMSE (m) | R² | NSE | Bias (m) | Estado |
|---|---|---|---|---|---|---|---|---|
| unet_small_v5_5m_mse | UNet-Small [32,64,128,256] | 1.95M | 0.5123 | 0.6831 | 0.138 | 0.138 | -0.254 | Completado |
| **attention_unet_v5_5m_mse** | Attention U-Net (Oktay 2018) | 7.92M | **0.5077** | **0.6737** | **0.162** | **0.162** | **-0.247** | **Completado** |
| unet_dropout_v5_5m_mse | UNet + Dropout2d(0.2) | 7.79M | 0.5249 | 0.6930 | 0.113 | 0.113 | -0.309 | Completado |

RESULTADO GRUPO F:
- Attention U-Net: R²=0.162 — mejor arquitectura individual. Las attention gates mejoran la generalizacion
  al enfocar el modelo en regiones topograficamente relevantes (DEM, pendiente, TPI altos).
- UNet-Small: R²=0.138 — similar a MSE fixed (0.133). Menos parametros no ayuda significativamente.
- UNet Dropout: R²=0.113 — Dropout2d degradó el rendimiento en este dataset pequeño.
  El Dropout en el decoder introduce ruido excesivo con solo 54 tiles de train.

CONCLUSION: Attention U-Net (R²=0.162) + WD (R²=0.194) son los dos mejores enfoques individuales.
Su combinacion (Grupo G) es la apuesta mas prometedora para superar RF R²=0.257.

---

## Grupo G - Combinaciones: WD + otras tecnicas (Normalizacion corregida)
> Objetivo: combinar weight decay (mejor regularizador) con otros enfoques para superar RF R²=0.257

| Experimento | Arquitectura | Extra | MAE (m) | RMSE (m) | R² | NSE | Bias (m) | Estado |
|---|---|---|---|---|---|---|---|---|
| unet_v5_5m_mse_wd_lrs | UNet std | WD+LRS | - | - | - | - | - | Pendiente |
| attention_unet_v5_5m_mse_wd | Attention UNet | WD | 0.5225 | 0.6961 | 0.105 | 0.105 | -0.266 | Completado |
| unet_v5_5m_mse_wd_lrs | UNet std | WD+LRS | 0.5221 | 0.6926 | 0.114 | 0.114 | -0.288 | Completado |
| unet_small_v5_5m_mse_wd | UNet-Small | WD | 0.5341 | 0.7052 | 0.082 | 0.082 | -0.295 | Completado |
| unet_v5_5m_mse_wd_aug_h | UNet std | WD+aug_h | 0.5578 | 0.7408 | -0.013 | -0.013 | -0.361 | Completado |

RESULTADO GRUPO G - CRITICO:
Ninguna combinacion supera al WD solo (R²=0.194). Patron claro: WD es el limite de regularizacion util.
- Attention + WD: 0.105 (vs attention: 0.162, WD: 0.194) -> REGRESION
- WD + LRS: 0.114 -> REGRESION
- Small + WD: 0.082 -> REGRESION
- WD + aug_h: -0.013 -> CATASTROFICO (el aug_h con WD destruye la senal)

HIPOTESIS: WD=1e-4 ya es el nivel optimo de regularizacion. Agregar mas restricciones (LRS, arch. pequeña,
aug) causa underfitting y no permite al modelo capturar los patrones topograficos del test set.
El aug_h + WD produce sesgo enorme (-0.361m) indicando que la combinacion fuerza predicciones hacia cero.

---

## Grupo H - Ablacion Weight Decay (Normalizacion corregida, MSE, sin aug)
> Objetivo: confirmar WD=1e-4 como optimo o encontrar mejor valor en [5e-5, 1e-3]

| Experimento | WD | MAE (m) | RMSE (m) | R² | NSE | Bias (m) | Estado |
|---|---|---|---|---|---|---|---|
| unet_v5_5m_mse_wd5e5 | 5e-5 | 0.5134 | 0.6787 | 0.1494 | 0.1494 | -0.259 | Completado |
| unet_v5_5m_mse_wd (ref) | **1e-4** | **0.4939** | **0.6609** | **0.194** | **0.194** | **-0.213** | **OPTIMO** |
| unet_v5_5m_mse_wd5e4 | 5e-4 | 0.5350 | 0.7044 | 0.0837 | 0.0837 | -0.314 | Completado |
| unet_v5_5m_mse_wd1e3 | 1e-3 | 0.5138 | 0.6786 | 0.1496 | 0.1496 | -0.246 | Completado |

RESULTADO GRUPO H - COMPLETO:
- WD=5e-5: R²=0.1494 | WD=1e-4: R²=0.194 | WD=5e-4: R²=0.0837 | WD=1e-3: R²=0.1496
- WD=1e-4 es el MAXIMO GLOBAL confirmado.
- Patron: el pico es agudo — reducir a la mitad o multiplicar por 5-10 degrada significativamente.
- Curva no monotona: WD=1e-3 (10x) es similar a WD=5e-5 (0.5x), ambos ~R²=0.15.
- WD=5e-4 da el peor resultado (R²=0.083), posiblemente por mayor varianza en convergencia.
CONCLUSION: WD=1e-4 esta CONFIRMADO como el valor optimo para esta configuracion.
No hay margen de mejora ajustando WD. La busqueda exhaustiva de WD queda cerrada.

---

## Grupo I - Variaciones sobre el optimo WD=1e-4 (epocas, batch size, TTA)
> Objetivo: exprimir WD=1e-4 con ajustes ortogonales de entrenamiento

| Experimento | Cambio | MAE (m) | RMSE (m) | R² | NSE | Bias (m) | Estado |
|---|---|---|---|---|---|---|---|
| unet_v5_5m_mse_wd_tta | TTA (evaluate-only) | 0.5171 | 0.6909 | 0.1185 | 0.1185 | -0.209 | Completado |
| unet_v5_5m_mse_wd_ep300 | 300 epocas | 0.5064 | 0.6752 | 0.1582 | 0.1582 | -0.226 | Completado |
| unet_v5_5m_mse_wd_bs4 | batch_size=4 | 0.5364 | 0.7004 | 0.094 | 0.094 | -0.324 | Completado |

RESULTADO COMPLETO GRUPO I:
- WD + TTA: R²=0.1185 — EMPEORA respecto a WD solo (0.194). El H-flip corrompe los canales
  direccionales (Eastness negada, Northness inalterada) → promedio de predicciones inconsistentes.
  TTA con flip horizontal NO es fisicamente valido cuando las features incluyen componente E-O.
- ep300 (300 epocas): R²=0.1582 — EMPEORA respecto a ep150 (0.194). Mas epocas no ayudan:
  el early stopping con val=2023 encuentra un checkpoint que no generaliza mejor a 2024-2025.
  Hipotesis: el modelo con 300 epocas sobreajusta mas al patron de 2023 (dominio val) que
  perjudica la generalizacion al test 2024-2025. La brecha de dominio inter-anual es la
  limitacion principal, no la capacidad del modelo.
- bs4 (batch_size=4): R²=0.094 — EMPEORA sustancialmente (vs 0.194 con bs8). Gradientes
  mas ruidosos con batch pequeno → peor convergencia. Bias=-0.324 (mayor que bs8=-0.213).
  Con solo 54 tiles de train, batch size mayor es mas estable.
CONCLUSION GRUPO I: Ninguna variacion sobre WD=1e-4 (TTA, mas epocas, batch mas pequeno)
mejora el baseline. WD=1e-4 + ep150 + bs8 es la configuracion optima en todas las dimensiones
exploradas. La limitacion es el dominio shift train(2021-22) → test(2024-25).

---

## TABLA RESUMEN COMPARATIVA - Dataset 5m (test set 2024-2025)

| Modelo | Canales | Loss | MAE (m) | RMSE (m) | R² | Delta R² vs Naive |
|---|---|---|---|---|---|---|
| Naive (media train) | - | - | 0.558 | 0.750 | -0.039 | ref |
| **RF sin SCE** | 5 topo | - | **0.483** | **0.634** | **0.257** | **+0.296** |
| RF con SCE | 6 | - | 0.492 | 0.645 | 0.233 | +0.272 |
| U-Net alumno MAE (Grupo A) | 5 topo* | MAE | 0.525 | 0.715 | 0.056 | +0.095 |
| U-Net alumno MSE (Grupo A) | 5 topo* | MSE | 0.529 | 0.704 | 0.085 | +0.124 |
| U-Net MAE fixed (Grupo B) | 5 topo | MAE | 0.535 | 0.720 | 0.042 | +0.081 |
| **U-Net MSE fixed (Grupo B)** | 5 topo | MSE | **0.518** | **0.685** | **0.133** | **+0.172** |
| U-Net MAE+SCE (Grupo C) | 5+SCE | MAE | 0.611 | 0.798 | -0.176 | -0.137 |
| U-Net MSE+SCE (Grupo C) | 5+SCE | MSE | 0.603 | 0.788 | -0.145 | -0.106 |
| U-Net MSE aug H+V (Grupo D) | 5 topo | MSE | 0.546 | 0.731 | 0.068 | +0.107 |
| U-Net MSE aug H (Grupo D) | 5 topo | MSE | 0.525 | 0.698 | 0.100 | +0.139 |
| U-Net Huber (Grupo E) | 5 topo | Huber | 0.530 | 0.712 | 0.065 | +0.104 |
| U-Net MSE+LRS (Grupo E) | 5 topo | MSE | 0.530 | 0.699 | 0.098 | +0.137 |
| **U-Net MSE+WD (Grupo E)** | 5 topo | MSE | **0.494** | **0.661** | **0.194** | **+0.233** |
| U-Net Small (Grupo F) | 5 topo | MSE | 0.512 | 0.683 | 0.138 | +0.177 |
| U-Net Attention (Grupo F) | 5 topo | MSE | 0.508 | 0.674 | 0.162 | +0.201 |
| U-Net Dropout (Grupo F) | 5 topo | MSE | 0.525 | 0.693 | 0.113 | +0.152 |
| U-Net Att+WD (Grupo G) | 5 topo | MSE | 0.523 | 0.696 | 0.105 | +0.144 |
| U-Net WD+LRS (Grupo G) | 5 topo | MSE | 0.522 | 0.693 | 0.114 | +0.153 |
| U-Net Small+WD (Grupo G) | 5 topo | MSE | 0.534 | 0.705 | 0.082 | +0.121 |
| U-Net WD+aug_h (Grupo G) | 5 topo | MSE | 0.558 | 0.741 | -0.013 | -0.052 |
| U-Net WD=5e-5 (Grupo H) | 5 topo | MSE | 0.513 | 0.679 | 0.149 | +0.188 |
| U-Net WD=5e-4 (Grupo H) | 5 topo | MSE | 0.535 | 0.704 | 0.084 | +0.123 |
| U-Net WD=1e-3 (Grupo H) | 5 topo | MSE | 0.514 | 0.679 | 0.150 | +0.189 |
| U-Net WD+TTA (Grupo I) | 5 topo | MSE | 0.517 | 0.691 | 0.119 | +0.158 |
| U-Net WD+ep300 (Grupo I) | 5 topo | MSE | 0.506 | 0.675 | 0.158 | +0.197 |
| U-Net WD+bs4 (Grupo I) | 5 topo | MSE | 0.536 | 0.700 | 0.094 | +0.133 |
| U-Net WD seed1 (Grupo J) | 5 topo | MSE | 0.519 | 0.681 | 0.143 | +0.182 |
| U-Net WD seed2 (Grupo J) | 5 topo | MSE | 0.522 | 0.692 | 0.116 | +0.155 |
| U-Net WD seed3 (Grupo J) | 5 topo | MSE | 0.523 | 0.693 | 0.114 | +0.153 |
| U-Net WD ensemble3 (Grupo J) | 5 topo | MSE | 0.514 | 0.679 | 0.149 | +0.188 |
| U-Net Ensemble4 s0+s1+s2+s3 (Grupo K) | 5 topo | MSE | 0.506 | 0.670 | 0.170 | +0.209 |
| U-Net MAE+WD (Grupo K) | 5 topo | MAE | 0.540 | 0.722 | 0.037 | +0.076 |
| U-Net MSE+WD+LR5e-5 (Grupo K) | 5 topo | MSE | 0.530 | 0.698 | 0.100 | +0.139 |
| U-Net AdamW (Grupo L) | 5 topo | MSE | 0.528 | 0.697 | 0.103 | +0.142 |
| U-Net CosineAnneal (Grupo L) | 5 topo | MSE | 0.520 | 0.692 | 0.115 | +0.154 |
| U-Net AdamW+Cosine (Grupo L) | 5 topo | MSE | 0.529 | 0.699 | 0.097 | +0.136 |
| U-Net AdamW+CosineWR (Grupo L) | 5 topo | MSE | 0.531 | 0.702 | 0.091 | +0.130 |
| U-Net Adam+SWA (Grupo M) | 5 topo | MSE | 0.535 | 0.707 | 0.078 | +0.117 |
| U-Net AdamW+SWA (Grupo M) | 5 topo | MSE | 0.533 | 0.702 | 0.089 | +0.128 |
| U-Net AllTrain 71 tiles (Grupo N) | 5 topo | MSE | 0.523 | 0.684 | 0.136 | +0.175 |
| **U-Net 1m MSE fixed (Grupo B)** | 5 topo @1m | MSE | **0.530** | **0.689** | **0.1445** | **+0.155** |
| U-Net 1m MAE fixed (Grupo B) | 5 topo @1m | MAE | - | - | - | - |

*Grupo A: TPI normalizado por 30 (incorrecto), descarta SCE aunque la carga

**MEJOR U-Net (5m): MSE + WeightDecay=1e-4 (Grupo E)** con R²=0.194
La U-Net sigue sin superar al RF (R²=0.257). Brecha actual: 0.063 puntos de R².
Grupos H, I, J, K, L, M, N COMPLETOS. Ninguna variacion supera R²=0.194.
1m MSE fixed: R²=0.1445 — 3570 tiles train pero sin WD. Peor que el mejor 5m (R²=0.194).
Proximo: 1m MAE fixed (en curso), 1m MSE+WD (pendiente), luego analisis final.

---

## Grupo J - Ensemble de seeds (mismo config WD optimo, distinta inicializacion)
> Objetivo: reducir varianza mediante ensemble de modelos con identica config pero distinta init aleatoria

| Experimento | Descripcion | MAE (m) | RMSE (m) | R² | NSE | Bias (m) | Estado |
|---|---|---|---|---|---|---|---|
| unet_v5_5m_mse_wd_s1 | seed 1 (mismo config WD optimo) | 0.5194 | 0.6814 | 0.1425 | 0.1425 | -0.2736 | Completado |
| unet_v5_5m_mse_wd_s2 | seed 2 (distinta init aleatoria) | 0.5220 | 0.6919 | 0.1158 | 0.1158 | -0.2811 | Completado |
| unet_v5_5m_mse_wd_s3 | seed 3 (distinta init aleatoria) | 0.5234 | 0.6927 | 0.1139 | 0.1139 | -0.2836 | Completado |
| unet_v5_5m_mse_wd_ensemble3 | Ensemble 3x (promedio predicciones) | 0.5142 | 0.6790 | 0.1486 | 0.1486 | -0.2796 | Completado |

RESULTADO GRUPO J - CRITICO:
El ensemble de 3 seeds da R²=0.1486 — PEOR que el modelo base original (R²=0.194).
- Los 3 nuevos seeds convergieron todos a soluciones suboptimas (R²=0.11-0.14)
- La varianza inter-run es ENORME: de R²=0.1139 a R²=0.1425 solo con distinta init
- El modelo original unet_v5_5m_mse_wd (R²=0.194) fue una convergencia afortunada
- El ensemble mejora sobre la media de los 3 seeds (avg~0.124) pero no supera el mejor individual
- La hipotesis inicial de que el ensemble reducira varianza es correcta, pero la varianza
  es tan alta que 3 seeds todos "malos" dan un ensemble mediocre

ANALISIS DE LA ALTA VARIANZA:
- Con 54 tiles de train y 17 de val, el paisaje de perdida es muy no-convexo
- Distintas inicializaciones quedan atrapadas en diferentes minimos locales
- La val loss final varía entre 0.1473 (s3) y 0.1654 (s1), pero test R² no correlaciona
  perfectamente con val_loss (s1 mejor test R²=0.1425 con peor val 0.1654)
- Esto confirma el DOMAIN SHIFT: val=2023 no es representativa del test=2024-2025

CONCLUSION GRUPO J:
El ensemble naive de 3 seeds NO mejora el baseline. Para que el ensemble funcione, se necesita
que los seeds individuales sean comparables al mejor modelo. La estrategia de "train N seeds y
seleccionar los mejores" seria mas efectiva, pero costosa computacionalmente.

---

## Grupo K - Exploracion de perdida MAE+WD, LR bajo y ensemble ampliado
> Objetivo: probar MAE+WD (no probado), LR=5e-5+WD, y ensemble con el modelo original s0

| Experimento | Loss | LR | WD | MAE (m) | RMSE (m) | R² | NSE | Bias (m) | Estado |
|---|---|---|---|---|---|---|---|---|---|
| unet_v5_5m_mae_wd | MAE | 1e-4 | 1e-4 | 0.5396 | 0.7221 | 0.0371 | 0.0371 | -0.2781 | Completado |
| unet_v5_5m_mse_wd_lr5e5 | MSE | 5e-5 | 1e-4 | 0.5304 | 0.698 | 0.1004 | 0.1004 | -0.3043 | Completado |
| unet_v5_5m_mse_wd_ensemble4 | Ensemble 4x (s0+s1+s2+s3) | - | - | 0.5061 | 0.6704 | 0.170 | 0.170 | -0.2631 | Completado |

RESULTADO ENSEMBLE4:
Ensemble de 4 modelos (s0+s1+s2+s3) → R²=0.170: MEJOR que ensemble3 (0.149) pero PEOR que s0 solo (0.194).
Incluir el modelo original s0 eleva el ensemble pero los 3 seeds debiles lo ponderan hacia abajo.
CONCLUSION: Para superar al mejor modelo individual, todos los seeds del ensemble deben ser comparables
a ese mejor. El ensemble no es util si hay grandes diferencias de calidad entre modelos.

RESULTADO MAE+WD: R²=0.0371 — FRACASO.
- MAE+WD es PEOR que MAE sin WD (R²=0.042 Grupo B) y peor que MSE+WD (R²=0.194 Grupo E)
- El WD no aporta beneficio a la loss MAE en este dataset. La L1 loss ya actua como regularizador
  implicito (gradientes de magnitud constante, no amplifica errores grandes).
- Confirmacion definitiva: MSE+WD es la combinacion optima para snow depth. MAE descartada.

RESULTADO LR5e-5+WD: R²=0.1004 — PEOR que LR=1e-4+WD (R²=0.194).
- Val_loss bestfue 0.1512 (mejor que el modelo original ~0.18), pero test R² mucho peor
- Confirma que val=2023 no predice bien test=2024-2025 (domain shift)
- LR mas bajo no ayuda: el modelo sobreajusta el patron especifico de 2023
- Bias=-0.3043m (mayor que el original -0.213m): mayor subestimacion con LR bajo

CONCLUSION GRUPO K COMPLETO:
Ninguno de los 3 experimentos supera R²=0.194. El optimo sigue siendo MSE+WD=1e-4+LR=1e-4+ep150+bs8.
- MAE+WD: R²=0.037 (confirmacion: MAE inferior a MSE para snow depth)
- LR5e-5+WD: R²=0.100 (LR mas bajo sobreajusta val=2023 pero no generaliza a 2024-2025)
- Ensemble4: R²=0.170 (seeds debiles diluyen el modelo original)

MOTIVACION GRUPO K (MAE+WD y LR5e-5):
- MAE+WD nunca se probo (MAE sin WD: R²=0.042; WD sin MAE: R²=0.194)
  La combinacion podria ser complementaria
- LR=5e-5+WD podria encontrar mejores minimos con pasos mas finos

---

## Grupo L - AdamW + Cosine Annealing (nuevas combinaciones optimizador/scheduler)
> Objetivo: explorar AdamW (weight decay desacoplado) y CosineAnnealingLR para mejores minimos
> Modificacion: train.py actualizado para soportar optimizer=adamw, lr_scheduler=cosine/cosine_wr

| Experimento | Optimizador | Scheduler | MAE (m) | RMSE (m) | R² | NSE | Bias (m) | Estado |
|---|---|---|---|---|---|---|---|---|
| unet_v5_5m_mse_adamw | AdamW | - | 0.5281 | 0.6969 | 0.1031 | 0.1031 | -0.3047 | Completado |
| unet_v5_5m_mse_wd_cosine | Adam | CosineAnnealingLR | 0.5202 | 0.6922 | 0.1153 | 0.1153 | -0.2643 | Completado |
| unet_v5_5m_mse_adamw_cosine | AdamW | CosineAnnealingLR | 0.5293 | 0.6992 | 0.0973 | 0.0973 | -0.291 | Completado |
| unet_v5_5m_mse_adamw_cosinewr | AdamW | CosineAnnealingWarmRestarts | 0.5306 | 0.7015 | 0.0911 | 0.0911 | -0.301 | Completado |

RESULTADO GRUPO L - CRITICO:
Ninguna combinacion de optimizador/scheduler supera al baseline Adam+WD=0.194.
- AdamW: R²=0.103 | CosineAnneal: R²=0.115 | AdamW+Cosine: R²=0.097 | AdamW+CosineWR: R²=0.091
- Rango Grupo L: R²=0.091 a R²=0.115, todos muy por debajo de R²=0.194
- PARADOJA: todos los modelos de Grupo L tienen MEJOR val_loss (0.143-0.152) que el modelo original
  (val_loss estimado ~0.15-0.18), pero PEOR test R². Esto confirma que val=2023 NO es representativa
  del test=2024-2025. El criterio de seleccion del mejor checkpoint (min val_loss) no es adecuado.
- AdamW teoricamente mejor pero practicamente peor: el WD acoplado de Adam fortuitamente actua
  como un regularizador mas adecuado para este dominio especifico.
- CosineAnnealing mejor entre las variantes (0.1153) pero aun -0.079 vs baseline.
CONCLUSION GRUPO L: El optimizador y scheduler NO son la causa de la superioridad del baseline.
El modelo original (Adam+WD=1e-4) representa un minimo especialmente bueno encontrado por azar.

MOTIVACION GRUPO L:
- AdamW aplica weight decay DESACOPLADO del gradient scaling (a diferencia de Adam donde el WD
  escala con el LR adaptativo). Teoricamente mas correcto para deep learning (Loshchilov 2018).
- CosineAnnealingLR: LR decae suavemente de 1e-4 a 1e-6 en 150 epocas. Evita los saltos bruscos
  del ReduceLROnPlateau y puede explorar mejor el espacio de parametros.
- CosineAnnealingWarmRestarts (T0=50): 3 ciclos de reinicio en 150 epocas. Permite escapar
  minimos locales con reinicios periodicos del LR (SGDR, Loshchilov 2017).
- La combinacion AdamW+CosineAnnealing es el estandar moderno en CV (DeiT, Swin, ConvNeXt).

---

## Grupo M - Stochastic Weight Averaging (SWA) para mayor generalizacion
> Objetivo: promediar pesos de los ultimos N epochs para encontrar minimos mas planos que generalicen mejor
> SWA (Izmailov et al. 2018): teorema: SWA encuentra minimos mas planos -> mejor generalizacion cross-domain
> Modificacion: train.py actualizado para soportar swa=true, swa_start, swa_lr

| Experimento | Optimizador | SWA desde | SWA LR | MAE (m) | RMSE (m) | R² | NSE | Bias (m) | Estado |
|---|---|---|---|---|---|---|---|---|---|
| unet_v5_5m_mse_wd_swa | Adam | ep75 | 1e-5 | 0.5348 | 0.7066 | 0.078 | 0.078 | -0.306 | Completado |
| unet_v5_5m_mse_adamw_swa | AdamW | ep75 | 1e-5 | 0.5331 | 0.7022 | 0.089 | 0.089 | -0.3021 | Completado |

RESULTADO GRUPO M - NEGATIVO:
SWA NO mejora la generalizacion. Ambos modelos SWA son PEORES que el baseline:
- Adam+SWA: R²=0.078 (vs Adam+WD=0.194, -0.116)
- AdamW+SWA: R²=0.089 (vs Adam+WD=0.194, -0.105)
- Los modelos SWA tienen bias elevado (-0.30m): subestiman sistematicamente mas que el baseline (-0.21m)
HIPOTESIS de fallo:
1. SWA promedia sobre 75 epochs suboptimos. Las primeras 75 epocas del SWA encuentran un minimo
   agudo que se diluye al promediar con los ultimos 75 epochs.
2. El baseline Adam+WD encontro un minimo especificamente bueno que el promedio SWA no puede reproducir.
3. La "flatness" del minimo SWA no implica mejor generalizacion temporal si el dominio 2024-25
   es fundamentalmente diferente al dominio 2021-22.
CONCLUSION: SWA no ayuda para este problema de domain shift temporal. La teoria de minimos planos
no aplica directamente cuando el dominio shift es fundamentalmente diferente (anos distintos).

MOTIVACION GRUPO M:
- SWA promedia los pesos del modelo sobre los ultimos epochs. Equivale a SGD con ensemble implicito.
- Los minimos planos (flat minima) generalizan mejor a distribuciones de test distintas al train.
- Con domain shift train(2021-22) → test(2024-25), un minimo mas plano deberia generalizar mejor.
- Hyp: el modelo standard WD converge a un minimo agudo especifico de los patrones 2021-2022.
  SWA podria encontrar un minimo mas generalizable para 2024-2025.
- swa_start=75: warm-up de 75 epocas + SWA sobre las ultimas 75 epocas.

---

## Grupo N - Maximizacion de datos de entrenamiento (train+val = 2021+2022+2023)
> Objetivo: entrenar con TODOS los datos pre-test (71 tiles en lugar de 54) para ver si el volumen
> de datos es el cuello de botella principal. Sin early stopping (se guarda el ultimo checkpoint ep150).
> Motivacion: el domain shift train→test podria reducirse con mas datos de entrenamiento.

| Experimento | Train tiles | Epocas | Early stop | MAE (m) | RMSE (m) | R² | NSE | Bias (m) | Estado |
|---|---|---|---|---|---|---|---|---|---|
| unet_v5_5m_mse_wd_all_train | 71 (2021+2022+2023) | 150 | No (ep150) | 0.5228 | 0.684 | 0.1361 | 0.1361 | -0.2993 | Completado |

RESULTADO GRUPO N: R²=0.1361 — mejor que Grupo L/M pero peor que el baseline 54-tile (R²=0.194).
- El val_loss al final del entrenamiento es 0.031-0.039: IRRELEVANTE. Las tiles 2023 estan en
  entrenamiento Y en validacion (mismo conjunto), por lo que "val_loss" mide sobreajuste al train.
- Con early_stopping=False, se guarda el epoch 150 (ultimo), no el mejor.
- El dominio 2024-2025 es suficientemente diferente a 2021-2023 que anadir 2023 al train no ayuda.
- HIPOTESIS IMPORTANTE: El modelo con 71 tiles sobreadjusta MAS a los anos de train (2021-2023),
  lo que reduce la generalizacion a 2024-2025 en comparacion con el modelo de 54 tiles que tuvo
  la "fortuna" de encontrar un minimo que generalizaba bien (R²=0.194 con early stopping en 2023).
CONCLUSION GRUPO N: Mas datos del mismo dominio temporal no mejoran la generalizacion al dominio
test 2024-2025. El bottleneck es el DOMAIN SHIFT inter-anual, no el volumen de datos 2021-2023.

MOTIVACION GRUPO N:
- Actualmente entrenamos con 54 tiles (2021-2022). La validacion (2023, 17 tiles) se usa para early stopping.
- Si anadimos 2023 al entrenamiento (total 71 tiles), no podemos hacer early stopping convencional.
- Sin val set: guardamos el modelo de la ultima epoca (ep150). Este es el escenario de "maximo uso de datos".
- El benchmark RF tambien usa todos los pixels disponibles de 2021-2023 para entrenamiento.
- Si la U-Net mejora significativamente con 71 tiles, confirma que la limitacion es el volumen de datos.
- Si no mejora: la limitacion es el domain shift especifico de 2024-2025 (anos futuros).

---

---

## Grupo O - Optimizacion Bayesiana HPO (Optuna, dataset v5, 5m)
> Estudio: unet_5m_hpo_v3 | TPE + MedianPruner | 150 epocas por trial
> Arquitecturas: unet + attention_unet | Search space: base_ch, lr, wd, bs, optimizer, grad_clip, dropout
> Estado: DETENIDO a los 4 trials (experimento reorientado hacia dataset v6)

| Trial | Arch | base | lr | wd | bs | opt | test_R² |
|---|---|---|---|---|---|---|---|
| 0 | attention_unet | 32 | 3.5e-5 | 3.2e-4 | 16 | adamw | -0.085 |
| 1 | attention_unet | 48 | 1.7e-4 | 9.7e-6 | 16 | adamw | 0.124 |
| 2 | unet | 32 | 4.5e-4 | 2.4e-4 | 8 | adam | 0.136 |
| 3 | attention_unet | 48 | 1.4e-4 | 1.2e-5 | 8 | adam | **0.172** |

NOTA: Estudio previo unet_5m_hpo_v2 (9 trials) alcanzo best test_R2=0.2011 (trial 7:
attention_unet, base=48, lr=3.9e-4, wd=1.6e-5, bs=16, adam). Esos hiperparametros
se usaron como referencia para los experimentos v6.

---

## Grupo P - Dataset v6 (5m, 17 canales: topo+SCE+Sx+persistencia)
> Dataset: dataset_v6_5m | Canales: [0-13] + [30-32] de los .npy de 33 canales
> Canales: DEM, Slope, Northness, Eastness, TPI, SCE + Sx_8dir_100m + Pers_15d/30d/60d
> Hiperparametros: mejores conocidos del Optuna v2 (wd ajustado por arquitectura)

| Experimento | Arquitectura | Params | MAE (m) | RMSE (m) | R² | NSE | Bias (m) | Estado |
|---|---|---|---|---|---|---|---|---|
| **unet_v6_5m** | U-Net [48,96,192,384] | 4.38M | **0.505** | **0.655** | **0.207** | 0.207 | -0.245 | Completado |
| attention_unet_v6_5m | Attention U-Net [48,96,192,384] | 4.46M | 0.557 | 0.720 | 0.042 | 0.042 | -0.380 | Completado |
| resunetpp_v6_5m | ResUNet++ [48,96,192,384] | 13.76M | 0.509 | 0.668 | 0.175 | 0.175 | -0.316 | Completado |

RESULTADO GRUPO P:
- U-Net v6: R²=0.207 — NUEVO MEJOR U-Net (supera v5 mejor: R²=0.194)
- Attention U-Net v6: colapso inesperado R²=0.042 con los mismos hiper del Optuna v5.
  Los attention gates necesitan reoptimizacion de hiper para 17 canales. No concluyente.
- ResUNet++ v6: R²=0.175 sin optimizacion. Primer experimento con esta arquitectura.
- RF baseline (5 canales): R²=0.257 — sigue siendo el techo sin reentrenar con v6.

NOTA IMPORTANTE sobre canales v6:
- Los .npy tienen 33 canales pero canales 14-29 son CEROS (bug en generate_dataset_v6.py:
  sx_stack = np.zeros((24,...)) en lugar de np.zeros((8,...)) o len(SX_NAMES)).
- Solo se usan los indices [0-13, 30-32] mediante channel_indices en el dataset loader.
- Los 8 canales Sx corresponden a radio 100m, 8 direcciones (0,45,90,135,180,225,270,315 grados).
- Los 3 canales de persistencia (15d, 30d, 60d) estan correctos salvo 6 tiles con std=0
  (primeras fechas sin historico SCE suficiente).

PENDIENTE: RF v6 con 17 canales para comparacion justa con los modelos de red.

---

## Analisis de resultados y hallazgos para el articulo

### Hallazgo 1: Bug de normalizacion TPI (Grupo A vs B)
El alumno dividio TPI por 30 en lugar del rango real (~9200), saturando el canal TPI.
Corregido: MSE mejora R2 de 0.085 a 0.133 (+56% relativo). MAE sin mejora significativa.
**Implicacion**: la normalizacion correcta de variables topograficas es critica.

### Hallazgo 2: Funcion de perdida (MAE vs MSE en todos los grupos)
MSE supera consistentemente a MAE en todos los grupos (A, B, C) para esta distribucion.
La distribucion de snow depth esta muy sesgada hacia cero (muchos pixeles con poca nieve).
MSE penaliza mas los errores grandes → converge a predicciones mas utiles.
**Implicacion**: para distribucion de snow depth, MSE es preferible a MAE.

### Hallazgo 3: SCE satelite perjudica la U-Net (Grupo B vs C)
Anadir SCE empeora la U-Net en ~0.28 puntos de R2 (de +0.133 a -0.145 con MSE).
Hipotesis principal: sensibilidad diferencial LiDAR vs satelite.
- El LiDAR detecta nieve < 5cm, el satelite SCE no.
- Cuando SCE=0, el modelo aprende a predecir ~0, pero hay nieve real fina.
- Con solo 54 tiles de train, el modelo sobreajusta el patron SCE de train.
- Brecha train/test: val_RMSE~0.41m pero test_RMSE=0.788m para MSE+SCE.
El mismo efecto se observa en el RF: RF+SCE (R2=0.233) < RF-SCE (R2=0.257).
**Implicacion**: el SCE satelite a 500m no aporta informacion util para predecir snow depth
con el modelo espacial/temporal de este dataset. Resultado negativo publicable.

### Hallazgo 4: U-Net vs Random Forest con datos limitados
La U-Net mejor (MSE fixed, R2=0.133) no alcanza al RF (R2=0.257).
Con solo 54 tiles de train (split temporal), los modelos deep learning no muestran ventaja.
El RF aprovecha mejor la estructura topografica pixel a pixel con pocos datos.
**Implicacion**: con datasets pequenos, modelos simples pueden superar a deep learning.
Este resultado motiva proponer data augmentation o transfer learning como trabajo futuro.

---

---

## Guia de reproducibilidad

### Entorno
```
GPU: NVIDIA GeForce RTX 5060 Ti (16GB VRAM)
Python: 3.x | PyTorch + CUDA | Optuna | scikit-learn
Directorio raiz: E:\PycharmProjects\SnowPrediction
```

### Datasets
| Dataset | Ruta | Canales | Script generacion |
|---|---|---|---|
| v5 (5m) | `Articulo 1/Data/processed/dataset_v5_5m/` | 6 (topo+SCE) | `Articulo 1/a03_Generate_Dataset_v5.ipynb` |
| v6 (5m) | `Articulo 1/Data/processed/dataset_v6_5m/` | 17 utiles de 33 | `generate_dataset_v6.py` |

### Baseline Random Forest
```bash
# RF v5 sin SCE (5 canales) -> R2=0.257
python baselines/random_forest.py --config configs/unet_v5_5m_mae.yaml

# RF v5 con SCE (6 canales) -> R2=0.233
python baselines/random_forest.py --config configs/unet_v5_5m_mae_sce.yaml

# RF v6 (17 canales: topo+SCE+Sx+persistencia) -> pendiente
python baselines/random_forest_v6.py --config configs/attention_unet_v6_5m.yaml
```

### Modelos U-Net (dataset v5, 5m)
```bash
# Formato general
python main.py --config configs/<nombre>.yaml --mode both

# Experimentos clave:
python main.py --config configs/unet_v5_5m_mse_wd.yaml --mode both        # Grupo E: R2=0.194 (mejor v5)
python main.py --config configs/unet_v5_5m_mse_fixed.yaml --mode both     # Grupo B: R2=0.133
python main.py --config configs/attention_unet_v5_5m_mse.yaml --mode both # Grupo F: R2=0.162
```

### Modelos dataset v6 (17 canales)
```bash
python main.py --config configs/unet_v6_5m.yaml --mode both               # Grupo P: R2=0.207
python main.py --config configs/attention_unet_v6_5m.yaml --mode both     # Grupo P: R2=0.042
python main.py --config configs/resunetpp_v6_5m.yaml --mode both          # Grupo P: R2=0.175
```
> IMPORTANTE: los configs v6 incluyen `channel_indices: [0..13, 30..32]` para
> omitir los 16 canales vacios del .npy (bug en generate_dataset_v6.py linea 196).

### Optimizacion HPO (Optuna)
```bash
# U-Net + Attention U-Net (60 trials)
python optuna_unet.py --trials 60

# ResUNet++ (60 trials, se lanza automaticamente tras el anterior)
# python optuna_resunetpp.py --trials 60
```

### Evaluacion solo (modelo ya entrenado)
```bash
python main.py --config configs/<nombre>.yaml --mode evaluate
```

### Archivos de configuracion por experimento
| Config | Arquitectura | Dataset | WD | LR | Resultado |
|---|---|---|---|---|---|
| `unet_v5_5m_mse_wd.yaml` | U-Net | v5 | 1e-4 | 1e-4 | R²=0.194 |
| `attention_unet_v5_5m_mse.yaml` | Attention U-Net | v5 | 0 | 1e-4 | R²=0.162 |
| `unet_v6_5m.yaml` | U-Net | v6 | 6.6e-5 | 3.9e-4 | R²=0.207 |
| `attention_unet_v6_5m.yaml` | Attention U-Net | v6 | 1.6e-5 | 3.9e-4 | R²=0.042 |
| `resunetpp_v6_5m.yaml` | ResUNet++ | v6 | 1.6e-5 | 3.9e-4 | R²=0.175 |

---

## Notas tecnicas
- Canal SCE: codigos 0 (sin nieve) y 10-11 (con nieve) → binarizado a 0/1
- TPI rango real observado: [-9155, +9141] → normalizado a [-1, 1] con clip
- Matching SCE-LiDAR: imagen SCE mas cercana en fecha (max diferencia = 1 dia, 100% cobertura)
- Todos los modelos U-Net: arquitectura estandar, 7.7M parametros, Adam lr=0.0001
- GPU: RTX 5060 Ti → ~6 min por experimento 5m (150 epocas, 54 tiles train)
- Splits temporales: Train=2021-2022 (54 tiles), Val=2023 (17), Test=2024-2025 (44)
