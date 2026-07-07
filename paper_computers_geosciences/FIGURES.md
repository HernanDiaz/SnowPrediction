# Plan de figuras — Paper Computers & Geosciences

Lista de figuras referenciadas en el LaTeX, con el archivo esperado, la
sección donde aparecen y los datos/scripts necesarios para generarlas.

| # | Archivo | Sección | Contenido | Fuente de datos |
|---|---------|---------|-----------|-----------------|
| 1 | `fig01_study_area.pdf` | Área de estudio | (a) Localización Pirineos/Península, (b) límite cuenca Izas sobre hillshade DEM 1m + AWS, (c) ejemplo mapa HS LiDAR con patrones de viento | DEM 1m, shapefile cuenca, un mapa HS de muestra |
| 2 | `fig02_channels.pdf` | Datos | Mosaico del stack de 22 canales para una fecha: topo 1m/5m, 8×Sx, SCE, persistencia 15/30/60d + target HS | un tile de `dataset_v4_ms_sx200` |
| 3 | `fig03_split_distribution.pdf` | Datos | Split temporal train/val/test + histogramas/violines de HS por año (resaltar 2025 atípico +44%) | CSV del dataset, máscaras HS |
| 4 | `fig04_model_comparison.pdf` | Resultados §5.1 | Barras R² y SPAEF (media±std 3 seeds) para RF, UNet, ResUNet++ | `results/rf_v6_s*`, `unet_*_sp00_s*`, `lambda_sweep_hpo/*sp04*` |
| 5 | `fig05_prediction_maps.pdf` | Resultados §5.1 | Mapas predichos de un tile test 2025: GT \| RF \| UNet \| ResUNet++ | pesos .pth + RF reentrenado |
| 6 | `fig06_lambda_sweep.pdf` | Resultados §5.2 | Curvas R² y SPAEF vs λ (0→1) con banda ±1 std | Tabla `tab:lambda` (8 λ × 3 seeds) |
| 7 | `fig07_ablation.pdf` | Resultados §5.3 | Barras de cambio en R²/SPAEF al quitar cada grupo de canales | Tabla `tab:ablation` (6 grupos × 3 seeds) |
| 8 | `fig08_meteo_comparison.pdf` | Resultados §5.4 | Comparación 22ch vs 26ch en todas las métricas (media±std) | Tabla `tab:meteo` |

## Figuras opcionales (suplementario)
- Curvas de entrenamiento (train/val loss) ilustrando que val no es buen proxy en 2025.
- Scatter predicho vs observado por modelo (densidad de puntos).
- Matriz de correlación entre canales / importancia de features del RF.
- Mapa de error espacial (predicho − GT) por modelo.

## Notas
- Figuras en **PDF vectorial** preferentemente (matplotlib `savefig(..., format='pdf')`).
- Paleta consistente entre figuras (un color por modelo).
- `\graphicspath{{figures/}}` ya está configurado en `main.tex`.
- Todas las figuras del LaTeX llevan marca `[PLACEHOLDER FIGURE]` en el caption
  hasta que se generen.
