# Snow Depth Prediction — Articulo 1

Prediccion de profundidad de nieve en la cuenca de Izas (Pirineos) mediante CNN y Random Forest, usando datos LiDAR y Pleiades a distintas resoluciones.

## Estructura del repositorio

```
SnowPrediction/
├── main.py                  # Punto de entrada: train / evaluate
├── configs/                 # Configuraciones YAML de cada experimento
├── scripts/                 # Lanzadores y utilidades
├── baselines/               # Random Forest (Optuna + manual)
├── data/                    # Dataset, loaders, generacion de CSVs
├── models/                  # Arquitecturas UNet, AttUNet, ResUNet++
├── training/                # Bucle de entrenamiento y evaluacion
├── utils/                   # Metricas, visualizacion
├── results/                 # Metricas JSON, curvas, scatters (no en git)
├── dataset_v4_fisico/       # Tiles LiDAR 1m + CSV (imagenes excluidas)
└── Articulo 1/
    ├── Data/processed/      # Datasets procesados (no en git — >100 GB)
    └── Models/              # Pesos .pth entrenados (no en git)
```

## Requisitos

- Python 3.10+
- CUDA 11.8+ (recomendado; funciona en CPU pero muy lento)

### Instalacion

```bash
# Crear entorno virtual
python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate    # Linux/Mac

# Instalar dependencias
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install numpy pandas scikit-learn tqdm matplotlib pyyaml optuna joblib
```

> **Nota**: Los datos procesados (`.npy`) no estan en el repositorio por su tamano.
> Contacta con los autores para acceder a los datasets `dataset_v5_5m` y `dataset_v6_5m`.

## Datasets

| Dataset       | Resolucion | Contexto/tile | N tiles (train/val/test) | Canales |
|---------------|-----------|---------------|--------------------------|---------|
| v4_fisico     | 1 m       | 256 m × 256 m | 3134 / 1107 / 2575       | 6       |
| v5_5m         | 5 m       | 640 m × 640 m | ~60 / ~23 / ~44          | 5       |
| v6_5m         | 5 m       | 640 m × 640 m | ~60 / ~23 / ~44          | 33 → 17 utiles |

Split temporal: train=2021-2022, val=2023, test=2024-2025.

## Experimentos principales

### Lanzar todos los experimentos (RF + CNN secuencial)

```bash
.venv\Scripts\python.exe scripts\run_all_experiments.py
```

Orden: RF v5 Optuna → RF v6 Optuna → UNet v6 → Attention UNet v6 → ResUNet++ v6

Tiempo estimado: ~9-10 h en GPU NVIDIA RTX.

### Lanzar un experimento individual

```bash
# Entrenar
.venv\Scripts\python.exe main.py --config configs/resunetpp_v6_300ep.yaml --mode train

# Evaluar (requiere pesos ya entrenados)
.venv\Scripts\python.exe main.py --config configs/resunetpp_v6_300ep.yaml --mode evaluate

# Entrenar y evaluar en secuencia
.venv\Scripts\python.exe main.py --config configs/resunetpp_v6_300ep.yaml --mode both
```

### Experimentos v4 (1 m de resolucion)

```bash
.venv\Scripts\python.exe scripts\run_unet_v4_1m.py
```

### Experimentos post-v4 (ResUNet++ 300ep + UNet v6 topo5)

```bash
.venv\Scripts\python.exe scripts\run_post_v4_experiments.py
```

Este script espera a que los experimentos v4 terminen antes de lanzar los siguientes.

## Resultados resumidos

| Modelo                  | Dataset | Canales | R²     | RMSE (m) |
|-------------------------|---------|---------|--------|----------|
| RF v5 Optuna (baseline) | v5, 5m  | 5 topo  | 0.2555 | 0.640    |
| RF v6 Optuna (baseline) | v6, 5m  | 17      | 0.2570 | 0.636    |
| UNet v6 quick           | v6, 5m  | 17      | 0.2495 | 0.643    |
| Attention UNet v6       | v6, 5m  | 17      | ~0.17  | —        |
| ResUNet++ v6 quick      | v6, 5m  | 17      | 0.2495 | 0.643    |
| **ResUNet++ v6 300ep**  | v6, 5m  | 17      | **0.2710** | **0.628** |
| UNet v4 1m (topo5)      | v4, 1m  | 5 topo  | 0.0811 | —        |
| UNet v6 topo5-only      | v6, 5m  | 5 topo  | 0.0188 | —        |

> **Hallazgo clave**: ResUNet++ 300ep (R²=0.271) supera el baseline RF.
> El contexto espacial 640m×640m (5m/pixel) es critico: los modelos a 1m (256m×256m) son mucho peores.
> Los canales Sx_100m + persistencia son esenciales para CNN (sin ellos: R² cae de 0.25 a 0.02).

## Compilar resultados

```bash
.venv\Scripts\python.exe scripts\compile_results.py
# Salida: results/resultados_todos.csv
```

## Rutas

Todos los paths en configs y scripts son **relativos a la raiz del repositorio**.
`main.py` los resuelve automaticamente usando `Path(__file__).resolve().parent`.
No es necesario editar nada al clonar en una ruta distinta.

## Autores

- Herdi — Universidad de Zaragoza
