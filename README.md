# Snow Depth Prediction — Izas Basin, Pyrenees

Deep learning and Random Forest models for snow depth prediction in the Izas experimental catchment (Spanish Pyrenees), using LiDAR and Pleiades imagery at multiple resolutions.

## Repository structure

```
SnowPrediction/
├── main.py                  # Entry point: train / evaluate
├── configs/                 # YAML config files for each experiment
├── scripts/                 # Launchers and utilities
├── baselines/               # Random Forest (Optuna HPO + manual)
├── data/                    # Dataset classes, loaders, CSV generation
├── models/                  # UNet, Attention UNet, ResUNet++ architectures
├── training/                # Training loop and evaluation
├── utils/                   # Metrics, visualization
├── results/                 # JSON metrics, curves, scatter plots (not in git)
├── dataset_v4_fisico/       # LiDAR 1m tiles + CSV (images excluded)
└── Articulo 1/
    ├── Data/processed/      # Processed datasets (not in git — >100 GB)
    └── Models/              # Trained .pth weights (not in git)
```

## Requirements

- Python 3.10+
- CUDA 11.8+ (recommended; CPU works but is very slow)

### Installation

```bash
# Create virtual environment
python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate    # Linux/Mac

# Install dependencies
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install numpy pandas scikit-learn tqdm matplotlib pyyaml optuna joblib
```

> **Note**: Processed tile data (`.npy` files) are not included in the repository due to size (>100 GB).
> Contact the authors to access the `dataset_v5_5m` and `dataset_v6_5m` datasets.

## Datasets

| Dataset   | Resolution | Spatial context | Split (train/val/test) | Channels       |
|-----------|-----------|-----------------|------------------------|----------------|
| v4_fisico | 1 m       | 256 m × 256 m   | 3134 / 1107 / 2575     | 6              |
| v5_5m     | 5 m       | 640 m × 640 m   | ~60 / ~23 / ~44        | 5              |
| v6_5m     | 5 m       | 640 m × 640 m   | ~60 / ~23 / ~44        | 33 → 17 useful |

Temporal split: train=2021–2022, val=2023, test=2024–2025.

Channel groups (v6): DEM, Slope, Northness, Eastness, TPI, SCE + Sx_100m ×8 (wind shelter index) + snow persistence (15d, 30d, 60d).

## Running experiments

### Run all experiments sequentially (RF + CNN)

```bash
.venv\Scripts\python.exe scripts\run_all_experiments.py
```

Order: RF v5 Optuna → RF v6 Optuna → UNet v6 → Attention UNet v6 → ResUNet++ v6

Estimated time: ~9–10 h on a NVIDIA RTX GPU.

### Run a single experiment

```bash
# Train
.venv\Scripts\python.exe main.py --config configs/resunetpp_v6_300ep.yaml --mode train

# Evaluate (requires trained weights)
.venv\Scripts\python.exe main.py --config configs/resunetpp_v6_300ep.yaml --mode evaluate

# Train and evaluate in sequence
.venv\Scripts\python.exe main.py --config configs/resunetpp_v6_300ep.yaml --mode both
```

### 1m resolution experiments (v4 dataset)

```bash
.venv\Scripts\python.exe scripts\run_unet_v4_1m.py
```

### Post-v4 experiments (ResUNet++ 300ep + UNet v6 topo5-only)

```bash
.venv\Scripts\python.exe scripts\run_post_v4_experiments.py
```

This script polls for the v4 experiments to finish before launching the next ones.

## Results summary

| Model                   | Dataset | Channels | R²         | RMSE (m)      |
|-------------------------|---------|----------|------------|---------------|
| RF v5 Optuna (baseline) | v5, 5m  | 5 topo   | 0.2555     | 0.640         |
| RF v6 Optuna (baseline) | v6, 5m  | 17       | 0.2570     | 0.636         |
| UNet v6                 | v6, 5m  | 17       | 0.2495     | 0.643         |
| Attention UNet v6       | v6, 5m  | 17       | ~0.17      | —             |
| ResUNet++ v6            | v6, 5m  | 17       | 0.2495     | 0.643         |
| **ResUNet++ v6 300ep**  | v6, 5m  | 17       | **0.2710** | **0.628**     |
| UNet v4 1m (topo5)      | v4, 1m  | 5 topo   | 0.0811     | —             |
| UNet v6 topo5-only      | v6, 5m  | 5 topo   | 0.0188     | —             |

Key findings:
- **ResUNet++ 300ep (R²=0.271) beats the RF baseline** with longer training, cosine LR and horizontal flip augmentation.
- **Spatial context matters**: 5m tiles (640m×640m) outperform 1m tiles (256m×256m) significantly.
- **Sx_100m + persistence channels are critical for CNNs**: removing them drops R² from 0.25 to 0.02; RF is unaffected.

## Compile all results into a table

```bash
.venv\Scripts\python.exe scripts\compile_results.py
# Output: results/resultados_todos.csv
```

## Path handling

All paths in configs and scripts are **relative to the repository root**.
`main.py` resolves them automatically via `Path(__file__).resolve().parent`, so no edits are needed after cloning to a different location.
