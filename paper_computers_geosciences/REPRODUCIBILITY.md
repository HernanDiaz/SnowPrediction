# Reproducibility — *Spatially-aware deep learning for high-resolution snow depth mapping*

This document maps every figure and table of the manuscript to the exact code,
configuration and data needed to reproduce it. All commands are run from the
repository root with the project virtual environment
(`.venv/Scripts/python.exe` on Windows, `.venv/bin/python` on Linux/Mac).

> The legacy `README.md` at the repository root describes earlier (5 m / 17-channel)
> experiments and is **superseded** by this document for everything related to the paper.

## 1. Environment

```bash
python -m venv .venv
.venv/Scripts/activate            # Windows  (source .venv/bin/activate on Linux/Mac)
pip install -r requirements.txt
```
Python 3.10+, PyTorch with CUDA recommended (CPU works but is slow). Figures
additionally require `matplotlib`, `rasterio`, `geopandas` (Fig. 1 only).

## 2. Data and trained weights

| Artefact | Location | In git? |
|---|---|---|
| Reference dataset (22 ch, 1 m) | `dataset_v4_ms_sx200/` (`images/`, `masks/`, `dataset_v4_ms_sx200.csv`) | tiles excluded (size) |
| Meteo dataset (26 ch) | `dataset_v4_ms_sx200_meteo/` | tiles excluded |
| Trained CNN weights | `Articulo 1/Models/*.pth` | excluded |
| Raw LiDAR rasters (Fig. 1) | `Articulo 1/Data/izas/LiDAR/` | excluded |

Temporal split (column `exp_temporal_split` in the CSV): train = 2021–2023
(4241 tiles), val = 2024 (431), test = 2025 (220).

## 3. Reference model and key configs

| Config | Purpose |
|---|---|
| `configs/resunetpp_v4_ms_sx200_hpo_sp04.yaml` | **Reference model**: ResUNet++, spatial loss λ=0.4, seed 42 |
| `configs/unet_v4_ms_sx200_hpo_sp00.yaml` | U-Net baseline (Optuna trial 11), seed 42 |
| `configs/resunetpp_hpo_sp*_s{123,7}.yaml` | λ sweep, seeds 123/7 |
| `configs/resunetpp_v4_ms_sx200_hpo_sp*.yaml` | λ sweep, seed 42 |
| `configs/resunetpp_ablation_*_s*.yaml` | leave-one-group-out ablation |
| `configs/resunetpp_meteo_sp*_s*.yaml` | 26-channel meteo experiment |

Train / evaluate a single model:
```bash
.venv/Scripts/python.exe main.py --config configs/resunetpp_v4_ms_sx200_hpo_sp04.yaml --mode both
```
Conventions: U-Net is read at its **last** epoch, ResUNet++ at its
**best-validation** epoch; 50 epochs, no early stopping, no augmentation; the
seed is fixed just before model construction in `main.py`.

Random Forest (v6, 22 ch, retrained on train+val per seed):
```bash
.venv/Scripts/python.exe baselines/compute_spaef_rf_v6.py
```

## 4. Tables

| Table | Source metrics |
|---|---|
| Tab. 1 (comparison) & Tab. 2 (per-seed) | `results/rf_v6_s{42,123,7}/`, U-Net `results/unet_v4_ms_sx200_hpo_50ep/` (s42) + `results/unet_v4_ms_sx200_hpo_sp00_s{123,7}/` (`*_last_metrics.json`), ResUNet++ `results/resunetpp_v4_ms_sx200_hpo_sp04/` (s42) + `results/lambda_sweep_hpo/resunetpp_hpo_sp04_s{123,7}/` |
| Tab. 3 (λ sweep) | `results/resunetpp_v4_ms_sx200_hpo[_sp*]/` (s42) + `results/lambda_sweep_hpo/resunetpp_hpo_sp*_s{123,7}/` |
| Tab. 4 (ablation) | `results/resunetpp_v4_ms_sx200_hpo/` (full, s42) + `results/ablation/resunetpp_ablation_*_s*/` |
| Tab. 5 (meteo) | `results/lambda_sweep_meteo/resunetpp_meteo_sp*_s{42,123,7}/` |

Note: λ=0 (`resunetpp_v4_ms_sx200_hpo`, loss `mse`) is the full-stack baseline of
the ablation, so the ablation "Full" row equals the λ=0 row of the sweep.

## 5. Figures

Scripts live in `paper_computers_geosciences/figscripts/`; output PDFs/PNGs go to
`paper_computers_geosciences/figures/`. Run e.g.:
```bash
.venv/Scripts/python.exe paper_computers_geosciences/figscripts/fig06_lambda_sweep.py
```

| Figure | Script | Inputs |
|---|---|---|
| Fig. 1 study area | `fig01_study_area.py` | LiDAR DEM + snow-depth rasters |
| Fig. 2 channel stack | `fig02_channels.py` | one tile of `dataset_v4_ms_sx200` |
| Fig. 3 split distribution | `fig03_split_distribution.py` | raw `SD_*` snow-depth maps |
| Fig. 4 model comparison | `fig04_model_comparison.py` | table-1 metrics JSON |
| Fig. 5 prediction maps | `fig05_prediction_maps.py` | `.pth` weights + RF (auto-retrained and cached to `results/rf_v6_s42/rf_v6_s42.joblib`) |
| Fig. 6 λ sweep | `fig06_lambda_sweep.py` | table-3 metrics JSON |
| Fig. 7 ablation | `fig07_ablation.py` | table-4 metrics JSON |
| Fig. 8 meteo | `fig08_meteo.py` | table-3 + table-5 metrics JSON |

## 6. Build the manuscript

```bash
cd paper_computers_geosciences
pdflatex -interaction=nonstopmode main.tex
bibtex main
pdflatex -interaction=nonstopmode main.tex
pdflatex -interaction=nonstopmode main.tex
```
Produces `main.pdf` (~26 pp). MiKTeX/TeX Live with the `elsarticle` class.
