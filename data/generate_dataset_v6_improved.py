"""
Genera dataset_v6_improved: tiles a 5m con 17 canales.
=======================================================

Mejoras respecto al dataset v6 original:
  1. Split: train=2021-2023, val=2024, test=2025  (vs 2020-22/2023/2024+)
  2. Stride: 128 train / 256 val-test  (vs 128 para todos)
  3. Sx_200m en vez de Sx_100m  (recomendado por experto)
  4. Solo LiDAR (no Pleiades), consistente con v4_17ch
  5. Codigo limpio, mismo patron que generate_dataset_v4_17ch.py

Canales (17):
  [0]  DEM
  [1]  Slope          (pixel_size=5m)
  [2]  Northness
  [3]  Eastness
  [4]  TPI            (kernel 7x7 → ~35m radio a 5m)
  [5]  SCE
  [6]  Sx_200m_0
  [7]  Sx_200m_45
  [8]  Sx_200m_90
  [9]  Sx_200m_135
  [10] Sx_200m_180
  [11] Sx_200m_225
  [12] Sx_200m_270
  [13] Sx_200m_315
  [14] Persistencia_15d
  [15] Persistencia_30d
  [16] Persistencia_60d

Tiles: 256x256 px a 5m = 1280m x 1280m de cobertura

Uso:
    .venv\\Scripts\\python.exe data/generate_dataset_v6_improved.py
"""

import re
import numpy as np
import pandas as pd
import rasterio
from rasterio.enums import Resampling
from rasterio.warp import reproject
import cv2
from datetime import datetime, timedelta
from pathlib import Path
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Rutas
# ---------------------------------------------------------------------------
_REPO = Path(__file__).resolve().parent.parent

BASE_DATA    = _REPO / "Articulo 1/Data/izas/LiDAR"
TOPO_DIR     = BASE_DATA / "Topografia"
SNOW_DIR     = BASE_DATA / "SnowDepth"
SCE_DIR      = BASE_DATA / "images_SCE"
DEM_PATH     = TOPO_DIR / "DEMbigIzas_5m.tif"
OUTPUT_DIR   = _REPO / "dataset_v6_improved"

# ---------------------------------------------------------------------------
# Parametros
# ---------------------------------------------------------------------------
TILE_SIZE       = 256
STRIDE_TRAIN    = 128    # 50% overlap → mas tiles de entrenamiento
STRIDE_EVALTEST = 256    # sin overlap para val/test
MIN_VALID_FRAC  = 0.30
NODATA_VAL      = -9999.0
PIXEL_SIZE      = 5.0    # resolucion DEM en metros

TRAIN_YEARS = {2021, 2022, 2023}
VAL_YEARS   = {2024}
TEST_YEARS  = {2025}

SX_DIRS         = ['0', '45', '90', '135', '180', '225', '270', '315']
SX_RADIUS       = '200'          # <-- Sx a 200m (mejora sobre v6 original)
PERSIST_WINDOWS = [15, 30, 60]

DATE_FIXES_LIDAR = {'20210608': '20210607'}

# ---------------------------------------------------------------------------
# Funciones topograficas
# ---------------------------------------------------------------------------

def calculate_topography(dem, pixel_size=5.0):
    x, y = np.gradient(dem, pixel_size)
    slope = np.arctan(np.sqrt(x**2 + y**2)) * (180 / np.pi)
    aspect = np.arctan2(-x, y) * (180 / np.pi)
    aspect = np.where(aspect < 0, aspect + 360, aspect)
    aspect_rad = np.radians(aspect)
    northness = np.cos(aspect_rad)
    eastness  = np.sin(aspect_rad)
    return slope, northness, eastness


def calculate_tpi(dem, kernel_size=7):
    """kernel_size=7 → radio ~35m a 5m/px, similar a 31x31 en 1m."""
    kernel = np.ones((kernel_size, kernel_size)) / (kernel_size**2 - 1)
    kernel[kernel_size // 2, kernel_size // 2] = 0
    mean_dem = cv2.filter2D(dem.astype(np.float32), -1, kernel,
                            borderType=cv2.BORDER_REPLICATE)
    return dem - mean_dem


# ---------------------------------------------------------------------------
# Carga de capas estaticas en memoria (una sola vez)
# ---------------------------------------------------------------------------

def load_sx_arrays(dem_height, dem_width):
    """
    Carga los 8 rasters Sx_200m (en 1m) remuestreados al grid 5m del DEM.
    Usa rasterio out_shape para resamplear en una sola lectura (eficiente).
    """
    print(f"Cargando Sx_{SX_RADIUS}m (8 dirs) → resampleado a {dem_height}x{dem_width}...")
    sx_arrays = []
    for d in SX_DIRS:
        path = TOPO_DIR / f"sx_{d}_{SX_RADIUS}.tif"
        with rasterio.open(path) as src:
            arr = src.read(
                1,
                out_shape=(dem_height, dem_width),
                resampling=Resampling.average,
            ).astype(np.float32)
            nodata = src.nodata
            if nodata is not None:
                arr[arr == nodata] = 0.0
            arr = np.where(np.isfinite(arr), arr, 0.0)
            sx_arrays.append(arr)
    print(f"  Sx cargados: shape={sx_arrays[0].shape}")
    return sx_arrays


def load_all_sce(dem_height, dem_width, dem_profile):
    """
    Pre-carga todos los rasters SCE (10m) reproyectados al grid DEM 5m.
    Devuelve dict {date_str: array (H,W) float32}.
    """
    print("Pre-cargando rasters SCE a 5m...")
    sce_dict = {}
    for fpath in sorted(SCE_DIR.glob("izas_*.tif")):
        m = re.search(r'izas_(\d{8})\.tif', fpath.name)
        if not m:
            continue
        date_str = m.group(1)
        dst = np.zeros((dem_height, dem_width), dtype=np.float32)
        with rasterio.open(fpath) as src:
            reproject(
                source=rasterio.band(src, 1),
                destination=dst,
                src_transform=src.transform,
                src_crs=src.crs,
                dst_transform=dem_profile['transform'],
                dst_crs=dem_profile['crs'],
                resampling=Resampling.average,
            )
        sce_dict[date_str] = dst
    print(f"  SCE cargados: {len(sce_dict)} fechas")
    return sce_dict


# ---------------------------------------------------------------------------
# Calculo de persistencia
# ---------------------------------------------------------------------------

def compute_persistence(target_date_str, sce_dict, dem_height, dem_width):
    """
    Calcula fraccion de dias con nieve (valor>=10.5 → SCE==11) para cada
    ventana temporal. Devuelve (3, H, W).
    """
    target_dt = datetime.strptime(target_date_str, '%Y%m%d')
    result    = np.zeros((len(PERSIST_WINDOWS), dem_height, dem_width),
                         dtype=np.float32)

    for wi, window in enumerate(PERSIST_WINDOWS):
        cutoff = target_dt - timedelta(days=window)
        imgs = []
        for date_str, arr in sce_dict.items():
            dt = datetime.strptime(date_str, '%Y%m%d')
            if cutoff <= dt < target_dt:
                imgs.append(arr)
        if imgs:
            stack      = np.stack(imgs, axis=0)
            snow_days  = np.sum(stack >= 10.5, axis=0).astype(np.float32)
            valid_days = np.sum(stack >=  9.5, axis=0).astype(np.float32)
            with np.errstate(divide='ignore', invalid='ignore'):
                result[wi] = np.where(valid_days > 0,
                                      snow_days / valid_days, 0.0)
    return result


# ---------------------------------------------------------------------------
# Script principal
# ---------------------------------------------------------------------------

def main():
    out_images = OUTPUT_DIR / "images"
    out_masks  = OUTPUT_DIR / "masks"
    out_images.mkdir(parents=True, exist_ok=True)
    out_masks.mkdir(parents=True, exist_ok=True)

    # Abrir DEM 5m
    with rasterio.open(DEM_PATH) as src_dem:
        dem_h       = src_dem.height
        dem_w       = src_dem.width
        dem_profile = src_dem.profile
        dem_crs     = src_dem.crs

    print(f"DEM 5m: {dem_h}x{dem_w} px  |  cobertura={dem_h*5}x{dem_w*5} m")

    # Cargar capas estaticas una sola vez
    sx_arrays = load_sx_arrays(dem_h, dem_w)
    sce_dict  = load_all_sce(dem_h, dem_w, dem_profile)

    lidar_files = sorted(SNOW_DIR.glob("SD_*_1m.tif"))
    print(f"\nArchivos LiDAR encontrados: {len(lidar_files)}")

    metadata = []

    with rasterio.open(DEM_PATH) as src_dem:
        for fp in tqdm(lidar_files, desc="Generando tiles v6-improved"):
            fname    = fp.name
            date_str = fname.split('_')[1]
            year     = int(date_str[:4])

            if year not in (TRAIN_YEARS | VAL_YEARS | TEST_YEARS):
                continue

            sce_date_str = DATE_FIXES_LIDAR.get(date_str, date_str)
            stride = STRIDE_TRAIN if year in TRAIN_YEARS else STRIDE_EVALTEST
            split  = ('train' if year in TRAIN_YEARS
                      else 'val' if year in VAL_YEARS else 'test')

            # -----------------------------------------------------------------
            # A. DEM crop (5m) alineado con el extent del LiDAR
            # -----------------------------------------------------------------
            with rasterio.open(fp) as src_target:
                try:
                    win            = src_dem.window(*src_target.bounds)
                    win            = win.round_offsets().round_lengths()
                    transform_crop = src_dem.window_transform(win)
                    dem_crop       = src_dem.read(1, window=win,
                                                  boundless=True,
                                                  fill_value=NODATA_VAL)
                    h, w = dem_crop.shape
                except Exception as e:
                    print(f"  [SKIP] {fname}: {e}")
                    continue

                # Snow depth reproyectado al grid 5m
                sd_crop = np.zeros((h, w), dtype=np.float32)
                reproject(
                    source=rasterio.band(src_target, 1),
                    destination=sd_crop,
                    src_transform=src_target.transform,
                    src_crs=src_target.crs,
                    dst_transform=transform_crop,
                    dst_crs=dem_crs,
                    resampling=Resampling.bilinear,
                )

            # -----------------------------------------------------------------
            # B. SCE crop (del dict pre-cargado)
            # -----------------------------------------------------------------
            row_off = int(win.row_off)
            col_off = int(win.col_off)

            sce_crop = np.zeros((h, w), dtype=np.float32)
            if sce_date_str in sce_dict:
                arr = sce_dict[sce_date_str]
                r0 = max(row_off, 0);  r1 = min(row_off + h, arr.shape[0])
                c0 = max(col_off, 0);  c1 = min(col_off + w, arr.shape[1])
                if r1 > r0 and c1 > c0:
                    sce_crop[r0-row_off:r1-row_off,
                             c0-col_off:c1-col_off] = arr[r0:r1, c0:c1]

            # -----------------------------------------------------------------
            # C. Canales topograficos (calculados al vuelo sobre el crop 5m)
            # -----------------------------------------------------------------
            slope, northness, eastness = calculate_topography(
                dem_crop, pixel_size=PIXEL_SIZE)
            tpi = calculate_tpi(dem_crop)

            # -----------------------------------------------------------------
            # D. Sx crop (del array pre-cargado a 5m)
            # -----------------------------------------------------------------
            sx_crops = []
            for sx_arr in sx_arrays:
                r0 = max(row_off, 0);  r1 = min(row_off + h, sx_arr.shape[0])
                c0 = max(col_off, 0);  c1 = min(col_off + w, sx_arr.shape[1])
                patch = np.zeros((h, w), dtype=np.float32)
                if r1 > r0 and c1 > c0:
                    patch[r0-row_off:r1-row_off,
                          c0-col_off:c1-col_off] = sx_arr[r0:r1, c0:c1]
                sx_crops.append(patch)

            # -----------------------------------------------------------------
            # E. Persistencia (del dict SCE pre-cargado)
            # -----------------------------------------------------------------
            pers_full = compute_persistence(sce_date_str, sce_dict, dem_h, dem_w)
            pers_crop = np.zeros((3, h, w), dtype=np.float32)
            r0 = max(row_off, 0);  r1 = min(row_off + h, dem_h)
            c0 = max(col_off, 0);  c1 = min(col_off + w, dem_w)
            if r1 > r0 and c1 > c0:
                pers_crop[:, r0-row_off:r1-row_off,
                              c0-col_off:c1-col_off] = pers_full[:, r0:r1, c0:c1]

            # -----------------------------------------------------------------
            # F. Stack 17 canales
            # -----------------------------------------------------------------
            stack = np.stack(
                [dem_crop, slope, northness, eastness, tpi, sce_crop]
                + sx_crops
                + [pers_crop[0], pers_crop[1], pers_crop[2]],
                axis=0,
            ).astype(np.float32)

            # -----------------------------------------------------------------
            # G. Trocear en tiles 256x256 y guardar
            # -----------------------------------------------------------------
            for ty in range(0, h - TILE_SIZE + 1, stride):
                for tx in range(0, w - TILE_SIZE + 1, stride):
                    tile_img  = stack[:, ty:ty+TILE_SIZE, tx:tx+TILE_SIZE]
                    tile_mask = sd_crop[ty:ty+TILE_SIZE, tx:tx+TILE_SIZE]

                    if np.mean(tile_img[0] == NODATA_VAL) > 0.10:
                        continue
                    if np.min(tile_mask) < -100:
                        continue
                    valid_frac = np.mean(tile_mask > -100)
                    if valid_frac < MIN_VALID_FRAC:
                        continue

                    tile_name = f"{date_str}_lidar_tile_{ty}_{tx}.npy"
                    np.save(out_images / tile_name, tile_img)
                    np.save(out_masks  / tile_name, tile_mask)
                    metadata.append({
                        'tile_id':            tile_name,
                        'date':               date_str,
                        'year':               year,
                        'source':             'lidar',
                        'valid_frac':         round(float(valid_frac), 4),
                        'exp_temporal_split': split,
                    })

    # -------------------------------------------------------------------------
    # Guardar CSV
    # -------------------------------------------------------------------------
    df = pd.DataFrame(metadata)
    df.to_csv(OUTPUT_DIR / "dataset_v6_improved.csv", index=False)

    print(f"\n{'='*60}")
    print(f"GENERACION COMPLETADA: {len(df)} tiles  |  17 canales")
    print(f"  Resolucion  : 5m  |  Tile: {TILE_SIZE}px = {TILE_SIZE*5}m")
    print(f"  Sx radio    : {SX_RADIUS}m")
    print(f"  Split       : train=2021-23 / val=2024 / test=2025")
    print(f"  Stride train: {STRIDE_TRAIN}px (50% overlap)")
    print(f"Guardado en: {OUTPUT_DIR}")
    counts = df['exp_temporal_split'].value_counts()
    total  = len(df)
    for s in ['train', 'val', 'test']:
        n = counts.get(s, 0)
        print(f"  {s:6s}: {n:5d}  ({100*n/total:.1f}%)")


if __name__ == '__main__':
    main()
