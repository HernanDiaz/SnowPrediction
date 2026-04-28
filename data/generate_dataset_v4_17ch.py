"""
Genera dataset_v4_17ch: tiles a 1m de resolucion con 17 canales.
================================================================

Replica la logica del notebook a02_Generate_Dataset_v4.ipynb y anade
los 11 canales presentes en v6 (Sx_100m x8 + persistencia x3).

Canales (17):
  [0]  DEM
  [1]  Slope
  [2]  Northness  (cos aspect)
  [3]  Eastness   (sin aspect)
  [4]  TPI        (kernel 31x31)
  [5]  SCE        (valores brutos: 0/10/11)
  [6]  Sx_100m_0
  [7]  Sx_100m_45
  [8]  Sx_100m_90
  [9]  Sx_100m_135
  [10] Sx_100m_180
  [11] Sx_100m_225
  [12] Sx_100m_270
  [13] Sx_100m_315
  [14] Persistencia_15d  (fraccion dias con nieve en ventana de 15d previos)
  [15] Persistencia_30d
  [16] Persistencia_60d

Split temporal:
  Train : 2021, 2022, 2023  (stride=128, 50% overlap)
  Val   : 2024               (stride=256, sin overlap)
  Test  : 2025               (stride=256, sin overlap)

Filtro: se descartan tiles con < MIN_VALID_FRAC pixeles validos en mascara.

Uso:
    .venv\\Scripts\\python.exe data/generate_dataset_v4_17ch.py
"""

import os
import re
import glob
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
DEM_PATH     = TOPO_DIR / "DEMbigIzas_1m.tif"
OUTPUT_DIR   = _REPO / "dataset_v4_17ch"

# ---------------------------------------------------------------------------
# Parametros
# ---------------------------------------------------------------------------
TILE_SIZE       = 256
STRIDE_TRAIN    = 128   # 50% overlap para train (mas tiles)
STRIDE_EVALTEST = 256   # sin overlap para val/test
MIN_VALID_FRAC  = 0.30
NODATA_VAL      = -9999.0

TRAIN_YEARS = {2021, 2022, 2023}
VAL_YEARS   = {2024}
TEST_YEARS  = {2025}

SX_DIRS = ['0', '45', '90', '135', '180', '225', '270', '315']
PERSIST_WINDOWS = [15, 30, 60]

# Correcciones de fecha lidar → SCE (heredadas del notebook original)
DATE_FIXES_LIDAR = {'20210608': '20210607'}

# ---------------------------------------------------------------------------
# Funciones topograficas (igual que en el notebook original)
# ---------------------------------------------------------------------------

def calculate_topography(dem):
    x, y = np.gradient(dem)
    slope = np.arctan(np.sqrt(x**2 + y**2)) * (180 / np.pi)
    aspect = np.arctan2(-x, y) * (180 / np.pi)
    aspect = np.where(aspect < 0, aspect + 360, aspect)
    aspect_rad = np.radians(aspect)
    northness = np.cos(aspect_rad)
    eastness  = np.sin(aspect_rad)
    return slope, northness, eastness


def calculate_tpi(dem, kernel_size=31):
    kernel = np.ones((kernel_size, kernel_size)) / (kernel_size**2 - 1)
    kernel[kernel_size // 2, kernel_size // 2] = 0
    mean_dem = cv2.filter2D(dem.astype(np.float32), -1, kernel,
                            borderType=cv2.BORDER_REPLICATE)
    return dem - mean_dem

# ---------------------------------------------------------------------------
# Carga de capas estaticas (Sx x8) en memoria — ~380 MB total
# ---------------------------------------------------------------------------

def load_sx_arrays():
    """Carga los 8 rasters Sx_100m a 1m. Devuelve lista ordenada por direccion."""
    print("Cargando capas Sx_100m (8 direcciones) en memoria...")
    sx_arrays = []
    sx_transform = None
    sx_crs = None
    for d in SX_DIRS:
        path = TOPO_DIR / f"sx_{d}_100.tif"
        with rasterio.open(path) as src:
            sx_arrays.append(src.read(1).astype(np.float32))
            if sx_transform is None:
                sx_transform = src.transform
                sx_crs = src.crs
    # Reemplazar nodata (~-3.4e38) por 0
    for i in range(len(sx_arrays)):
        sx_arrays[i] = np.where(np.isfinite(sx_arrays[i]), sx_arrays[i], 0.0)
    print(f"  Sx cargados: shape={sx_arrays[0].shape}, CRS={sx_crs}")
    return sx_arrays, sx_transform, sx_crs

# ---------------------------------------------------------------------------
# Construccion del indice de archivos SCE por fecha
# ---------------------------------------------------------------------------

def build_sce_index():
    """Devuelve dict {datetime: Path} de todos los TIF de SCE disponibles."""
    idx = {}
    for fpath in sorted(SCE_DIR.glob("izas_*.tif")):
        m = re.search(r'izas_(\d{8})\.tif', fpath.name)
        if m:
            idx[datetime.strptime(m.group(1), '%Y%m%d')] = fpath
    print(f"SCE index: {len(idx)} fechas disponibles "
          f"({min(idx):%Y-%m-%d} a {max(idx):%Y-%m-%d})")
    return idx

# ---------------------------------------------------------------------------
# Calculo de persistencia para una fecha y un crop DEM
# ---------------------------------------------------------------------------

def compute_persistence(target_date: datetime,
                        sce_index: dict,
                        dst_transform,
                        dst_crs,
                        dst_shape: tuple) -> np.ndarray:
    """
    Devuelve array (3, H, W) con persistencia 15d/30d/60d
    reproyectada al grid del crop DEM (dst_transform, dst_crs, dst_shape).
    """
    max_window = max(PERSIST_WINDOWS)
    # Recoger todos los SCE en la ventana maxima
    candidates = [
        (dt, path) for dt, path in sce_index.items()
        if target_date - timedelta(days=max_window) <= dt < target_date
    ]
    if not candidates:
        return np.zeros((len(PERSIST_WINDOWS), *dst_shape), dtype=np.float32)

    # Cargar todos los SCE en la ventana maxima y reproyectar a 1m
    snow_by_date = {}
    ref_transform = None
    ref_crs = None
    ref_shape = None

    for dt, path in candidates:
        with rasterio.open(path) as src:
            if ref_transform is None:
                ref_transform = src.transform
                ref_crs = src.crs
                ref_shape = (src.height, src.width)
            sce = src.read(1).astype(np.float32)
        snow_10m = (sce == 11).astype(np.float32)  # 1=nieve, 0=sin nieve/nodata

        # Reproyectar de 10m al grid destino (1m, crop del DEM)
        snow_1m = np.zeros(dst_shape, dtype=np.float32)
        reproject(
            source=snow_10m,
            destination=snow_1m,
            src_transform=ref_transform,
            src_crs=ref_crs,
            dst_transform=dst_transform,
            dst_crs=dst_crs,
            resampling=Resampling.bilinear,
        )
        snow_by_date[dt] = snow_1m

    # Calcular fraccion para cada ventana
    result = np.zeros((len(PERSIST_WINDOWS), *dst_shape), dtype=np.float32)
    for wi, window in enumerate(PERSIST_WINDOWS):
        start = target_date - timedelta(days=window)
        maps = [v for dt, v in snow_by_date.items() if dt >= start]
        if maps:
            result[wi] = np.mean(maps, axis=0)

    return result

# ---------------------------------------------------------------------------
# Script principal
# ---------------------------------------------------------------------------

def main():
    # Crear directorios de salida
    out_images = OUTPUT_DIR / "images"
    out_masks  = OUTPUT_DIR / "masks"
    out_images.mkdir(parents=True, exist_ok=True)
    out_masks.mkdir(parents=True, exist_ok=True)

    # Cargar Sx estaticos
    sx_arrays, sx_transform, sx_crs = load_sx_arrays()

    # Indice de SCE
    sce_index = build_sce_index()

    # Buscar todos los archivos LiDAR SnowDepth
    lidar_files = sorted(SNOW_DIR.glob("SD_*_1m.tif"))
    print(f"\nArchivos LiDAR encontrados: {len(lidar_files)}")

    metadata = []

    with rasterio.open(DEM_PATH) as src_dem:
        dem_profile  = src_dem.profile
        dem_transform_full = src_dem.transform
        dem_crs      = src_dem.crs

        for fp in tqdm(lidar_files, desc="Generando tiles v4-17ch"):
            fname    = fp.name
            date_str = fname.split('_')[1]          # e.g. '20210202'
            year     = int(date_str[:4])

            # Solo procesamos los anos del split
            if year not in (TRAIN_YEARS | VAL_YEARS | TEST_YEARS):
                continue

            # Fecha corregida para buscar SCE
            sce_date_str = DATE_FIXES_LIDAR.get(date_str, date_str)
            target_date  = datetime.strptime(sce_date_str, '%Y%m%d')

            # Stride segun split
            if year in TRAIN_YEARS:
                stride = STRIDE_TRAIN
            else:
                stride = STRIDE_EVALTEST

            # Split label
            if year in TRAIN_YEARS:
                split = 'train'
            elif year in VAL_YEARS:
                split = 'val'
            else:
                split = 'test'

            # -----------------------------------------------------------------
            # A. Calcular crop DEM alineado con este archivo LiDAR
            # -----------------------------------------------------------------
            with rasterio.open(fp) as src_target:
                try:
                    win           = src_dem.window(*src_target.bounds)
                    win           = win.round_offsets().round_lengths()
                    transform_crop = src_dem.window_transform(win)
                    dem_crop      = src_dem.read(1, window=win,
                                                 boundless=True,
                                                 fill_value=NODATA_VAL)
                    h, w = dem_crop.shape
                except Exception as e:
                    print(f"  [SKIP] {fname}: error DEM crop — {e}")
                    continue

                # Mascara (snow depth)
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

            # SCE (canal 5) — reproyectar desde 10m
            sce_path = SCE_DIR / f"izas_{sce_date_str}.tif"
            sce_crop = np.zeros((h, w), dtype=np.float32)
            if sce_path.exists():
                with rasterio.open(sce_path) as src_sce:
                    sce_raw = src_sce.read(1).astype(np.float32)
                    reproject(
                        source=sce_raw,
                        destination=sce_crop,
                        src_transform=src_sce.transform,
                        src_crs=src_sce.crs,
                        dst_transform=transform_crop,
                        dst_crs=dem_crs,
                        resampling=Resampling.bilinear,
                    )

            # -----------------------------------------------------------------
            # B. Canales topograficos (calculados igual que el notebook)
            # -----------------------------------------------------------------
            slope, northness, eastness = calculate_topography(dem_crop)
            tpi = calculate_tpi(dem_crop)

            # -----------------------------------------------------------------
            # C. Canales Sx (crop del raster global al area del DEM crop)
            # -----------------------------------------------------------------
            row_off = int(win.row_off)
            col_off = int(win.col_off)
            win_h   = int(win.height)
            win_w   = int(win.width)

            sx_crops = []
            for sx_arr in sx_arrays:
                # Extraer patch del raster Sx global
                r0 = max(row_off, 0)
                r1 = min(row_off + win_h, sx_arr.shape[0])
                c0 = max(col_off, 0)
                c1 = min(col_off + win_w, sx_arr.shape[1])
                patch = np.full((win_h, win_w), 0.0, dtype=np.float32)
                patch[r0 - row_off:r1 - row_off,
                      c0 - col_off:c1 - col_off] = sx_arr[r0:r1, c0:c1]
                sx_crops.append(patch)

            # -----------------------------------------------------------------
            # D. Persistencia (computada para este crop y esta fecha)
            # -----------------------------------------------------------------
            persistence = compute_persistence(
                target_date=target_date,
                sce_index=sce_index,
                dst_transform=transform_crop,
                dst_crs=dem_crs,
                dst_shape=(h, w),
            )

            # -----------------------------------------------------------------
            # E. Stack completo de 17 canales: [topo6] + [Sx8] + [pers3]
            # -----------------------------------------------------------------
            stack = np.stack(
                [dem_crop, slope, northness, eastness, tpi, sce_crop]
                + sx_crops
                + [persistence[0], persistence[1], persistence[2]],
                axis=0,
            ).astype(np.float32)

            # -----------------------------------------------------------------
            # F. Trocear en tiles y guardar
            # -----------------------------------------------------------------
            for ty in range(0, h - TILE_SIZE + 1, stride):
                for tx in range(0, w - TILE_SIZE + 1, stride):

                    tile_img  = stack[:, ty:ty + TILE_SIZE, tx:tx + TILE_SIZE]
                    tile_mask = sd_crop[ty:ty + TILE_SIZE, tx:tx + TILE_SIZE]

                    # Filtros de calidad
                    nodata_frac = np.mean(tile_img[0] == NODATA_VAL)
                    if nodata_frac > 0.10:
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
                        'tile_id':           tile_name,
                        'date':              date_str,
                        'year':              year,
                        'source':            'lidar',
                        'valid_frac':        round(float(valid_frac), 4),
                        'exp_temporal_split': split,
                    })

    # -------------------------------------------------------------------------
    # Guardar CSV
    # -------------------------------------------------------------------------
    df = pd.DataFrame(metadata)
    csv_path = OUTPUT_DIR / "dataset_v4_17ch.csv"
    df.to_csv(csv_path, index=False)

    print(f"\n{'='*60}")
    print(f"GENERACION COMPLETADA: {len(df)} tiles")
    print(f"Guardado en: {OUTPUT_DIR}")
    print(f"\nDistribucion por split:")
    counts = df['exp_temporal_split'].value_counts()
    total  = len(df)
    for split_name in ['train', 'val', 'test']:
        n = counts.get(split_name, 0)
        print(f"  {split_name:6s}: {n:5d}  ({100*n/total:.1f}%)")


if __name__ == '__main__':
    main()
