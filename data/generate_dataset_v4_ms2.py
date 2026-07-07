"""
Genera dataset_v4_ms2: tiles 1m con 30 canales (multi-escala extendido).
========================================================================

Extiende v4_ms añadiendo los 8 canales Sx tambien a escala 5m.
Los Sx_1m se downsamplea a 5m (block-average, INTER_AREA) y se upsamplea
de vuelta a 256x256 por tile, igual que los canales topograficos de contexto.

Canales (30):
  --- Escala fina (1m) ---
  [0]  DEM
  [1]  Slope
  [2]  Northness
  [3]  Eastness
  [4]  TPI
  [5]  SCE
  [6-13] Sx_100m x8 (0,45,90,135,180,225,270,315)
  [14] Persistencia_15d
  [15] Persistencia_30d
  [16] Persistencia_60d
  --- Contexto grueso 5m (topo) ---
  [17] DEM_5m
  [18] Slope_5m
  [19] Northness_5m
  [20] Eastness_5m
  [21] TPI_5m
  --- Contexto grueso 5m (Sx) ---
  [22] Sx_5m_0
  [23] Sx_5m_45
  [24] Sx_5m_90
  [25] Sx_5m_135
  [26] Sx_5m_180
  [27] Sx_5m_225
  [28] Sx_5m_270
  [29] Sx_5m_315

Split temporal identico a v4_17ch / v4_ms.

Uso:
    .venv\\Scripts\\python.exe data/generate_dataset_v4_ms2.py
"""

import re
import numpy as np
import pandas as pd
import rasterio
from rasterio.enums import Resampling
from rasterio.warp import reproject
from rasterio.transform import array_bounds
import rasterio.windows
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
DEM_1M_PATH  = TOPO_DIR / "DEMbigIzas_1m.tif"
DEM_5M_PATH  = TOPO_DIR / "DEMbigIzas_5m.tif"
OUTPUT_DIR   = _REPO / "dataset_v4_ms2"

# ---------------------------------------------------------------------------
# Parametros
# ---------------------------------------------------------------------------
TILE_SIZE       = 256
STRIDE_TRAIN    = 128
STRIDE_EVALTEST = 256
MIN_VALID_FRAC  = 0.30
NODATA_VAL      = -9999.0

TRAIN_YEARS = {2021, 2022, 2023}
VAL_YEARS   = {2024}
TEST_YEARS  = {2025}

SX_DIRS = ['0', '45', '90', '135', '180', '225', '270', '315']
PERSIST_WINDOWS = [15, 30, 60]
DATE_FIXES_LIDAR = {'20210608': '20210607'}

# ---------------------------------------------------------------------------
# Funciones topograficas
# ---------------------------------------------------------------------------

def calculate_topography(dem, pixel_size=1.0):
    x, y = np.gradient(dem, pixel_size)
    slope = np.arctan(np.sqrt(x**2 + y**2)) * (180 / np.pi)
    aspect = np.arctan2(-x, y) * (180 / np.pi)
    aspect = np.where(aspect < 0, aspect + 360, aspect)
    aspect_rad = np.radians(aspect)
    return slope, np.cos(aspect_rad), np.sin(aspect_rad)


def calculate_tpi(dem, kernel_size=31):
    kernel = np.ones((kernel_size, kernel_size)) / (kernel_size**2 - 1)
    kernel[kernel_size // 2, kernel_size // 2] = 0
    mean_dem = cv2.filter2D(dem.astype(np.float32), -1, kernel,
                            borderType=cv2.BORDER_REPLICATE)
    return dem - mean_dem


# ---------------------------------------------------------------------------
# Carga de capas estaticas
# ---------------------------------------------------------------------------

def load_sx_arrays():
    """Carga Sx_100m a 1m y devuelve tambien version downsampled a 5m."""
    print("Cargando capas Sx_100m (8 dirs) y downsampling a 5m...")
    sx_1m = []
    sx_transform = None
    for d in SX_DIRS:
        path = TOPO_DIR / f"sx_{d}_100.tif"
        with rasterio.open(path) as src:
            arr = src.read(1).astype(np.float32)
            if sx_transform is None:
                sx_transform = src.transform
            arr = np.where(np.isfinite(arr), arr, 0.0)
            sx_1m.append(arr)

    # Downsample 1m → 5m usando block-average (INTER_AREA)
    h1, w1 = sx_1m[0].shape
    h5, w5 = h1 // 5, w1 // 5
    sx_5m = []
    for arr in sx_1m:
        arr_5m = cv2.resize(arr, (w5, h5), interpolation=cv2.INTER_AREA)
        sx_5m.append(arr_5m)

    print(f"  Sx_1m shape={sx_1m[0].shape}, Sx_5m shape={sx_5m[0].shape}")
    return sx_1m, sx_5m, sx_transform


def load_context_5m():
    """Carga DEM 5m y calcula features topograficas de contexto."""
    print("Cargando DEM 5m y calculando features de contexto...")
    with rasterio.open(DEM_5M_PATH) as src:
        dem5 = src.read(1).astype(np.float32)
        transform_5m = src.transform
        pixel_size_5m = abs(src.res[0])

    dem5 = np.where(np.isfinite(dem5), dem5,
                    np.nanmedian(dem5[np.isfinite(dem5)]))

    slope5, north5, east5 = calculate_topography(dem5, pixel_size=pixel_size_5m)
    tpi5 = calculate_tpi(dem5, kernel_size=31)

    ctx = {
        'dem':       dem5,
        'slope':     slope5,
        'northness': north5,
        'eastness':  east5,
        'tpi':       tpi5,
    }
    print(f"  Contexto topo 5m listo: shape={dem5.shape}")
    return ctx, transform_5m


def extract_context_patch(ctx_topo, sx_5m, transform_5m, bounds_1m, out_h, out_w):
    """
    Extrae los 13 canales de contexto 5m (5 topo + 8 Sx) para un tile,
    upsampled a (out_h, out_w).
    Devuelve array (13, out_h, out_w).
    """
    win = rasterio.windows.from_bounds(*bounds_1m, transform=transform_5m)
    win = win.round_offsets().round_lengths()

    row_off = int(win.row_off)
    col_off = int(win.col_off)
    win_h   = max(1, int(win.height))
    win_w   = max(1, int(win.width))

    # Helper para extraer patch de un array 5m
    def _patch(arr):
        h_arr, w_arr = arr.shape
        r0 = max(row_off, 0);  r1 = min(row_off + win_h, h_arr)
        c0 = max(col_off, 0);  c1 = min(col_off + win_w, w_arr)
        patch = np.zeros((win_h, win_w), dtype=np.float32)
        if r1 > r0 and c1 > c0:
            patch[r0-row_off:r1-row_off, c0-col_off:c1-col_off] = arr[r0:r1, c0:c1]
        return cv2.resize(patch, (out_w, out_h), interpolation=cv2.INTER_LINEAR)

    topo_keys = ['dem', 'slope', 'northness', 'eastness', 'tpi']
    topo_patches = [_patch(ctx_topo[k]) for k in topo_keys]   # 5 canales
    sx_patches   = [_patch(arr) for arr in sx_5m]              # 8 canales

    return np.stack(topo_patches + sx_patches, axis=0)  # (13, out_h, out_w)


# ---------------------------------------------------------------------------
# SCE e indice
# ---------------------------------------------------------------------------

def build_sce_index():
    idx = {}
    for fpath in sorted(SCE_DIR.glob("izas_*.tif")):
        m = re.search(r'izas_(\d{8})\.tif', fpath.name)
        if m:
            idx[datetime.strptime(m.group(1), '%Y%m%d')] = fpath
    print(f"SCE index: {len(idx)} fechas")
    return idx


def compute_persistence(target_date, sce_index, dst_transform, dst_crs, dst_shape):
    max_window = max(PERSIST_WINDOWS)
    candidates = [(dt, p) for dt, p in sce_index.items()
                  if target_date - timedelta(days=max_window) <= dt < target_date]
    if not candidates:
        return np.zeros((len(PERSIST_WINDOWS), *dst_shape), dtype=np.float32)

    snow_by_date = {}
    ref_transform = ref_crs = None
    for dt, path in candidates:
        with rasterio.open(path) as src:
            if ref_transform is None:
                ref_transform, ref_crs = src.transform, src.crs
            sce = src.read(1).astype(np.float32)
        snow_10m = (sce == 11).astype(np.float32)
        snow_1m = np.zeros(dst_shape, dtype=np.float32)
        reproject(source=snow_10m, destination=snow_1m,
                  src_transform=ref_transform, src_crs=ref_crs,
                  dst_transform=dst_transform, dst_crs=dst_crs,
                  resampling=Resampling.bilinear)
        snow_by_date[dt] = snow_1m

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
    out_images = OUTPUT_DIR / "images"
    out_masks  = OUTPUT_DIR / "masks"
    out_images.mkdir(parents=True, exist_ok=True)
    out_masks.mkdir(parents=True, exist_ok=True)

    sx_1m, sx_5m, sx_transform = load_sx_arrays()
    ctx_topo, transform_5m     = load_context_5m()
    sce_index                  = build_sce_index()

    lidar_files = sorted(SNOW_DIR.glob("SD_*_1m.tif"))
    print(f"\nArchivos LiDAR encontrados: {len(lidar_files)}")

    metadata = []

    with rasterio.open(DEM_1M_PATH) as src_dem:
        dem_crs = src_dem.crs

        for fp in tqdm(lidar_files, desc="Generando tiles v4-ms2"):
            fname    = fp.name
            date_str = fname.split('_')[1]
            year     = int(date_str[:4])

            if year not in (TRAIN_YEARS | VAL_YEARS | TEST_YEARS):
                continue

            sce_date_str = DATE_FIXES_LIDAR.get(date_str, date_str)
            target_date  = datetime.strptime(sce_date_str, '%Y%m%d')
            stride = STRIDE_TRAIN if year in TRAIN_YEARS else STRIDE_EVALTEST
            split  = ('train' if year in TRAIN_YEARS
                      else 'val' if year in VAL_YEARS else 'test')

            # DEM crop
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

                sd_crop = np.zeros((h, w), dtype=np.float32)
                reproject(source=rasterio.band(src_target, 1),
                          destination=sd_crop,
                          src_transform=src_target.transform,
                          src_crs=src_target.crs,
                          dst_transform=transform_crop,
                          dst_crs=dem_crs,
                          resampling=Resampling.bilinear)

            # SCE
            sce_path = SCE_DIR / f"izas_{sce_date_str}.tif"
            sce_crop = np.zeros((h, w), dtype=np.float32)
            if sce_path.exists():
                with rasterio.open(sce_path) as src_sce:
                    sce_raw = src_sce.read(1).astype(np.float32)
                    reproject(source=sce_raw, destination=sce_crop,
                              src_transform=src_sce.transform,
                              src_crs=src_sce.crs,
                              dst_transform=transform_crop,
                              dst_crs=dem_crs,
                              resampling=Resampling.bilinear)

            # Topo fino (1m)
            slope, northness, eastness = calculate_topography(dem_crop, 1.0)
            tpi = calculate_tpi(dem_crop)

            # Sx fino (1m)
            row_off, col_off = int(win.row_off), int(win.col_off)
            win_h,   win_w   = int(win.height),  int(win.width)
            sx_crops = []
            for sx_arr in sx_1m:
                r0 = max(row_off, 0);  r1 = min(row_off + win_h, sx_arr.shape[0])
                c0 = max(col_off, 0);  c1 = min(col_off + win_w, sx_arr.shape[1])
                patch = np.zeros((win_h, win_w), dtype=np.float32)
                patch[r0-row_off:r1-row_off, c0-col_off:c1-col_off] = sx_arr[r0:r1, c0:c1]
                sx_crops.append(patch)

            # Persistencia
            persistence = compute_persistence(
                target_date, sce_index, transform_crop, dem_crs, (h, w))

            # Contexto 5m: 5 topo + 8 Sx = 13 canales
            bounds_crop = array_bounds(h, w, transform_crop)
            ctx_channels = extract_context_patch(
                ctx_topo, sx_5m, transform_5m, bounds_crop, h, w)

            # Stack: 17 + 13 = 30 canales
            stack = np.concatenate([
                np.stack(
                    [dem_crop, slope, northness, eastness, tpi, sce_crop]
                    + sx_crops
                    + [persistence[0], persistence[1], persistence[2]],
                    axis=0,
                ).astype(np.float32),
                ctx_channels,
            ], axis=0)

            # Trocear en tiles
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

    df = pd.DataFrame(metadata)
    df.to_csv(OUTPUT_DIR / "dataset_v4_ms2.csv", index=False)

    print(f"\n{'='*60}")
    print(f"GENERACION COMPLETADA: {len(df)} tiles  |  30 canales")
    print(f"  [0-16]  detalle 1m  (DEM, Slope, N, E, TPI, SCE, Sx×8, Pers×3)")
    print(f"  [17-21] contexto 5m topo  (DEM, Slope, N, E, TPI)")
    print(f"  [22-29] contexto 5m Sx    (8 direcciones)")
    print(f"Guardado en: {OUTPUT_DIR}")
    counts = df['exp_temporal_split'].value_counts()
    total  = len(df)
    for s in ['train', 'val', 'test']:
        n = counts.get(s, 0)
        print(f"  {s:6s}: {n:5d}  ({100*n/total:.1f}%)")


if __name__ == '__main__':
    main()
