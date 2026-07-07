"""
Genera dataset_v4_2p5m: tiles a 2.5m de resolucion con 17 canales.
====================================================================

Motivacion:
  - A 1m (256x256 tiles) el modelo ve 256m x 256m → insuficiente para capturar
    los quiebres de escala de la nieve (primer quiebre ~20-30m, segundo ~200-500m).
  - A 5m (256x256 tiles) el modelo ve 1280m x 1280m → mas contexto pero pocas tiles.
  - A 2.5m (256x256 tiles) el modelo ve 640m x 640m → dentro del rango optimo
    600-800m sugerido por el experto del dominio, capturando ambos quiebres de escala.

Diferencias respecto a dataset_v4_17ch (1m):
  - Resolucion: 2.5m (remuestreado desde DEM 1m con bilinear)
  - Sx: radio 200m en lugar de 100m (mayor correlacion con distribucion de nieve)
  - Cobertura por tile: 256 * 2.5m = 640m x 640m

Canales (17, misma estructura que v4_17ch y v6_5m):
  [0]  DEM
  [1]  Slope
  [2]  Northness  (cos aspect)
  [3]  Eastness   (sin aspect)
  [4]  TPI        (kernel 31x31 @ 2.5m ~ 77m de radio)
  [5]  SCE        (valores brutos: 0/10/11, reproyectado de 10m a 2.5m)
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

Split temporal:
  Train : 2021, 2022, 2023  (stride=128, 50% overlap)
  Val   : 2024               (stride=256, sin overlap)
  Test  : 2025               (stride=256, sin overlap)

Uso:
    .venv\\Scripts\\python.exe data/generate_dataset_v4_2p5m.py
"""

import re
import numpy as np
import pandas as pd
import rasterio
from rasterio.enums import Resampling
from rasterio.warp import reproject
from rasterio.windows import from_bounds as window_from_bounds
from rasterio.transform import from_origin
from affine import Affine
import cv2
from datetime import datetime, timedelta
from pathlib import Path
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Rutas
# ---------------------------------------------------------------------------
_REPO = Path(__file__).resolve().parent.parent

BASE_DATA  = _REPO / "Articulo 1/Data/izas/LiDAR"
TOPO_DIR   = BASE_DATA / "Topografia"
SNOW_DIR   = BASE_DATA / "SnowDepth"
SCE_DIR    = BASE_DATA / "images_SCE"
DEM_PATH   = TOPO_DIR / "DEMbigIzas_1m.tif"     # DEM fuente a 1m
OUTPUT_DIR = _REPO / "dataset_v4_2p5m"

# ---------------------------------------------------------------------------
# Parametros
# ---------------------------------------------------------------------------
SCALE_FACTOR    = 2.5    # factor de remuestreo 1m -> 2.5m
TILE_SIZE       = 256    # 256 * 2.5m = 640m de cobertura por tile
STRIDE_TRAIN    = 128    # 50% overlap para train
STRIDE_EVALTEST = 256    # sin overlap para val/test
MIN_VALID_FRAC  = 0.30
NODATA_VAL      = -9999.0

TRAIN_YEARS = {2021, 2022, 2023}
VAL_YEARS   = {2024}
TEST_YEARS  = {2025}

SX_DIRS         = ['0', '45', '90', '135', '180', '225', '270', '315']
SX_RADIUS       = '200'              # 200m en lugar de 100m
PERSIST_WINDOWS = [15, 30, 60]

DATE_FIXES_LIDAR = {'20210608': '20210607'}


# ---------------------------------------------------------------------------
# Funciones topograficas
# ---------------------------------------------------------------------------

def calculate_topography(dem, pixel_size=1.0):
    """Calcula slope, northness, eastness. pixel_size en metros."""
    x, y = np.gradient(dem, pixel_size)
    slope   = np.arctan(np.sqrt(x**2 + y**2)) * (180 / np.pi)
    aspect  = np.arctan2(-x, y) * (180 / np.pi)
    aspect  = np.where(aspect < 0, aspect + 360, aspect)
    rad     = np.radians(aspect)
    return slope, np.cos(rad), np.sin(rad)


def calculate_tpi(dem, kernel_size=31):
    """TPI con kernel NxN. A 2.5m, kernel 31x31 ~ radio 77m."""
    kernel = np.ones((kernel_size, kernel_size)) / (kernel_size**2 - 1)
    kernel[kernel_size // 2, kernel_size // 2] = 0
    mean_dem = cv2.filter2D(dem.astype(np.float32), -1, kernel,
                            borderType=cv2.BORDER_REPLICATE)
    return dem - mean_dem


# ---------------------------------------------------------------------------
# Remuestreo del DEM 1m -> 2.5m y computo del transform destino
# ---------------------------------------------------------------------------

def resample_dem_to_2p5m(dem_path: Path):
    """
    Carga el DEM a 1m y lo remuestrea a 2.5m en memoria.
    Devuelve (dem_2p5m, transform_2p5m, crs).
    """
    print(f"Remuestreando DEM 1m -> 2.5m (factor {SCALE_FACTOR})...")
    with rasterio.open(dem_path) as src:
        src_transform = src.transform
        src_crs       = src.crs
        src_h, src_w  = src.height, src.width

        # Dimensiones del raster destino
        dst_w = int(src_w / SCALE_FACTOR)
        dst_h = int(src_h / SCALE_FACTOR)

        # Transform destino: mismo origen, pixel_size * SCALE_FACTOR
        dst_transform = Affine(
            src_transform.a * SCALE_FACTOR, src_transform.b, src_transform.c,
            src_transform.d, src_transform.e * SCALE_FACTOR, src_transform.f,
        )

        dem_2p5m = np.zeros((dst_h, dst_w), dtype=np.float32)
        reproject(
            source=rasterio.band(src, 1),
            destination=dem_2p5m,
            src_transform=src_transform,
            src_crs=src_crs,
            dst_transform=dst_transform,
            dst_crs=src_crs,
            resampling=Resampling.bilinear,
        )

    # Reemplazar nodata
    dem_2p5m[dem_2p5m < -9000] = NODATA_VAL
    print(f"  DEM 2.5m: shape={dem_2p5m.shape}, "
          f"rango=[{dem_2p5m[dem_2p5m > -9000].min():.0f}, "
          f"{dem_2p5m[dem_2p5m > -9000].max():.0f}] m")
    return dem_2p5m, dst_transform, src_crs


# ---------------------------------------------------------------------------
# Carga de Sx_200m a 1m y remuestreo a 2.5m
# ---------------------------------------------------------------------------

def load_sx_arrays_2p5m(dem_1m_transform, dem_1m_crs, dst_h, dst_w, dst_transform):
    """
    Carga los 8 rasters Sx_200m a 1m y los remuestrea a 2.5m.
    Misma forma que dem_2p5m.
    """
    print(f"Cargando y remuestreando Sx_200m (8 dirs) a 2.5m...")
    sx_arrays = []
    for d in SX_DIRS:
        path = TOPO_DIR / f"sx_{d}_{SX_RADIUS}.tif"
        with rasterio.open(path) as src:
            sx_1m = src.read(1).astype(np.float32)
            sx_1m = np.where(np.isfinite(sx_1m), sx_1m, 0.0)  # nodata -> 0
            sx_t  = src.transform
            sx_c  = src.crs

        sx_2p5m = np.zeros((dst_h, dst_w), dtype=np.float32)
        reproject(
            source=sx_1m,
            destination=sx_2p5m,
            src_transform=sx_t,
            src_crs=sx_c,
            dst_transform=dst_transform,
            dst_crs=dem_1m_crs,
            resampling=Resampling.bilinear,
        )
        sx_arrays.append(sx_2p5m)

    print(f"  Sx_200m @ 2.5m: shape={sx_arrays[0].shape}")
    return sx_arrays


# ---------------------------------------------------------------------------
# Indice SCE
# ---------------------------------------------------------------------------

def build_sce_index():
    idx = {}
    for fpath in sorted(SCE_DIR.glob("izas_*.tif")):
        m = re.search(r'izas_(\d{8})\.tif', fpath.name)
        if m:
            idx[datetime.strptime(m.group(1), '%Y%m%d')] = fpath
    print(f"SCE index: {len(idx)} fechas "
          f"({min(idx):%Y-%m-%d} a {max(idx):%Y-%m-%d})")
    return idx


# ---------------------------------------------------------------------------
# Persistencia a 2.5m
# ---------------------------------------------------------------------------

def compute_persistence(target_date, sce_index, dst_transform, dst_crs, dst_shape):
    """Fraccion dias nevados en ventanas 15/30/60d, reproyectado a grid 2.5m."""
    max_window = max(PERSIST_WINDOWS)
    candidates = [
        (dt, path) for dt, path in sce_index.items()
        if target_date - timedelta(days=max_window) <= dt < target_date
    ]
    if not candidates:
        return np.zeros((len(PERSIST_WINDOWS), *dst_shape), dtype=np.float32)

    snow_by_date = {}
    for dt, path in candidates:
        with rasterio.open(path) as src:
            sce    = src.read(1).astype(np.float32)
            src_tr = src.transform
            src_cr = src.crs
        snow_10m = (sce == 11).astype(np.float32)

        snow_2p5m = np.zeros(dst_shape, dtype=np.float32)
        reproject(
            source=snow_10m,
            destination=snow_2p5m,
            src_transform=src_tr,
            src_crs=src_cr,
            dst_transform=dst_transform,
            dst_crs=dst_crs,
            resampling=Resampling.bilinear,
        )
        snow_by_date[dt] = snow_2p5m

    result = np.zeros((len(PERSIST_WINDOWS), *dst_shape), dtype=np.float32)
    for wi, window in enumerate(PERSIST_WINDOWS):
        start = target_date - timedelta(days=window)
        maps  = [v for dt, v in snow_by_date.items() if dt >= start]
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

    # ------------------------------------------------------------------
    # 1. DEM remuestreado a 2.5m
    # ------------------------------------------------------------------
    dem_2p5m, dem_2p5m_transform, dem_crs = resample_dem_to_2p5m(DEM_PATH)
    dst_h, dst_w = dem_2p5m.shape

    # Necesitamos el transform original 1m para cargar Sx
    with rasterio.open(DEM_PATH) as src:
        dem_1m_transform = src.transform
        dem_1m_crs       = src.crs

    # ------------------------------------------------------------------
    # 2. Sx_200m remuestreado a 2.5m (cargado en memoria, ~50 MB total)
    # ------------------------------------------------------------------
    sx_arrays = load_sx_arrays_2p5m(
        dem_1m_transform, dem_1m_crs, dst_h, dst_w, dem_2p5m_transform
    )

    # ------------------------------------------------------------------
    # 3. Indice SCE
    # ------------------------------------------------------------------
    sce_index = build_sce_index()

    # ------------------------------------------------------------------
    # 4. Procesar cada fecha LiDAR
    # ------------------------------------------------------------------
    lidar_files = sorted(SNOW_DIR.glob("SD_*_1m.tif"))
    print(f"\nArchivos LiDAR encontrados: {len(lidar_files)}")

    metadata = []

    for fp in tqdm(lidar_files, desc="Generando tiles v4-2p5m"):
        fname    = fp.name
        date_str = fname.split('_')[1]
        year     = int(date_str[:4])

        if year not in (TRAIN_YEARS | VAL_YEARS | TEST_YEARS):
            continue

        sce_date_str = DATE_FIXES_LIDAR.get(date_str, date_str)
        target_date  = datetime.strptime(sce_date_str, '%Y%m%d')

        stride = STRIDE_TRAIN if year in TRAIN_YEARS else STRIDE_EVALTEST
        split  = ('train' if year in TRAIN_YEARS else
                  'val'   if year in VAL_YEARS   else 'test')

        # --------------------------------------------------------------
        # A. Encontrar crop en el DEM 2.5m usando los bounds del LiDAR
        # --------------------------------------------------------------
        with rasterio.open(fp) as src_lidar:
            lidar_bounds = src_lidar.bounds

        try:
            win = window_from_bounds(
                lidar_bounds.left, lidar_bounds.bottom,
                lidar_bounds.right, lidar_bounds.top,
                dem_2p5m_transform,
            ).round_offsets().round_lengths()

            row_off = max(0, int(win.row_off))
            col_off = max(0, int(win.col_off))
            win_h   = int(win.height)
            win_w   = int(win.width)

            # Clamp a los limites del DEM
            row_end = min(row_off + win_h, dst_h)
            col_end = min(col_off + win_w, dst_w)
            win_h   = row_end - row_off
            win_w   = col_end - col_off

            if win_h <= 0 or win_w <= 0:
                continue

            dem_crop       = dem_2p5m[row_off:row_end, col_off:col_end]
            transform_crop = dem_2p5m_transform * Affine.translation(col_off, row_off)

        except Exception as e:
            print(f"  [SKIP] {fname}: error crop 2.5m — {e}")
            continue

        h, w = dem_crop.shape
        if h < TILE_SIZE or w < TILE_SIZE:
            continue

        # --------------------------------------------------------------
        # B. Snow depth LiDAR reproyectado a 2.5m
        # --------------------------------------------------------------
        sd_crop = np.zeros((h, w), dtype=np.float32)
        with rasterio.open(fp) as src_lidar:
            reproject(
                source=rasterio.band(src_lidar, 1),
                destination=sd_crop,
                src_transform=src_lidar.transform,
                src_crs=src_lidar.crs,
                dst_transform=transform_crop,
                dst_crs=dem_crs,
                resampling=Resampling.bilinear,
            )

        # --------------------------------------------------------------
        # C. SCE reproyectado a 2.5m
        # --------------------------------------------------------------
        sce_crop = np.zeros((h, w), dtype=np.float32)
        sce_path = SCE_DIR / f"izas_{sce_date_str}.tif"
        if sce_path.exists():
            with rasterio.open(sce_path) as src_sce:
                reproject(
                    source=src_sce.read(1).astype(np.float32),
                    destination=sce_crop,
                    src_transform=src_sce.transform,
                    src_crs=src_sce.crs,
                    dst_transform=transform_crop,
                    dst_crs=dem_crs,
                    resampling=Resampling.bilinear,
                )

        # --------------------------------------------------------------
        # D. Canales topograficos calculados sobre DEM 2.5m
        # --------------------------------------------------------------
        slope, northness, eastness = calculate_topography(dem_crop,
                                                          pixel_size=SCALE_FACTOR)
        tpi = calculate_tpi(dem_crop)

        # --------------------------------------------------------------
        # E. Sx_200m crop del raster 2.5m global
        # --------------------------------------------------------------
        sx_crops = []
        for sx_arr in sx_arrays:
            patch = sx_arr[row_off:row_end, col_off:col_end]
            sx_crops.append(patch)

        # --------------------------------------------------------------
        # F. Persistencia a 2.5m
        # --------------------------------------------------------------
        persistence = compute_persistence(
            target_date=target_date,
            sce_index=sce_index,
            dst_transform=transform_crop,
            dst_crs=dem_crs,
            dst_shape=(h, w),
        )

        # --------------------------------------------------------------
        # G. Stack 17 canales
        # --------------------------------------------------------------
        stack = np.stack(
            [dem_crop, slope, northness, eastness, tpi, sce_crop]
            + sx_crops
            + [persistence[0], persistence[1], persistence[2]],
            axis=0,
        ).astype(np.float32)

        # --------------------------------------------------------------
        # H. Trocear en tiles y guardar
        # --------------------------------------------------------------
        for ty in range(0, h - TILE_SIZE + 1, stride):
            for tx in range(0, w - TILE_SIZE + 1, stride):

                tile_img  = stack[:, ty:ty + TILE_SIZE, tx:tx + TILE_SIZE]
                tile_mask = sd_crop[ty:ty + TILE_SIZE, tx:tx + TILE_SIZE]

                # Filtros de calidad
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

    # ------------------------------------------------------------------
    # Guardar CSV
    # ------------------------------------------------------------------
    df = pd.DataFrame(metadata)
    csv_path = OUTPUT_DIR / "dataset_v4_2p5m.csv"
    df.to_csv(csv_path, index=False)

    print(f"\n{'='*60}")
    print(f"GENERACION COMPLETADA: {len(df)} tiles")
    print(f"Cobertura por tile: {TILE_SIZE * SCALE_FACTOR:.0f}m x "
          f"{TILE_SIZE * SCALE_FACTOR:.0f}m")
    print(f"Guardado en: {OUTPUT_DIR}")
    print(f"\nDistribucion por split:")
    counts = df['exp_temporal_split'].value_counts()
    total  = len(df)
    for s in ['train', 'val', 'test']:
        n = counts.get(s, 0)
        print(f"  {s:6s}: {n:5d}  ({100*n/total:.1f}%)")


if __name__ == '__main__':
    main()
