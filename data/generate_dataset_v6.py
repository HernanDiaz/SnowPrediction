"""
Generacion de dataset_v6_5m: igual que v5 pero con 24 canales Sx adicionales.
==============================================================================

Canales de salida (30 total):
  [0]  DEM
  [1]  Slope
  [2]  Northness
  [3]  Eastness
  [4]  TPI
  [5]  SCE (Snow Cover Extent, Sentinel-2)
  [6-29] Sx (Wind Shelter Index):
         sx_0_50, sx_0_100, sx_0_200,
         sx_45_50, sx_45_100, sx_45_200,
         sx_90_50, sx_90_100, sx_90_200,
         sx_135_50, sx_135_100, sx_135_200,
         sx_180_50, sx_180_100, sx_180_200,
         sx_225_50, sx_225_100, sx_225_200,
         sx_270_50, sx_270_100, sx_270_200,
         sx_315_50, sx_315_100, sx_315_200

Dataset generado: dataset_v6_5m  (mismo CSV que v5)
"""

import os
import glob
import numpy as np
import pandas as pd
import rasterio
from rasterio.enums import Resampling
from rasterio.warp import reproject
import cv2
from tqdm import tqdm

# ── Rutas ─────────────────────────────────────────────────────────────────────
BASE_DIR    = r"E:\PycharmProjects\SnowPrediction\Articulo 1\Data"
TOPO_DIR    = os.path.join(BASE_DIR, "izas", "LiDAR", "Topografia")
OUTPUT_DIR  = os.path.join(BASE_DIR, "processed", "dataset_v6_5m")
DEM_PATH    = os.path.join(TOPO_DIR, "DEMbigIzas_5m.tif")
DIR_SNOWDEPTH = os.path.join(BASE_DIR, "izas", "LiDAR", "SnowDepth")
DIR_PLEIADES  = os.path.join(BASE_DIR, "izas", "Pleaiades")
DIR_INPUTS    = os.path.join(BASE_DIR, "izas", "LiDAR", "images_SCE")

# ── Parámetros de tiling ──────────────────────────────────────────────────────
TILE_SIZE   = 256
STRIDE      = 128
NODATA_VAL  = -9999.0

WEIGHT_LIDAR    = 1.0
WEIGHT_PLEIADES = 0.5

# ── Orden fijo de los 24 canales Sx ───────────────────────────────────────────
SX_DIRECTIONS = [0, 45, 90, 135, 180, 225, 270, 315]
SX_RADIUS     = 100   # Radio unico: 100m (compromiso entre detalle local y escala mesotopografica)
SX_NAMES = [f"sx_{d}_{SX_RADIUS}" for d in SX_DIRECTIONS]   # 8 nombres

# Ventanas de persistencia nival (dias)
PERSISTENCE_WINDOWS = [15, 30, 60]

os.makedirs(os.path.join(OUTPUT_DIR, "images"), exist_ok=True)
os.makedirs(os.path.join(OUTPUT_DIR, "masks"),  exist_ok=True)


# ─────────────────────────────────────────────────────────────────────────────
# Funciones auxiliares
# ─────────────────────────────────────────────────────────────────────────────
def calculate_topography(dem, resolucion=5.0):
    y, x = np.gradient(dem, resolucion, resolucion)
    slope   = np.degrees(np.arctan(np.sqrt(x**2 + y**2)))
    aspect  = np.degrees(np.arctan2(y, -x))
    aspect  = np.where(aspect < 0, aspect + 360, aspect)
    aspect_rad = np.radians(aspect)
    northness  = np.cos(aspect_rad)
    eastness   = np.sin(aspect_rad)
    return slope, northness, eastness


def calculate_tpi(dem, kernel_size=7):
    kernel = np.ones((kernel_size, kernel_size)) / (kernel_size**2 - 1)
    kernel[kernel_size//2, kernel_size//2] = 0
    mean_dem = cv2.filter2D(dem, -1, kernel, borderType=cv2.BORDER_REPLICATE)
    return dem - mean_dem


def analyze_yearly_peaks(file_list):
    print("Analizando picos de nieve anuales...")
    yearly_stats = {}
    for fp in tqdm(file_list, desc="Escaneando maximos"):
        filename = os.path.basename(fp)
        try:
            date_str = filename.split('_')[1]
            year = int(date_str[:4])
        except:
            continue

        with rasterio.open(fp) as src:
            data = src.read(1, out_shape=(src.height//10, src.width//10))
            valid_data = data[data > -100]
            if valid_data.size == 0:
                continue
            mean_sd = np.mean(valid_data)

        if year not in yearly_stats or mean_sd > yearly_stats[year]['max_sd']:
            yearly_stats[year] = {'date': date_str, 'max_sd': mean_sd}

    return {year: stats['date'] for year, stats in sorted(yearly_stats.items())}


# ─────────────────────────────────────────────────────────────────────────────
# Carga de todas las imágenes SCE remuestreadas a 5m (una sola vez)
# ─────────────────────────────────────────────────────────────────────────────
def load_all_sce_5m(dem_height, dem_width, dem_profile):
    """
    Carga todos los rasters SCE (10m) reproyectados al grid del DEM (5m).
    Retorna dict {date_str: array (H,W) uint8} donde:
        0  = sin observacion valida (nube)
        10 = sin nieve
        11 = con nieve
    """
    from datetime import datetime
    print("\nCargando imagenes SCE remuestreadas a 5m...")
    sce_files = sorted(glob.glob(os.path.join(DIR_INPUTS, "izas_*.tif")))
    sce_dict  = {}

    for fp in tqdm(sce_files, desc="  SCE rasters"):
        basename  = os.path.basename(fp)
        date_str  = basename.replace("izas_", "").replace(".tif", "").replace("(1)", "").strip()
        if len(date_str) != 8:
            continue
        try:
            datetime.strptime(date_str, "%Y%m%d")
        except ValueError:
            continue

        dst = np.zeros((dem_height, dem_width), dtype=np.float32)
        with rasterio.open(fp) as src:
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


def compute_persistence_maps(target_date_str, sce_dict, windows, dem_height, dem_width):
    """
    Para una fecha objetivo, calcula mapas de persistencia nival para cada ventana.
    Persistencia = fraccion de dias con nieve validos (valor==11) / total dias validos (valor==10 o 11).
    Retorna array (len(windows), H, W) con valores en [0,1].
    """
    from datetime import datetime, timedelta
    target_dt = datetime.strptime(target_date_str, "%Y%m%d")
    result    = np.zeros((len(windows), dem_height, dem_width), dtype=np.float32)

    for wi, w in enumerate(windows):
        cutoff_dt = target_dt - timedelta(days=w)
        # Seleccionar fechas SCE dentro de la ventana (sin incluir el dia objetivo)
        imgs_in_window = []
        for date_str, arr in sce_dict.items():
            dt = datetime.strptime(date_str, "%Y%m%d")
            if cutoff_dt <= dt < target_dt:
                imgs_in_window.append(arr)

        if len(imgs_in_window) == 0:
            # Sin datos en la ventana: persistencia = 0
            continue

        stack      = np.stack(imgs_in_window, axis=0)   # (N, H, W)
        snow_days  = np.sum(stack >= 10.5, axis=0).astype(np.float32)   # valor==11
        valid_days = np.sum(stack  >= 9.5, axis=0).astype(np.float32)   # valor==10 o 11

        with np.errstate(divide='ignore', invalid='ignore'):
            pers = np.where(valid_days > 0, snow_days / valid_days, 0.0)

        result[wi] = pers.astype(np.float32)

    return result


# ─────────────────────────────────────────────────────────────────────────────
# Carga de Sx a 5m (una sola vez, en memoria)
# ─────────────────────────────────────────────────────────────────────────────
def load_sx_5m(dem_height, dem_width):
    """
    Carga los 24 rasters Sx (1m) remuestreados a la resolucion del DEM 5m.
    Retorna un array (24, H, W) en el mismo grid que el DEM.
    """
    print("\nCargando 24 rasters Sx remuestreados a 5m...")
    sx_stack = np.zeros((24, dem_height, dem_width), dtype=np.float32)

    for i, name in enumerate(tqdm(SX_NAMES, desc="  Sx rasters")):
        path = os.path.join(TOPO_DIR, f"{name}.tif")
        if not os.path.exists(path):
            print(f"  AVISO: no encontrado {path}, canal {i} sera cero")
            continue
        with rasterio.open(path) as src:
            data = src.read(
                1,
                out_shape=(dem_height, dem_width),
                resampling=Resampling.average
            ).astype(np.float32)
            # Reemplazar nodata con 0
            nodata = src.nodata
            if nodata is not None:
                data[data == nodata] = 0.0
            sx_stack[i] = data

    print(f"  Sx cargados: shape={sx_stack.shape} | "
          f"min={sx_stack.min():.2f} | max={sx_stack.max():.2f}")
    return sx_stack


# ─────────────────────────────────────────────────────────────────────────────
# Proceso principal
# ─────────────────────────────────────────────────────────────────────────────
def process_dataset():
    # 1. Listar archivos SD
    files_lidar    = glob.glob(os.path.join(DIR_SNOWDEPTH, "SD_*.tif"))
    files_pleiades = glob.glob(os.path.join(DIR_PLEIADES,  "SD_*.tif"))
    all_files = files_lidar + files_pleiades

    if not all_files:
        print("Error: no se encontraron archivos SD.")
        return

    peaks_per_year = analyze_yearly_peaks(all_files)

    # 2. Abrir DEM y pre-cargar Sx
    with rasterio.open(DEM_PATH) as src_dem:
        dem_h, dem_w = src_dem.height, src_dem.width
        dem_profile  = src_dem.profile

        sx_full  = load_sx_5m(dem_h, dem_w)          # (24, H_dem, W_dem)
        sce_dict = load_all_sce_5m(dem_h, dem_w, dem_profile)   # {date: (H,W)}

        metadata_list = []

        # 3. Bucle principal
        for fp in tqdm(all_files, desc="Generando tiles v6"):
            filename = os.path.basename(fp)
            try:
                date_str = filename.split('_')[1]
                year     = int(date_str[:4])
            except:
                continue

            # Fuente y peso
            path_lower = fp.lower().replace('\\', '/')
            if "ple" in path_lower or "fli" in filename.lower():
                source = "pleiades"
                weight = WEIGHT_PLEIADES
            else:
                source = "lidar"
                weight = WEIGHT_LIDAR

            peak_date = peaks_per_year.get(year, '99999999')
            phase = "accumulation" if date_str <= peak_date else "ablation"

            # Corrección de fechas SCE
            sat_date_str = date_str
            if source == "pleiades":
                if date_str == "20200219":   continue
                elif date_str == "20220511": sat_date_str = "20220510"
                elif date_str == "20230215": sat_date_str = "20230214"
            elif source == "lidar":
                if date_str == "20210608":   sat_date_str = "20210607"

            input_path = os.path.join(DIR_INPUTS, f"izas_{sat_date_str}.tif")

            # Leer ventana DEM alineada con el archivo SD
            with rasterio.open(fp) as src_target:
                try:
                    win = src_dem.window(*src_target.bounds)
                    win = win.round_offsets().round_lengths()
                    dem_crop      = src_dem.read(1, window=win, boundless=True,
                                                  fill_value=NODATA_VAL)
                    transform_crop = src_dem.window_transform(win)
                    h, w = dem_crop.shape
                except:
                    continue

                # Target: snow depth reprojected to DEM grid
                target_crop = np.zeros((h, w), dtype=np.float32)
                reproject(
                    source=rasterio.band(src_target, 1),
                    destination=target_crop,
                    src_transform=src_target.transform,
                    src_crs=src_target.crs,
                    dst_transform=transform_crop,
                    dst_crs=dem_profile['crs'],
                    resampling=Resampling.bilinear,
                )

            # SCE
            input_crop = np.zeros((h, w), dtype=np.float32)
            if os.path.exists(input_path):
                with rasterio.open(input_path) as src_input:
                    reproject(
                        source=rasterio.band(src_input, 1),
                        destination=input_crop,
                        src_transform=src_input.transform,
                        src_crs=src_input.crs,
                        dst_transform=transform_crop,
                        dst_crs=dem_profile['crs'],
                        resampling=Resampling.average,
                    )

            # Variables topograficas
            slope, northness, eastness = calculate_topography(dem_crop)
            tpi = calculate_tpi(dem_crop)

            # Sx crop: extraer la misma ventana del array pre-cargado
            # Hay que manejar el caso boundless (ventana fuera del extent del DEM)
            win_round = win.round_offsets().round_lengths()
            r0 = int(win_round.row_off)
            c0 = int(win_round.col_off)

            sx_crop = np.zeros((24, h, w), dtype=np.float32)
            # Interseccion entre la ventana y el extent del DEM
            rs = max(0, r0);        re = min(dem_h, r0 + h)
            cs = max(0, c0);        ce = min(dem_w, c0 + w)
            # Coordenadas destino dentro del crop
            drs = rs - r0;  dre = drs + (re - rs)
            dcs = cs - c0;  dce = dcs + (ce - cs)
            if re > rs and ce > cs:
                sx_crop[:, drs:dre, dcs:dce] = sx_full[:, rs:re, cs:ce]

            # Mapas de persistencia nival (3 ventanas: 15, 30, 60 dias)
            pers_full = compute_persistence_maps(
                date_str, sce_dict, PERSISTENCE_WINDOWS, dem_h, dem_w
            )                                                         # (3, H_dem, W_dem)
            # Recortar la misma ventana que el DEM
            pers_crop = np.zeros((len(PERSISTENCE_WINDOWS), h, w), dtype=np.float32)
            if re > rs and ce > cs:
                pers_crop[:, drs:dre, dcs:dce] = pers_full[:, rs:re, cs:ce]

            # Stack completo: 6 + 8 + 3 = 17 canales
            topo_stack = np.stack(
                [dem_crop, slope, northness, eastness, tpi, input_crop], axis=0
            )                                                         # (6, h, w)
            stack = np.concatenate(
                [topo_stack, sx_crop, pers_crop], axis=0
            )                                                         # (33, h, w)

            # Split temporal
            if year in [2020, 2021, 2022]:  temp_split = 'train'
            elif year == 2023:              temp_split = 'val'
            elif year >= 2024:              temp_split = 'test'
            else:                           temp_split = 'ignore'

            # Tiling
            for y_off in range(0, h - TILE_SIZE + 1, STRIDE):
                for x_off in range(0, w - TILE_SIZE + 1, STRIDE):
                    tile_mask = target_crop[y_off:y_off+TILE_SIZE, x_off:x_off+TILE_SIZE]
                    dem_tile  = stack[0, y_off:y_off+TILE_SIZE, x_off:x_off+TILE_SIZE]

                    if np.mean(dem_tile == NODATA_VAL) > 0.1: continue
                    if np.min(tile_mask)  < -100:              continue

                    spatial_mod = (y_off + x_off) % 10
                    if spatial_mod == 0:   spat_split = 'test'
                    elif spatial_mod == 1: spat_split = 'val'
                    else:                  spat_split = 'train'

                    tile_name = f"{date_str}_{source}_tile_{y_off}_{x_off}.npy"
                    np.save(os.path.join(OUTPUT_DIR, "images", tile_name),
                            stack[:, y_off:y_off+TILE_SIZE, x_off:x_off+TILE_SIZE])
                    np.save(os.path.join(OUTPUT_DIR, "masks",  tile_name), tile_mask)

                    metadata_list.append({
                        'tile_id':          tile_name,
                        'date':             date_str,
                        'year':             year,
                        'phase':            phase,
                        'source':           source,
                        'weight':           weight,
                        'row_idx':          y_off,
                        'col_idx':          x_off,
                        'exp_temporal_split': temp_split,
                        'exp_spatial_split':  spat_split,
                    })

    # 4. Guardar CSV
    df = pd.DataFrame(metadata_list)
    csv_path = os.path.join(OUTPUT_DIR, "dataset_v6_fisico.csv")
    df.to_csv(csv_path, index=False)

    print(f"\nDataset v6 generado: {len(df)} tiles")
    print(f"CSV: {csv_path}")
    print(f"Canales por tile: 17 (6 topo+SCE + 8 Sx_100m + 3 persistencia nival)")
    print("\nDesglose por fuente:")
    print(df['source'].value_counts())
    print("\nDesglose temporal:")
    print(df['exp_temporal_split'].value_counts())

    return df


if __name__ == '__main__':
    df = process_dataset()
