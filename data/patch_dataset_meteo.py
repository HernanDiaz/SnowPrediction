"""
Parchea dataset_v4_ms_sx200 añadiendo 4 canales meteorológicos escalares.

Canales nuevos (22 -> 26):
  [22] t2m_7d      -- Temperatura media 7 dias anteriores (degC)
  [23] t2m_15d     -- Temperatura media 15 dias anteriores (degC)
  [24] t2m_30d     -- Temperatura media 30 dias anteriores (degC)
  [25] ppAcc_from_oct -- Precipitacion acumulada desde 1 Oct (mm)

Cada canal es escalar por fecha (mismo valor en todos los pixeles del tile).

Uso:
    .venv\\Scripts\\python.exe data/patch_dataset_meteo.py
"""

import numpy as np
import pandas as pd
import shutil
from pathlib import Path
from tqdm import tqdm

_REPO = Path(__file__).resolve().parent.parent

SRC_DIR  = _REPO / "dataset_v4_ms_sx200"
DST_DIR  = _REPO / "dataset_v4_ms_sx200_meteo"
TEMP_CSV = _REPO / "datos jesus/correo2/t2m_datesUAV.csv"
PP_CSV   = _REPO / "datos jesus/correo2/ppAcc.csv"


def build_meteo_lookup(all_tile_dates: set):
    """
    Construye lookup: date_str (YYYYMMDD) -> (t2m_7d, t2m_15d, t2m_30d, ppAcc_from_oct)

    Para fechas con tiles pero sin temperatura disponible, interpola linealmente
    entre las dos fechas mas proximas con datos.
    """
    t = pd.read_csv(TEMP_CSV)
    t['dateUAV'] = pd.to_datetime(t['dateUAV'])
    t = t.sort_values('dateUAV').set_index('dateUAV')

    pp = pd.read_csv(PP_CSV)
    pp['date'] = pd.to_datetime(pp['date'])
    pp = pp.sort_values('date').set_index('date')

    def ppAcc_for(dt_ts):
        if dt_ts.month >= 10:
            oct1 = pd.Timestamp(dt_ts.year, 10, 1)
        else:
            oct1 = pd.Timestamp(dt_ts.year - 1, 10, 1)
        return float(pp.loc[oct1:dt_ts, 'pp_mm'].sum())

    # Fechas con temperatura disponible
    lookup = {}
    for dt_ts, row in t.iterrows():
        date_str = dt_ts.strftime('%Y%m%d')
        lookup[date_str] = (
            float(row['t2m_7d']),
            float(row['t2m_15d']),
            float(row['t2m_30d']),
            ppAcc_for(dt_ts),
        )

    # Interpolar para fechas de tiles sin temperatura
    t_dates = sorted(t.index)
    for date_str in sorted(all_tile_dates):
        if date_str in lookup:
            continue
        dt = pd.to_datetime(date_str, format='%Y%m%d')
        # Encontrar fechas anterior y posterior con datos
        prev = [d for d in t_dates if d <= dt]
        nxt  = [d for d in t_dates if d >= dt]
        if not prev or not nxt:
            print(f"  [WARN] {date_str}: sin datos de temperatura vecinos, se omite")
            continue
        d0, d1 = prev[-1], nxt[0]
        if d0 == d1:
            alpha = 0.0
        else:
            alpha = (dt - d0).days / (d1 - d0).days
        r0, r1 = t.loc[d0], t.loc[d1]
        t7  = float(r0['t2m_7d']  + alpha * (r1['t2m_7d']  - r0['t2m_7d']))
        t15 = float(r0['t2m_15d'] + alpha * (r1['t2m_15d'] - r0['t2m_15d']))
        t30 = float(r0['t2m_30d'] + alpha * (r1['t2m_30d'] - r0['t2m_30d']))
        lookup[date_str] = (t7, t15, t30, ppAcc_for(dt))
        print(f"  [INTERP] {date_str}: t7={t7:.2f} t15={t15:.2f} t30={t30:.2f} ppAcc={lookup[date_str][3]:.1f}mm")

    return lookup


def main():
    # Obtener todas las fechas de tiles para interpolar las que falten
    src_images = SRC_DIR / "images"
    all_tile_dates = set(p.name.split('_')[0] for p in src_images.glob("*.npy"))

    print("Construyendo lookup meteorologico...")
    meteo = build_meteo_lookup(all_tile_dates)
    print(f"  Fechas con datos meteo: {len(meteo)}")

    # Crear directorios destino
    dst_images = DST_DIR / "images"
    dst_masks  = DST_DIR / "masks"
    dst_images.mkdir(parents=True, exist_ok=True)
    dst_masks.mkdir(parents=True, exist_ok=True)

    # Copiar CSV (mismo split)
    src_csv = SRC_DIR / "dataset_v4_ms_sx200.csv"
    dst_csv = DST_DIR / "dataset_v4_ms_sx200_meteo.csv"
    shutil.copy(src_csv, dst_csv)
    print(f"CSV copiado a {dst_csv}")

    # Parchear tiles
    tiles = sorted(src_images.glob("*.npy"))
    print(f"Tiles a parchear: {len(tiles)}")

    skipped = 0
    for tile_path in tqdm(tiles, desc="Parcheando tiles"):
        # Extraer fecha del nombre: YYYYMMDD_lidar_tile_Y_X.npy
        date_str = tile_path.name.split('_')[0]

        if date_str not in meteo:
            skipped += 1
            continue

        t7, t15, t30, ppAcc = meteo[date_str]

        # Cargar imagen original (22, H, W)
        img = np.load(tile_path)
        H, W = img.shape[1], img.shape[2]

        # Crear 4 canales constantes (1, H, W)
        ch_t7    = np.full((1, H, W), t7,    dtype=np.float32)
        ch_t15   = np.full((1, H, W), t15,   dtype=np.float32)
        ch_t30   = np.full((1, H, W), t30,   dtype=np.float32)
        ch_ppAcc = np.full((1, H, W), ppAcc, dtype=np.float32)

        # Stack: (26, H, W)
        img_new = np.concatenate([img, ch_t7, ch_t15, ch_t30, ch_ppAcc], axis=0)

        # Guardar
        np.save(dst_images / tile_path.name, img_new)

        # Copiar mascara (identica)
        mask_src = SRC_DIR / "masks" / tile_path.name
        mask_dst = DST_DIR / "masks" / tile_path.name
        if mask_src.exists() and not mask_dst.exists():
            shutil.copy(mask_src, mask_dst)

    print(f"\nCompletado: {len(tiles) - skipped} tiles parcheados, {skipped} omitidos")
    print(f"Dataset guardado en: {DST_DIR}")
    print(f"Canales: 22 originales + 4 meteo = 26 total")


if __name__ == '__main__':
    main()
