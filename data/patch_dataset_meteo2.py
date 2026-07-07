"""
Parchea dataset_v4_ms_sx200_meteo (26ch) añadiendo 9 canales meteorológicos adicionales.

Canales nuevos (26 -> 35):
  [26] ws_7d       -- Velocidad media viento 7 dias anteriores (m/s)
  [27] ws_15d      -- Velocidad media viento 15 dias anteriores (m/s)
  [28] ws_30d      -- Velocidad media viento 30 dias anteriores (m/s)
  [29] rh_7d       -- Humedad relativa media 7 dias anteriores (%)
  [30] rh_15d      -- Humedad relativa media 15 dias anteriores (%)
  [31] rh_30d      -- Humedad relativa media 30 dias anteriores (%)
  [32] rad_7d      -- Radiacion solar media 7 dias anteriores (W/m2)
  [33] rad_15d     -- Radiacion solar media 15 dias anteriores (W/m2)
  [34] rad_30d     -- Radiacion solar media 30 dias anteriores (W/m2)

Uso:
    .venv\\Scripts\\python.exe data/patch_dataset_meteo2.py
"""

import csv
import shutil

import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm

_REPO   = Path(__file__).resolve().parent.parent
SRC_DIR = _REPO / "dataset_v4_ms_sx200_meteo"
DST_DIR = _REPO / "dataset_v4_ms_sx200_meteo2"
METEO3  = _REPO / "datos jesus/correo3/meteo_izas_daily.csv"

LIMITS = {
    'WS_ms':   (0,   50),
    'RH_perc': (0,  100),
    'Rad_Wm':  (0, 1200),
}


def load_meteo3() -> pd.DataFrame:
    """Lee meteo_izas_daily.csv (formato R con header mal entrecomillado),
    filtra outliers fisicos e interpola."""
    rows = []
    with open(METEO3, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        next(reader)  # skip malformed header
        for row in reader:
            if len(row) == 7:
                rows.append({
                    'day':     row[1],
                    'WS_ms':   float(row[3]),
                    'RH_perc': float(row[5]),
                    'Rad_Wm':  float(row[6]),
                })
    df = pd.DataFrame(rows)
    df['day'] = pd.to_datetime(df['day'])
    df = df.set_index('day').sort_index()

    # Filtrar outliers con limites fisicos -> NaN
    for col, (lo, hi) in LIMITS.items():
        bad = (df[col] < lo) | (df[col] > hi)
        if bad.sum() > 0:
            print(f"  {col}: {bad.sum()} outliers → NaN")
            df.loc[bad, col] = np.nan

    # Reindexar a frecuencia diaria para rellenar dias faltantes (mantenimiento sensor)
    full_idx = pd.date_range(df.index[0], df.index[-1], freq='D')
    n_missing = len(full_idx) - len(df)
    if n_missing > 0:
        print(f"  Dias faltantes en CSV: {n_missing} → interpolados")
    df = df.reindex(full_idx)

    # Interpolar outliers y dias faltantes conjuntamente
    for col in ['WS_ms', 'RH_perc', 'Rad_Wm']:
        df[col] = df[col].interpolate(method='time')

    return df


def rolling_mean(df: pd.DataFrame, col: str, days: int, target_date: pd.Timestamp) -> float:
    """Media de `col` en los `days` dias anteriores a `target_date` (inclusive)."""
    end   = target_date
    start = target_date - pd.Timedelta(days=days - 1)
    mask  = (df.index >= start) & (df.index <= end)
    vals  = df.loc[mask, col]
    return float(vals.mean()) if len(vals) > 0 else float('nan')


def build_lookup(meteo: pd.DataFrame, all_dates: set) -> dict:
    """Construye lookup: YYYYMMDD -> (ws7, ws15, ws30, rh7, rh15, rh30, rad7, rad15, rad30)"""
    lookup = {}
    for date_str in sorted(all_dates):
        dt = pd.to_datetime(date_str, format='%Y%m%d')
        lookup[date_str] = (
            rolling_mean(meteo, 'WS_ms',   7,  dt),
            rolling_mean(meteo, 'WS_ms',   15, dt),
            rolling_mean(meteo, 'WS_ms',   30, dt),
            rolling_mean(meteo, 'RH_perc', 7,  dt),
            rolling_mean(meteo, 'RH_perc', 15, dt),
            rolling_mean(meteo, 'RH_perc', 30, dt),
            rolling_mean(meteo, 'Rad_Wm',  7,  dt),
            rolling_mean(meteo, 'Rad_Wm',  15, dt),
            rolling_mean(meteo, 'Rad_Wm',  30, dt),
        )
    return lookup


def main():
    print("Cargando datos meteorologicos (correo3)...")
    meteo = load_meteo3()
    print(f"  Dias disponibles: {len(meteo)} | Rango: {meteo.index[0].date()} – {meteo.index[-1].date()}")

    src_images = SRC_DIR / "images"
    all_dates  = set(p.name.split('_')[0] for p in src_images.glob("*.npy"))
    print(f"  Fechas de tiles: {sorted(all_dates)}")

    print("Construyendo lookup de ventanas temporales...")
    lookup = build_lookup(meteo, all_dates)
    for d, v in sorted(lookup.items()):
        print(f"  {d}: ws7={v[0]:.2f} rh7={v[3]:.1f} rad7={v[6]:.1f}")

    # Crear directorios destino
    dst_images = DST_DIR / "images"
    dst_masks  = DST_DIR / "masks"
    dst_images.mkdir(parents=True, exist_ok=True)
    dst_masks.mkdir(parents=True, exist_ok=True)

    # Copiar CSV
    src_csv = SRC_DIR / "dataset_v4_ms_sx200_meteo.csv"
    dst_csv = DST_DIR / "dataset_v4_ms_sx200_meteo2.csv"
    shutil.copy(src_csv, dst_csv)
    print(f"CSV copiado a {dst_csv}")

    # Parchear tiles
    tiles   = sorted(src_images.glob("*.npy"))
    skipped = 0
    print(f"Tiles a parchear: {len(tiles)}")

    for tile_path in tqdm(tiles, desc="Parcheando tiles"):
        date_str = tile_path.name.split('_')[0]

        if date_str not in lookup:
            skipped += 1
            continue

        vals = lookup[date_str]
        if any(np.isnan(v) for v in vals):
            print(f"  [WARN] {date_str}: NaN en lookup, omitido")
            skipped += 1
            continue

        img  = np.load(tile_path)          # (26, H, W)
        H, W = img.shape[1], img.shape[2]

        new_chs = np.stack(
            [np.full((H, W), v, dtype=np.float32) for v in vals],
            axis=0
        )  # (9, H, W)

        img_new = np.concatenate([img, new_chs], axis=0)  # (35, H, W)
        np.save(dst_images / tile_path.name, img_new)

        mask_src = SRC_DIR / "masks" / tile_path.name
        mask_dst = DST_DIR / "masks"  / tile_path.name
        if mask_src.exists() and not mask_dst.exists():
            shutil.copy(mask_src, mask_dst)

    print(f"\nCompletado: {len(tiles) - skipped} tiles parcheados, {skipped} omitidos")
    print(f"Dataset guardado en: {DST_DIR}")
    print(f"Canales: 26 originales + 9 meteo = 35 total")


if __name__ == '__main__':
    main()
