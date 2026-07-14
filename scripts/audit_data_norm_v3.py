"""Auditoria a posteriori de datos + normalizacion v3.

Revisa sobre una muestra amplia de tiles de los tres splits temporales:
  1. TARGETS (masks): distribucion, negativos residuales (entre -100 y 0,
     que _clean NO limpia), extremos, fraccion nodata.
  2. CANALES CRUDOS: fraccion de artefactos por canal (segun umbrales v3),
     rangos, canales constantes/muertos, valores de SCE y persistencia.
  3. SALIDA v3 NORMALIZADA: sin NaN/Inf, rangos finales por canal.

Uso:
    .venv\\Scripts\\python.exe scripts/audit_data_norm_v3.py [--n 300]
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from data.dataset import SnowDataset  # noqa: E402

CSV = os.path.join(ROOT, 'dataset_v4_ms_sx200', 'dataset_v4_ms_sx200.csv')
IMG = os.path.join(ROOT, 'dataset_v4_ms_sx200', 'images')
MSK = os.path.join(ROOT, 'dataset_v4_ms_sx200', 'masks')

CH_NAMES = ['DEM_1m', 'Slope_1m', 'North_1m', 'East_1m', 'TPI_1m', 'SCE',
            'Sx_0', 'Sx_45', 'Sx_90', 'Sx_135', 'Sx_180', 'Sx_225',
            'Sx_270', 'Sx_315', 'Pers_15d', 'Pers_30d', 'Pers_60d',
            'DEM_5m', 'Slope_5m', 'North_5m', 'East_5m', 'TPI_5m']


def sample_tiles(df, split, n, rng):
    sub = df[df['exp_temporal_split'] == split]['tile_id'].tolist()
    if len(sub) > n:
        sub = list(rng.choice(sub, n, replace=False))
    return sub


def audit_masks(tiles, split):
    neg_soft = 0        # pixeles en (-100, -0.001): sobreviven a _clean
    nodata = 0
    total = 0
    zeros = 0
    snow_low = 0        # (0, 0.01]: excluidos de la evaluacion
    vmax = -np.inf
    huge = 0            # > 10 m: sospechosos
    for t in tiles:
        m = np.load(os.path.join(MSK, t)).astype(np.float64)
        total += m.size
        nodata += (m <= -100).sum()
        neg_soft += ((m > -100) & (m < -0.001)).sum()
        zeros += (np.abs(m) <= 0.001).sum()
        snow_low += ((m > 0.001) & (m <= 0.01)).sum()
        valid = m[m > -100]
        if valid.size:
            vmax = max(vmax, valid.max())
            huge += (valid > 10).sum()
    print(f'  [{split}] px={total:,}  nodata={100*nodata/total:.1f}%  '
          f'ceros={100*zeros/total:.1f}%  negativos_residuales='
          f'{100*neg_soft/total:.3f}% ({neg_soft:,})  '
          f'(0-1cm]={100*snow_low/total:.2f}%  max={vmax:.2f}m  '
          f'>10m={huge:,}')


def audit_raw_channels(tiles, split):
    n_art = {i: 0 for i in range(22)}
    n_tot = {i: 0 for i in range(22)}
    mins = {i: np.inf for i in range(22)}
    maxs = {i: -np.inf for i in range(22)}
    sce_vals = set()
    for t in tiles:
        a = np.load(os.path.join(IMG, t), mmap_mode='r')
        for i in range(22):
            ch = np.asarray(a[i], dtype=np.float64)
            ch = ch[ch != -9999]          # sentinel exacto ya tratado
            if ch.size == 0:
                continue
            n_tot[i] += ch.size
            mins[i] = min(mins[i], ch.min())
            maxs[i] = max(maxs[i], ch.max())
            if i in (0, 17):
                n_art[i] += (ch < 1700).sum()
            elif i == 4:
                n_art[i] += (np.abs(ch) > 50).sum()
            elif i == 21:
                n_art[i] += (np.abs(ch) > 100).sum()
            elif i in (2, 3, 19, 20):     # north/east: fuera de [-1,1]
                n_art[i] += (np.abs(ch) > 1.001).sum()
            elif i in (1, 18):            # slope fuera de [0,90]
                n_art[i] += ((ch < 0) | (ch > 90)).sum()
            elif 6 <= i <= 13:            # Sx fuera de [-90,90]
                n_art[i] += (np.abs(ch) > 90).sum()
            elif 14 <= i <= 16:           # persistencia fuera de [0,1]
                n_art[i] += ((ch < 0) | (ch > 1)).sum()
            if i == 5:
                sce_vals.update(np.unique(ch).tolist())
    print(f'  [{split}] artefactos por canal (% de px, umbrales v3/rango fisico):')
    for i in range(22):
        if n_tot[i] == 0:
            print(f'    {CH_NAMES[i]:9s}: SIN DATOS')
            continue
        pct = 100 * n_art[i] / n_tot[i]
        flag = '  <-- ATENCION' if pct > 0.01 else ''
        if pct > 0 or flag:
            print(f'    {CH_NAMES[i]:9s}: {pct:8.4f}%  '
                  f'rango=[{mins[i]:.2f}, {maxs[i]:.2f}]{flag}')
    print(f'    SCE valores unicos (muestra): {sorted(sce_vals)[:12]}')


def audit_v3_output(tiles, split):
    df = pd.DataFrame({'tile_id': tiles})
    ds = SnowDataset(df, IMG, MSK, n_channels=22)
    ds.norm_version = 'v3'
    bad = 0
    gmin = {i: np.inf for i in range(22)}
    gmax = {i: -np.inf for i in range(22)}
    for k in range(len(ds)):
        img, msk = ds[k]
        img = img.numpy()
        if not np.isfinite(img).all():
            bad += 1
            continue
        for i in range(22):
            gmin[i] = min(gmin[i], float(img[i].min()))
            gmax[i] = max(gmax[i], float(img[i].max()))
    print(f'  [{split}] tiles con NaN/Inf tras v3: {bad}')
    out_of_range = [(CH_NAMES[i], gmin[i], gmax[i]) for i in range(22)
                    if gmin[i] < -3.001 or gmax[i] > 3.001]
    if out_of_range:
        print(f'  [{split}] CANALES FUERA DE [-3,3]: {out_of_range}')
    else:
        rng_str = ', '.join(f'{CH_NAMES[i]}[{gmin[i]:.2f},{gmax[i]:.2f}]'
                            for i in (0, 4, 17, 21))
        print(f'  [{split}] todos los canales en [-3,3]. Clave: {rng_str}')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n', type=int, default=300,
                    help='tiles por split a muestrear')
    args = ap.parse_args()

    rng = np.random.default_rng(42)
    df = pd.read_csv(CSV)

    splits = {}
    for split in ['train', 'val', 'test']:
        splits[split] = sample_tiles(df, split, args.n, rng)
        print(f'{split}: {len(splits[split])} tiles muestreados')

    print('\n=== 1. TARGETS (masks) ===')
    for split, tiles in splits.items():
        audit_masks(tiles, split)

    print('\n=== 2. CANALES CRUDOS ===')
    for split, tiles in splits.items():
        audit_raw_channels(tiles, split)

    print('\n=== 3. SALIDA NORMALIZADA v3 ===')
    for split, tiles in splits.items():
        audit_v3_output(tiles, split)


if __name__ == '__main__':
    main()
