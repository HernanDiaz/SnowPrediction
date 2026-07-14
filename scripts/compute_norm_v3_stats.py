"""Calcula estadisticas empiricas de los canales sobre el TRAIN set para
derivar las constantes de la normalizacion v3.

Motivacion: la norm "nueva" usa constantes arbitrarias que destruyen canales:
  - tpi_max=9200 (calibrada sobre artefactos nodata; el TPI real es ~[-3,3] m
    a 1m y ~[-15,15] m a 5m) -> canal aplastado a ~0.
  - dem_std=1000 (la std real de la cuenca es ~decenas de m) -> canal casi
    plano tras normalizar.

Este script recorre los tiles de train (split temporal) y calcula media, std
y percentiles de cada canal usando solo pixeles validos (> -9000). Los canales
se leen con mmap para no cargar los arrays completos.

Salida: results/norm_v3/norm_v3_stats.json + resumen por pantalla.

Uso:
    .venv\\Scripts\\python.exe scripts/compute_norm_v3_stats.py
"""
import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
IMAGES_DIR = os.path.join(ROOT, 'dataset_v4_ms_sx200', 'images')
OUT_DIR = os.path.join(ROOT, 'results', 'norm_v3')

# (csv, columna de split) por tipo de split
SPLIT_CFG = {
    'temporal': ('dataset_v4_ms_sx200.csv', 'exp_temporal_split'),
    'spatial':  ('dataset_v4_ms_sx200_spatial.csv', 'exp_spatial_split'),
}

CHANNELS = {
    0: 'DEM_1m',
    1: 'Slope_1m',
    2: 'Northness_1m',
    3: 'Eastness_1m',
    4: 'TPI_1m',
    5: 'SCE',
    6: 'Sx_0',          # representante del grupo Sx (6-13, misma naturaleza)
    14: 'Pers_15d',
    17: 'DEM_5m',
    18: 'Slope_5m',
    19: 'Northness_5m',
    20: 'Eastness_5m',
    21: 'TPI_5m',
}

PCTS = [0.1, 0.5, 1, 5, 25, 50, 75, 95, 99, 99.5, 99.9]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--split', choices=list(SPLIT_CFG), default='temporal',
                    help="split cuyo 'train' se usa para las estadisticas")
    args = ap.parse_args()
    csv_name, split_col = SPLIT_CFG[args.split]
    out_json = os.path.join(OUT_DIR, f'norm_v3_stats_{args.split}.json')

    df = pd.read_csv(os.path.join(ROOT, 'dataset_v4_ms_sx200', csv_name))
    train = df[df[split_col] == 'train']
    print(f"Split '{args.split}' | tiles de train: {len(train)}")

    # Acumuladores online (media/std via sumas) + muestra para percentiles.
    # Con ~4241 tiles x 65536 px por canal, guardar todo en memoria son
    # ~1.1 GB por canal en float64; muestreamos 4096 px validos por tile
    # (suficiente para percentiles estables) y usamos sumas exactas para
    # media/std.
    rng = np.random.default_rng(42)
    n_sample_px = 4096
    acc = {c: {'n': 0, 's': 0.0, 's2': 0.0, 'min': np.inf, 'max': -np.inf,
               'samples': []} for c in CHANNELS}

    for i, tile in enumerate(train['tile_id']):
        path = os.path.join(IMAGES_DIR, tile)
        arr = np.load(path, mmap_mode='r')
        for c in CHANNELS:
            ch = np.asarray(arr[c], dtype=np.float64)
            valid = ch[(ch > -9000) & np.isfinite(ch)]
            if valid.size == 0:
                continue
            a = acc[c]
            a['n'] += valid.size
            a['s'] += valid.sum()
            a['s2'] += (valid ** 2).sum()
            a['min'] = min(a['min'], float(valid.min()))
            a['max'] = max(a['max'], float(valid.max()))
            if valid.size > n_sample_px:
                a['samples'].append(rng.choice(valid, n_sample_px, replace=False))
            else:
                a['samples'].append(valid)
        if (i + 1) % 500 == 0:
            print(f'  {i + 1}/{len(train)} tiles...')

    stats = {}
    print(f"\n{'Canal':<14}{'media':>10}{'std':>9}{'min':>10}{'max':>10}"
          f"{'p0.5':>9}{'p50':>9}{'p99.5':>9}")
    for c, name in CHANNELS.items():
        a = acc[c]
        mean = a['s'] / a['n']
        std = float(np.sqrt(a['s2'] / a['n'] - mean ** 2))
        sample = np.concatenate(a['samples'])
        pct = {f'p{p}': float(np.percentile(sample, p)) for p in PCTS}
        stats[name] = {
            'channel_index': c, 'n_valid_px': int(a['n']),
            'mean': float(mean), 'std': std,
            'min': a['min'], 'max': a['max'], **pct,
        }
        print(f"{name:<14}{mean:>10.3f}{std:>9.3f}{a['min']:>10.2f}"
              f"{a['max']:>10.2f}{pct['p0.5']:>9.3f}{pct['p50']:>9.3f}"
              f"{pct['p99.5']:>9.3f}")

    os.makedirs(OUT_DIR, exist_ok=True)
    with open(out_json, 'w') as f:
        json.dump(stats, f, indent=2)
    print(f'\nGuardado: {out_json}')


if __name__ == '__main__':
    sys.exit(main())
