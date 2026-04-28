"""
Genera el CSV de metadatos para dataset_v4_fisico (tiles LiDAR a 1m).
========================================================================

Canales (6):
    [0] DEM        - Elevacion (metros)
    [1] Slope      - Pendiente (grados)
    [2] Northness  - cos(aspect)
    [3] Eastness   - sin(aspect)
    [4] TPI        - Topographic Position Index
    [5] SCE        - Snow Cover Extent (codigos 0/10/11 -> binario)

Split temporal:
    Train : 2021-2022
    Val   : 2023
    Test  : 2024-2025

Filtro: se descartan tiles con < MIN_VALID_FRAC de pixeles validos en la mascara.

Salida:
    dataset_v4_fisico/dataset_v4_fisico.csv
"""

import numpy as np
import pandas as pd
from pathlib import Path

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
ROOT          = Path("E:/PycharmProjects/SnowPrediction/dataset_v4_fisico")
MIN_VALID_FRAC = 0.30   # Descartar tiles con < 30% de pixeles validos

TRAIN_YEARS = {"2021", "2022"}
VAL_YEARS   = {"2023"}
TEST_YEARS  = {"2024", "2025"}

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
images_dir = ROOT / "images"
masks_dir  = ROOT / "masks"

images = sorted(images_dir.glob("*lidar*.npy"))
print(f"Total tiles LiDAR encontrados: {len(images)}")

records = []
skipped = 0

for img_path in images:
    tile_id = img_path.name
    mask_path = masks_dir / tile_id

    if not mask_path.exists():
        print(f"  AVISO: mascara no encontrada para {tile_id}")
        skipped += 1
        continue

    # Cobertura de la mascara
    mask = np.load(mask_path)
    valid_frac = (~np.isnan(mask)).mean()

    if valid_frac < MIN_VALID_FRAC:
        skipped += 1
        continue

    # Extraer fecha y anyo
    date_str = tile_id.split("_lidar_")[0]   # e.g. "20211216"
    year_str = date_str[:4]

    if year_str in TRAIN_YEARS:
        split = "train"
    elif year_str in VAL_YEARS:
        split = "val"
    elif year_str in TEST_YEARS:
        split = "test"
    else:
        skipped += 1
        continue

    records.append({
        "tile_id":            tile_id,
        "date":               date_str,
        "year":               int(year_str),
        "source":             "lidar",
        "valid_frac":         round(float(valid_frac), 4),
        "exp_temporal_split": split,
    })

df = pd.DataFrame(records)
print(f"\nTiles incluidos : {len(df)}")
print(f"Tiles descartados: {skipped}")
print(f"\nDistribucion por split:")
for split in ["train", "val", "test"]:
    sub = df[df["exp_temporal_split"] == split]
    years = sorted(sub["year"].unique().tolist())
    print(f"  {split:5s}: {len(sub):>5} tiles  ({years})")

out = ROOT / "dataset_v4_fisico.csv"
df.to_csv(out, index=False)
print(f"\nCSV guardado en: {out}")
