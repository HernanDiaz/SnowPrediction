"""
Build a leakage-free SPATIAL train/val/test split for dataset_v4_ms_sx200.

Tiles are 256 px with a 128 px stride -> 50% overlap between neighbours, so a
random tile split would leak. Instead we split by contiguous column bands
(easting) and DROP the tiles that straddle a band boundary, leaving a >=256 px
buffer so no pixel is shared across splits. The same (row,col) location is
assigned to the same split for every date (the split is purely spatial; all
years are seen in every split), testing generalisation to unseen LOCATIONS.

Outputs (isolated, nothing else is touched):
  - dataset_v4_ms_sx200/dataset_v4_ms_sx200_spatial.csv   (copy + exp_spatial_split)
  - results/spatial_split/spatial_split_map.png            (visualisation)

Usage:
    .venv/Scripts/python.exe data/make_spatial_split.py
"""

from pathlib import Path
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

_REPO = Path(__file__).resolve().parent.parent
DS = _REPO / "dataset_v4_ms_sx200"
CSV_IN = DS / "dataset_v4_ms_sx200.csv"
CSV_OUT = DS / "dataset_v4_ms_sx200_spatial.csv"
OUT_DIR = _REPO / "results" / "spatial_split"
OUT_DIR.mkdir(parents=True, exist_ok=True)

TILE = 256          # tile size (px)
# column-band boundaries (px); tiles straddling them are dropped (buffer)
B1 = 768            # test  = cols fully left of B1
B2 = 1280           # val   = cols in [B1, B2);  train = cols >= B2


def parse_rc(tile_id):
    m = re.match(r"\d+_lidar_tile_(\d+)_(\d+)\.npy", tile_id)
    return int(m.group(1)), int(m.group(2))


def assign(col):
    if col <= B1 - TILE:
        return "test"
    if B1 <= col <= B2 - TILE:
        return "val"
    if col >= B2:
        return "train"
    return "buffer"          # straddles a boundary -> excluded


def main():
    df = pd.read_csv(CSV_IN)
    rc = df["tile_id"].map(parse_rc)
    df["row"] = [r for r, _ in rc]
    df["col"] = [c for _, c in rc]
    df["exp_spatial_split"] = df["col"].map(assign)

    counts = df["exp_spatial_split"].value_counts()
    print("Tile counts by spatial split:")
    print(counts.to_string())
    kept = df[df["exp_spatial_split"] != "buffer"]
    print(f"\nKept {len(kept)}/{len(df)} tiles "
          f"({100*len(kept)/len(df):.0f}%); {len(df)-len(kept)} dropped as buffer.")
    for s in ("train", "val", "test"):
        sub = df[df["exp_spatial_split"] == s]
        print(f"  {s:5s}: {len(sub):5d} tiles | "
              f"unique locations {sub.groupby(['row','col']).ngroups}")

    # save CSV (drop helper row/col columns; keep original + new split)
    out = df.drop(columns=["row", "col"])
    out.to_csv(CSV_OUT, index=False)
    print(f"\nSaved: {CSV_OUT}")

    # ---- visualisation: one marker per unique location, coloured by split ----
    colours = {"train": "#3274a1", "val": "#e1812c",
               "test": "#c03d3e", "buffer": "#cccccc"}
    locs = df.groupby(["row", "col"])["exp_spatial_split"].first().reset_index()
    fig, ax = plt.subplots(figsize=(6.0, 5.2))
    for s, c in colours.items():
        m = locs["exp_spatial_split"] == s
        ax.scatter(locs.loc[m, "col"], locs.loc[m, "row"], s=80, marker="s",
                   c=c, edgecolors="white", linewidths=0.4,
                   label=f"{s} ({(df['exp_spatial_split']==s).sum()})")
    for b in (B1, B2):
        ax.axvline(b, color="0.4", ls="--", lw=0.8)
    ax.set_xlabel("column offset (px, easting)")
    ax.set_ylabel("row offset (px, northing)")
    ax.set_title("Spatial split (tile top-left positions)")
    ax.invert_yaxis()
    ax.set_aspect("equal")
    ax.legend(frameon=False, loc="upper right", fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "spatial_split_map.png", dpi=150, bbox_inches="tight")
    print(f"Saved: {OUT_DIR / 'spatial_split_map.png'}")


if __name__ == "__main__":
    main()
