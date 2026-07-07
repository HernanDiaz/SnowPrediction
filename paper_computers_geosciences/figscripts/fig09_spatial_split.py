"""
Figure - Spatial train/val/test split of the Izas tiles.

Each marker is a unique tile location (parsed from the tile_id); colour denotes
the split. Tiles are 256 px (= 256 m) with a 128 px stride, so contiguous
column bands are used with a >=256 m buffer (grey) that is dropped, guaranteeing
no pixel is shared across splits. Axes in kilometres.

Reads dataset_v4_ms_sx200/dataset_v4_ms_sx200_spatial.csv.
Output: paper_computers_geosciences/figures/fig09_spatial_split.pdf (+ .png)
"""

from pathlib import Path
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

_REPO = Path(__file__).resolve().parents[2]
CSV = _REPO / "dataset_v4_ms_sx200" / "dataset_v4_ms_sx200_spatial.csv"
OUT = _REPO / "paper_computers_geosciences" / "figures"
OUT.mkdir(parents=True, exist_ok=True)

TILE = 256
B1, B2 = 768, 1280          # band boundaries (px = m)
COL = {"train": "#3274a1", "val": "#e1812c", "test": "#c03d3e", "buffer": "#cfcfcf"}

plt.rcParams.update({"font.size": 9, "axes.titlesize": 10})


def main():
    df = pd.read_csv(CSV)
    rc = df["tile_id"].map(lambda t: re.match(r"\d+_lidar_tile_(\d+)_(\d+)", t).groups())
    df["row"] = [int(r) for r, _ in rc]
    df["col"] = [int(c) for _, c in rc]
    locs = df.groupby(["row", "col"])["exp_spatial_split"].first().reset_index()
    counts = df["exp_spatial_split"].value_counts()

    # tile-centre coordinates in km (1 px = 1 m)
    cx = (locs["col"] + TILE / 2) / 1000.0
    cy = (locs["row"] + TILE / 2) / 1000.0

    fig, ax = plt.subplots(figsize=(4.2, 4.0))
    order = ["train", "val", "test", "buffer"]
    label = {"train": "Train", "val": "Validation", "test": "Test",
             "buffer": "Buffer (dropped)"}
    for s in order:
        m = locs["exp_spatial_split"] == s
        ax.scatter(cx[m], cy[m], s=34, marker="s", c=COL[s],
                   edgecolors="white", linewidths=0.3,
                   label=f"{label[s]} ({int(counts.get(s, 0))})")
    for b in (B1, B2):
        ax.axvline(b / 1000.0, color="0.35", ls="--", lw=0.8)

    ax.set_xlabel("Easting (km)")
    ax.set_ylabel("Northing (km)")
    ax.invert_yaxis()                 # row increases southward
    ax.set_aspect("equal")
    # legend OUTSIDE the map (below), 2 columns, so it never overlaps the tiles
    ax.legend(frameon=False, fontsize=7.5, ncol=2,
              loc="upper center", bbox_to_anchor=(0.5, -0.16),
              handletextpad=0.3, columnspacing=1.2)
    fig.tight_layout()
    fig.savefig(OUT / "fig09_spatial_split.pdf", bbox_inches="tight")
    fig.savefig(OUT / "fig09_spatial_split.png", bbox_inches="tight", dpi=150)
    print("Saved:", OUT / "fig09_spatial_split.pdf")


if __name__ == "__main__":
    main()
