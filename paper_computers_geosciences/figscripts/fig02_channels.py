"""
Figure 2 - Input channel stack (22 predictors + target) for a single tile.

Shows one 256x256 tile of dataset_v4_ms_sx200 with all 22 input channels,
grouped by family (fine 1 m topography, coarse 5 m topography, 8 wind-shelter
Sx indices, satellite SCE, snow persistence) plus the LiDAR snow-depth target.

Output: paper_computers_geosciences/figures/fig02_channels.pdf (+ .png)
"""

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams.update({
    "font.size": 8,
    "axes.titlesize": 7.5,
})

_REPO = Path(__file__).resolve().parents[2]
DS = _REPO / "dataset_v4_ms_sx200"
OUT_DIR = _REPO / "paper_computers_geosciences/figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)

TILE = "20210202_lidar_tile_1024_640.npy"  # full-valid, rich HS + SCE structure

# Display order (row-major, 5 columns); each entry:
#   (channel_index_or_'HS', label, cmap, mode)
# mode: 'seq' sequential, 'div' diverging symmetric about 0, 'fixed' (vmin,vmax)
LAYOUT = [
    # row 1: fine 1 m topography
    (0,  "DEM (m)",          "terrain", "seq"),
    (1,  "Slope (\u00b0)",   "YlOrBr",  "seq"),
    (2,  "Northness",        "RdBu",    "fixed", (-1, 1)),
    (3,  "Eastness",         "RdBu",    "fixed", (-1, 1)),
    (4,  "TPI",              "RdBu_r",  "div"),
    # row 2: coarse 5 m topography
    (17, "DEM 5 m (m)",      "terrain", "seq"),
    (18, "Slope 5 m (\u00b0)", "YlOrBr", "seq"),
    (19, "Northness 5 m",    "RdBu",    "fixed", (-1, 1)),
    (20, "Eastness 5 m",     "RdBu",    "fixed", (-1, 1)),
    (21, "TPI 5 m",          "RdBu_r",  "div"),
    # row 3: wind-shelter Sx (first 5)
    (6,  "Sx 0\u00b0",       "RdBu_r",  "div"),
    (7,  "Sx 45\u00b0",      "RdBu_r",  "div"),
    (8,  "Sx 90\u00b0",      "RdBu_r",  "div"),
    (9,  "Sx 135\u00b0",     "RdBu_r",  "div"),
    (10, "Sx 180\u00b0",     "RdBu_r",  "div"),
    # row 4: Sx (last 3) + SCE
    (11, "Sx 225\u00b0",     "RdBu_r",  "div"),
    (12, "Sx 270\u00b0",     "RdBu_r",  "div"),
    (13, "Sx 315\u00b0",     "RdBu_r",  "div"),
    (5,  "SCE",              "Blues",   "seq"),
    None,
    # row 5: snow persistence + target
    (14, "Persistence 15 d", "Blues",   "seq"),
    (15, "Persistence 30 d", "Blues",   "seq"),
    (16, "Persistence 60 d", "Blues",   "seq"),
    ("HS", "Snow depth (m)", "viridis", "seq"),
    None,
]

NCOLS = 5


def main():
    img = np.load(DS / "images" / TILE)          # (22, 256, 256)
    msk = np.load(DS / "masks" / TILE)            # (256, 256)

    nrows = int(np.ceil(len(LAYOUT) / NCOLS))
    fig, axes = plt.subplots(nrows, NCOLS, figsize=(6.6, 6.9))
    axes = axes.ravel()

    for ax, item in zip(axes, LAYOUT):
        if item is None:
            ax.axis("off")
            continue
        idx, label, cmap, mode = item[0], item[1], item[2], item[3]
        extra = item[4] if len(item) > 4 else None
        data = msk if idx == "HS" else img[idx]

        if mode == "div":
            v = float(np.nanmax(np.abs(data)))
            vmin, vmax = -v, v
        elif mode == "fixed":
            vmin, vmax = extra
        else:  # seq
            vmin = float(np.nanpercentile(data, 1))
            vmax = float(np.nanpercentile(data, 99))

        im = ax.imshow(data, cmap=cmap, vmin=vmin, vmax=vmax, origin="upper")
        ax.set_title(label, pad=2)
        ax.set_xticks([])
        ax.set_yticks([])
        cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
        cb.ax.tick_params(labelsize=5, length=2, pad=1)
        cb.outline.set_linewidth(0.4)

    fig.subplots_adjust(left=0.01, right=0.99, top=0.97, bottom=0.01,
                        wspace=0.30, hspace=0.22)
    fig.savefig(OUT_DIR / "fig02_channels.pdf", bbox_inches="tight", dpi=300)
    fig.savefig(OUT_DIR / "fig02_channels.png", bbox_inches="tight", dpi=150)
    print("Saved:", OUT_DIR / "fig02_channels.pdf")


if __name__ == "__main__":
    main()
