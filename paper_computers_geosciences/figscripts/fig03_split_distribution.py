"""
Figure 3 - Temporal split and snow-depth distribution by year.

(a) Per-date mean snow depth (+/- IQR) along the acquisition timeline, coloured
    by the train / validation / test split, showing the seasonal cycle and the
    anomalously deep snow of the 2025 (test) year.
(b) Violin plots of the snow-depth distribution per year, with the train mean as
    reference, highlighting that 2025 is markedly snowier.

Data: original LiDAR snow-depth maps (Articulo 1/Data/.../SnowDepth/SD_*_1m.tif),
one map per acquisition date (avoids tile-overlap double counting).

Output: paper_computers_geosciences/figures/fig03_split_distribution.pdf (+ .png)
"""

from pathlib import Path
import re
import glob
import numpy as np
import rasterio
import matplotlib.pyplot as plt
from datetime import datetime
from matplotlib.patches import Patch

plt.rcParams.update({"font.size": 9, "axes.titlesize": 9})

_REPO = Path(__file__).resolve().parents[2]
SD_DIR = _REPO / "Articulo 1/Data/izas/LiDAR/SnowDepth"
OUT_DIR = _REPO / "paper_computers_geosciences/figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Split definition (by year)
SPLIT = {2021: "train", 2022: "train", 2023: "train",
         2024: "val", 2025: "test"}
COLORS = {"train": "#3274a1", "val": "#e1812c", "test": "#c03d3e"}
HS_CAP = 5.0          # clip upper tail (artefacts up to ~20 m) for display
SUBSAMPLE = 60000     # max pixels per year for the violins


def load_valid(path):
    with rasterio.open(path) as s:
        a = s.read(1).astype(np.float32)
    a = a[np.isfinite(a)]
    a = a[a > -0.5]                 # drop nodata-ish
    a = np.clip(a, 0.0, None)       # negatives -> 0 (LiDAR noise)
    return a


def main():
    files = sorted(glob.glob(str(SD_DIR / "SD_*_1m.tif")))
    dates, meds, q1, q3, splits, byyear = [], [], [], [], [], {}
    for p in files:
        ymd = re.search(r"SD_(\d{8})", p).group(1)
        dt = datetime.strptime(ymd, "%Y%m%d")
        yr = dt.year
        a = load_valid(p)
        ac = np.clip(a, 0, HS_CAP)
        dates.append(dt)
        meds.append(float(np.median(a)))
        q1.append(float(np.percentile(a, 25)))
        q3.append(float(np.percentile(a, 75)))
        splits.append(SPLIT[yr])
        byyear.setdefault(yr, []).append(ac)

    # Per-year pooled samples for violins
    years = sorted(byyear)
    year_data, year_split, year_mean = [], [], []
    rng = np.random.default_rng(0)
    for yr in years:
        pooled = np.concatenate(byyear[yr])
        if pooled.size > SUBSAMPLE:
            pooled = rng.choice(pooled, SUBSAMPLE, replace=False)
        year_data.append(pooled)
        year_split.append(SPLIT[yr])
        year_mean.append(float(np.mean(np.concatenate(byyear[yr]))))

    train_mean = float(np.mean(np.concatenate(
        [np.concatenate(byyear[y]) for y in years if SPLIT[y] == "train"])))
    test_mean = float(np.mean(np.concatenate(byyear[2025])))
    print(f"train mean HS = {train_mean:.3f} m, 2025 mean HS = {test_mean:.3f} m, "
          f"+{100*(test_mean/train_mean-1):.0f}%")
    for yr, mn in zip(years, year_mean):
        print(f"  {yr} ({SPLIT[yr]}): mean {mn:.3f} m, n_dates {len(byyear[yr])}")

    fig, (axa, axb) = plt.subplots(1, 2, figsize=(6.6, 3.0),
                                   gridspec_kw={"width_ratios": [1.45, 1.0],
                                                "wspace": 0.30})

    # ---- (a) timeline ---------------------------------------------------
    for dt, md, lo, hi, sp in zip(dates, meds, q1, q3, splits):
        axa.errorbar(dt, md, yerr=[[md - lo], [hi - md]], fmt="o", ms=4,
                     color=COLORS[sp], ecolor=COLORS[sp], elinewidth=1,
                     capsize=2, alpha=0.9)
    axa.set_ylabel("Snow depth (m)")
    axa.set_xlabel("Acquisition date")
    axa.grid(True, alpha=0.25, lw=0.5)
    for lab in axa.get_xticklabels():
        lab.set_rotation(35)
        lab.set_ha("right")
    handles = [Patch(color=COLORS[s], label=s) for s in ["train", "val", "test"]]
    axa.legend(handles=handles, fontsize=7, frameon=False, loc="upper left")

    # ---- (b) violins by year -------------------------------------------
    parts = axb.violinplot(year_data, positions=range(len(years)),
                           showmeans=False, showextrema=False, widths=0.8)
    for body, sp in zip(parts["bodies"], year_split):
        body.set_facecolor(COLORS[sp])
        body.set_edgecolor("#333333")
        body.set_alpha(0.75)
        body.set_linewidth(0.5)
    # mean markers
    axb.scatter(range(len(years)), year_mean, color="k", s=12, zorder=3)
    axb.axhline(train_mean, color="#3274a1", ls="--", lw=1,
                label=f"train mean ({train_mean:.2f} m)")
    axb.set_xticks(range(len(years)))
    axb.set_xticklabels([str(y) for y in years])
    axb.set_ylabel("Snow depth (m)")
    axb.set_xlabel("Year")
    axb.set_ylim(0, HS_CAP)
    axb.grid(True, axis="y", alpha=0.25, lw=0.5)
    axb.legend(fontsize=7, frameon=False, loc="upper left")

    fig.savefig(OUT_DIR / "fig03_split_distribution.pdf", bbox_inches="tight", dpi=300)
    fig.savefig(OUT_DIR / "fig03_split_distribution.png", bbox_inches="tight", dpi=150)
    print("Saved:", OUT_DIR / "fig03_split_distribution.pdf")


if __name__ == "__main__":
    main()
