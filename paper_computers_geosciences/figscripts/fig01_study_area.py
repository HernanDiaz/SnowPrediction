"""
Figure 1 - Study area (Izas experimental catchment, Central Spanish Pyrenees).

Panels:
  (a) Iberian Peninsula locator (schematic coastline) with a star at Izas.
  (b) Hillshade of the 1 m DEM (big extent) with the LiDAR catchment footprint
      outlined (derived from the valid-data mask of the snow-depth maps).
  (c) Example 1 m LiDAR snow-depth (HS) map showing wind-driven drift patterns.

Inputs (all already in the repo):
  Articulo 1/Data/izas/LiDAR/Topografia/DEMbigIzas_1m.tif   (EPSG:25830)
  Articulo 1/Data/izas/LiDAR/SnowDepth/SD_YYYYMMDD_1m.tif

Output:
  paper_computers_geosciences/figures/fig01_study_area.pdf (+ .png preview)

Notes:
  - No catchment shapefile / AWS coordinates / country basemap were available,
    so panel (a) is a schematic locator and the catchment outline in (b) is the
    measured LiDAR footprint, not a hydrological divide.
"""

from pathlib import Path
import numpy as np
import rasterio
from rasterio.warp import reproject, Resampling
import cv2
import matplotlib.pyplot as plt
from matplotlib.colors import LightSource
from matplotlib.patches import Polygon as MplPolygon
import matplotlib.ticker as mticker

# Designed close to the printed width (elsarticle preprint linewidth = 390 pt
# = 5.4 in) and embedded at \linewidth, so on-page text stays legible.
plt.rcParams.update({
    "font.size": 9,
    "axes.labelsize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
})


def _km(x, _pos):
    return f"{x/1000:g}"

try:
    import geopandas as gpd
    _NE = (Path(gpd.__file__).resolve().parents[1] / "pyogrio/tests/fixtures/"
           "naturalearth_lowres/naturalearth_lowres.shp")
    if not _NE.exists():
        _NE = None
except Exception:
    gpd, _NE = None, None

_REPO = Path(__file__).resolve().parents[2]
BASE = _REPO / "Articulo 1/Data/izas/LiDAR"
DEM_1M = BASE / "Topografia/DEMbigIzas_1m.tif"
SNOW_DIR = BASE / "SnowDepth"
HS_EXAMPLE = SNOW_DIR / "SD_20250401_1m.tif"
OUT_DIR = _REPO / "paper_computers_geosciences/figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Coarse Iberian Peninsula outline (lon, lat) for the schematic locator.
# Recognisable but intentionally low-resolution.
# ---------------------------------------------------------------------------
IBERIA = [
    (-8.9, 43.8), (-7.6, 43.7), (-5.8, 43.6), (-3.8, 43.5), (-1.8, 43.4),
    (0.7, 42.7), (3.3, 42.4), (1.0, 41.1), (0.9, 40.7), (-0.3, 39.5),
    (-0.2, 38.8), (-0.7, 37.9), (-1.9, 37.2), (-2.9, 36.7), (-4.4, 36.7),
    (-5.3, 36.1), (-6.3, 36.2), (-7.4, 37.2), (-8.8, 37.0), (-8.9, 38.5),
    (-9.5, 38.8), (-8.8, 40.2), (-8.8, 41.2), (-9.0, 43.0), (-8.9, 43.8),
]
IZAS_LON, IZAS_LAT = -0.4205, 42.7419


def load_hillshade(path):
    with rasterio.open(path) as src:
        dem = src.read(1).astype(float)
        nodata = src.nodata
        bounds = src.bounds
        tr = src.transform
    dem = np.where(dem == nodata, np.nan, dem)
    # Fill nodata with median for shading, keep mask to grey it out afterwards.
    filled = np.where(np.isfinite(dem), dem, np.nanmedian(dem))
    ls = LightSource(azdeg=315, altdeg=45)
    hs = ls.hillshade(filled, vert_exag=2.0, dx=1.0, dy=1.0)
    extent = (bounds.left, bounds.right, bounds.bottom, bounds.top)
    return hs, extent, dem, tr, bounds


def catchment_outline(dem_path, sd_paths):
    """Outer contour (in map coords) of the union LiDAR footprint across all
    snow-depth dates, resampled onto the DEM grid."""
    with rasterio.open(dem_path) as dem:
        H, W = dem.height, dem.width
        dst_tr = dem.transform
        dst_crs = dem.crs
    acc = np.zeros((H, W), np.uint8)
    for p in sd_paths:
        with rasterio.open(p) as src:
            arr = src.read(1).astype(float)
            nod = src.nodata
            src_tr = src.transform
        valid = np.isfinite(arr)
        if nod is not None and not np.isnan(nod):
            valid &= (arr != nod)
        dst = np.zeros((H, W), np.float32)
        reproject(valid.astype(np.float32), dst,
                  src_transform=src_tr, src_crs="EPSG:25830",
                  dst_transform=dst_tr, dst_crs=dst_crs,
                  resampling=Resampling.nearest)
        acc |= (dst > 0.5).astype(np.uint8)
    mask = acc * 255
    k = np.ones((15, 15), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((7, 7), np.uint8))
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnt = max(cnts, key=cv2.contourArea)
    eps = 0.0015 * cv2.arcLength(cnt, True)
    cnt = cv2.approxPolyDP(cnt, eps, True).reshape(-1, 2)
    xs, ys = rasterio.transform.xy(dst_tr, cnt[:, 1], cnt[:, 0])
    return np.column_stack([xs, ys])


def add_scalebar(ax, x0, y0, length=500, label="500 m"):
    ax.plot([x0, x0 + length], [y0, y0], color="k", lw=3, solid_capstyle="butt")
    ax.text(x0 + length / 2, y0 + length * 0.06, label, ha="center", va="bottom",
            fontsize=8, color="k")


def add_north(ax, x, y, size=180):
    ax.annotate("N", xy=(x, y + size), xytext=(x, y),
                arrowprops=dict(facecolor="k", width=3, headwidth=9),
                ha="center", va="center", fontsize=10, fontweight="bold")


def main():
    hs, dem_extent, _, _, dem_bounds = load_hillshade(DEM_1M)
    sd_paths = sorted(SNOW_DIR.glob("SD_*_1m.tif"))
    outline = catchment_outline(DEM_1M, sd_paths)

    with rasterio.open(HS_EXAMPLE) as src:
        hsmap = src.read(1).astype(float)
        nod = src.nodata
        hs_ext = (src.bounds.left, src.bounds.right, src.bounds.bottom, src.bounds.top)
    hsmap = np.where(np.isfinite(hsmap) & (hsmap != nod), hsmap, np.nan)

    fig = plt.figure(figsize=(6.4, 5.0))
    gs = fig.add_gridspec(2, 2, height_ratios=[0.8, 1.5],
                          hspace=0.40, wspace=0.30)

    # ---- (a) locator (top, centered) ------------------------------------
    axa = fig.add_subplot(gs[0, :])
    axa.set_facecolor("#dfeefc")  # sea
    if _NE is not None:
        world = gpd.read_file(_NE)
        land = world[world["name"].isin(["Spain", "Portugal", "France"])]
        land.plot(ax=axa, facecolor="#e7e2d6", edgecolor="#777777", lw=0.6,
                  zorder=1)
    else:
        axa.add_patch(MplPolygon(IBERIA, closed=True, facecolor="#e7e2d6",
                                 edgecolor="#555555", lw=0.8, zorder=1))
    axa.text(-3.5, 43.95, "Pyrenees", fontsize=8, style="italic", rotation=-4,
             color="#555555", ha="center", zorder=3)
    axa.plot(IZAS_LON, IZAS_LAT, marker="*", ms=14, color="#d7191c",
             mec="k", mew=0.6, zorder=4)
    axa.annotate("Izas", xy=(IZAS_LON, IZAS_LAT), xytext=(1.4, 41.3),
                 fontsize=9, fontweight="bold", color="#d7191c",
                 arrowprops=dict(arrowstyle="-", color="#d7191c", lw=0.8),
                 zorder=4)
    axa.set_xlim(-10.2, 4.2)
    axa.set_ylim(35.5, 45.0)
    axa.set_aspect(1.0 / np.cos(np.deg2rad(40)))
    axa.set_xlabel("Longitude (\u00b0)")
    axa.set_ylabel("Latitude (\u00b0)")

    # ---- (b) hillshade + footprint --------------------------------------
    axb = fig.add_subplot(gs[1, 0])
    axb.imshow(hs, cmap="gray", extent=dem_extent, origin="upper",
               vmin=0, vmax=1, zorder=1)
    axb.add_patch(MplPolygon(outline, closed=True, fill=False,
                             edgecolor="#d7191c", lw=1.8, zorder=3))
    axb.set_xlim(dem_bounds.left, dem_bounds.right)
    axb.set_ylim(dem_bounds.bottom, dem_bounds.top)
    axb.set_aspect("equal")
    add_scalebar(axb, dem_bounds.left + 250, dem_bounds.bottom + 250, 500, "500 m")
    add_north(axb, dem_bounds.right - 300, dem_bounds.bottom + 350)
    axb.xaxis.set_major_locator(mticker.MaxNLocator(4))
    axb.yaxis.set_major_locator(mticker.MaxNLocator(4))
    axb.xaxis.set_major_formatter(mticker.FuncFormatter(_km))
    axb.yaxis.set_major_formatter(mticker.FuncFormatter(_km))
    axb.set_xlabel("Easting (km, UTM 30N)")
    axb.set_ylabel("Northing (km)", labelpad=1)

    # ---- (c) example HS map ---------------------------------------------
    axc = fig.add_subplot(gs[1, 1])
    vmax = np.nanpercentile(hsmap, 99)
    im = axc.imshow(hsmap, cmap="viridis", extent=hs_ext, origin="upper",
                    vmin=0, vmax=vmax)
    axc.set_aspect("equal")
    cax = axc.inset_axes([1.03, 0.0, 0.045, 1.0])
    cb = fig.colorbar(im, cax=cax)
    cb.set_label("Snow depth (m)", fontsize=8)
    cb.ax.tick_params(labelsize=7)
    add_scalebar(axc, hs_ext[0] + 120, hs_ext[2] + 120, 500, "500 m")
    axc.xaxis.set_major_locator(mticker.MaxNLocator(3))
    axc.yaxis.set_major_locator(mticker.MaxNLocator(3))
    axc.xaxis.set_major_formatter(mticker.FuncFormatter(_km))
    axc.yaxis.set_major_formatter(mticker.FuncFormatter(_km))
    axc.set_xlabel("Easting (km, UTM 30N)")
    axc.set_ylabel("Northing (km)", labelpad=1)

    fig.savefig(OUT_DIR / "fig01_study_area.pdf", bbox_inches="tight", dpi=300)
    fig.savefig(OUT_DIR / "fig01_study_area.png", bbox_inches="tight", dpi=150)
    print("Saved:", OUT_DIR / "fig01_study_area.pdf")


if __name__ == "__main__":
    main()
