"""
Figure 4 - Model comparison: R2 and SPAEF (mean +/- std over 3 seeds) for the
Random Forest baseline, U-Net (GroupNorm) and ResUNet++ (lpear=0.4), all at
their best-validation checkpoint.

Values are read directly from the per-seed result JSONs so the figure stays in
sync with Tables in the paper. Seeds: 42, 123, 7. Std is the sample std (ddof=1).

Output: paper_computers_geosciences/figures/fig04_model_comparison.pdf (+ .png)
"""

from pathlib import Path
import json
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams.update({"font.size": 9, "axes.titlesize": 9})

_REPO = Path(__file__).resolve().parents[2]
RES = _REPO / "results"
OUT_DIR = _REPO / "paper_computers_geosciences/figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# One colour per model (reused across paper figures)
COL = {"Random Forest": "#7f7f7f", "U-Net": "#ff7f0e", "ResUNet++": "#1f77b4"}

# Result files per model and seed. Isolated, OLD-normalisation reference set
# (results/paper_oldnorm/), verified reproducible. All models best-validation.
#  RF: nested under 'test_metrics'. CNNs: top-level keys.
CMP = RES / "paper_oldnorm" / "comparison"
FILES = {
    "Random Forest": {
        42:  (CMP / "rf_s42/metrics.json", "test_metrics"),
        123: (CMP / "rf_s123/metrics.json", "test_metrics"),
        7:   (CMP / "rf_s7/metrics.json", "test_metrics"),
    },
    "U-Net": {
        42:  (CMP / "unet_gn_s42/metrics.json", None),
        123: (CMP / "unet_gn_s123/metrics.json", None),
        7:   (CMP / "unet_gn_s7/metrics.json", None),
    },
    "ResUNet++": {
        42:  (CMP / "resunetpp_s42/metrics.json", None),
        123: (CMP / "resunetpp_s123/metrics.json", None),
        7:   (CMP / "resunetpp_s7/metrics.json", None),
    },
}
MODELS = ["Random Forest", "U-Net", "ResUNet++"]
SEEDS = [42, 123, 7]


def value(path, sub, key):
    d = json.load(open(path))
    if sub:
        d = d[sub]
    return float(d[key])


def collect(metric):
    out = {}
    for m in MODELS:
        vals = [value(FILES[m][s][0], FILES[m][s][1], metric) for s in SEEDS]
        out[m] = (float(np.mean(vals)), float(np.std(vals, ddof=1)))
    return out


def main():
    r2 = collect("R2")
    spaef = collect("SPAEF")
    for m in MODELS:
        print(f"{m:14s} R2 {r2[m][0]:+.3f}±{r2[m][1]:.3f}   "
              f"SPAEF {spaef[m][0]:+.3f}±{spaef[m][1]:.3f}")

    fig, ax = plt.subplots(figsize=(5.0, 3.2))
    groups = ["$R^2$", "SPAEF"]
    xg = np.arange(len(groups))
    width = 0.26

    for i, m in enumerate(MODELS):
        means = [r2[m][0], spaef[m][0]]
        errs = [r2[m][1], spaef[m][1]]
        ax.bar(xg + (i - 1) * width, means, width, yerr=errs, capsize=3,
               color=COL[m], edgecolor="black", linewidth=0.5, label=m,
               error_kw=dict(elinewidth=1, ecolor="#333333"))

    ax.axhline(0, color="black", lw=0.7)
    ax.set_xticks(xg)
    ax.set_xticklabels(groups)
    ax.set_ylabel("Score")
    ax.set_ylim(0, 0.40)
    ax.legend(fontsize=8, frameon=False, loc="upper center", ncol=3,
              columnspacing=1.0, handlelength=1.2,
              bbox_to_anchor=(0.5, 1.12))
    ax.grid(True, axis="y", alpha=0.25, lw=0.5)

    fig.savefig(OUT_DIR / "fig04_model_comparison.pdf", bbox_inches="tight", dpi=300)
    fig.savefig(OUT_DIR / "fig04_model_comparison.png", bbox_inches="tight", dpi=150)
    print("Saved:", OUT_DIR / "fig04_model_comparison.pdf")


if __name__ == "__main__":
    main()
