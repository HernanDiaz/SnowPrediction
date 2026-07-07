"""
Figure 6 - Effect of the spatial-loss weight lambda on test performance.

ResUNet++ (HPO architecture) trained with SpatialMSELoss for
lambda in {0, 0.1, 0.25, 0.4, 0.5, 0.6, 0.75, 1.0}, 3 seeds (42, 123, 7).
Plots mean +/- std (ddof=1) of R^2 and SPAEF vs lambda; lambda=0.4 highlighted.

Sources:
  s42 :  results/resunetpp_v4_ms_sx200_hpo[_spXX]/...metrics.json
  s123:  results/lambda_sweep_hpo/resunetpp_hpo_spXX_s123/...metrics.json
  s7  :  results/lambda_sweep_hpo/resunetpp_hpo_spXX_s7/...metrics.json

Output: paper_computers_geosciences/figures/fig06_lambda_sweep.pdf (+ .png)
"""

from pathlib import Path
import json
import numpy as np
import matplotlib.pyplot as plt

_REPO = Path(__file__).resolve().parents[2]
OUT_DIR = _REPO / "paper_computers_geosciences/figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)

LAM = {"sp00": 0.0, "sp01": 0.1, "sp025": 0.25, "sp04": 0.4,
       "sp05": 0.5, "sp06": 0.6, "sp075": 0.75, "sp10": 1.0}
BEST = 0.4

# colours consistent with the rest of the paper
C_R2 = "#1f77b4"     # ResUNet++ blue
C_SP = "#d62728"     # red

plt.rcParams.update({"font.size": 9, "axes.titlesize": 10})


def seed_paths(tag):
    # Isolated OLD-normalisation reference set (verified reproducible, 3 seeds).
    return [_REPO / f"results/paper_oldnorm/lambda/{tag}_s{s}/metrics.json"
            for s in (42, 123, 7)]


def collect():
    lams, r2m, r2s, spm, sps = [], [], [], [], []
    for tag, lam in LAM.items():
        r2, sp = [], []
        for p in seed_paths(tag):
            d = json.load(open(p))
            r2.append(d["R2"]); sp.append(d["SPAEF"])
        lams.append(lam)
        r2m.append(np.mean(r2)); r2s.append(np.std(r2, ddof=1))
        spm.append(np.mean(sp)); sps.append(np.std(sp, ddof=1))
    order = np.argsort(lams)
    arr = lambda x: np.array(x)[order]
    return (arr(lams), arr(r2m), arr(r2s), arr(spm), arr(sps))


def main():
    lams, r2m, r2s, spm, sps = collect()

    fig, ax = plt.subplots(figsize=(5.0, 3.3))
    ax.axvline(BEST, color="0.6", ls="--", lw=1, zorder=0)

    ax.errorbar(lams, r2m, yerr=r2s, marker="o", color=C_R2, capsize=3,
                lw=1.5, label="$R^2$")
    ax.errorbar(lams, spm, yerr=sps, marker="s", color=C_SP, capsize=3,
                lw=1.5, label="SPAEF")

    ax.set_xlabel(r"Spatial-loss weight $\lambda$")
    ax.set_ylabel("Test score")
    ax.set_xticks(list(LAM.values()))
    ax.grid(True, alpha=0.3)
    ax.legend(frameon=False, loc="lower center", ncol=2)

    # annotate the best lambda
    ibest = int(np.argmin(np.abs(lams - BEST)))
    ax.annotate(rf"$\lambda={BEST}$", xy=(BEST, r2m[ibest]),
                xytext=(BEST + 0.05, r2m[ibest] - 0.06),
                fontsize=8, color="0.3")

    fig.tight_layout()
    fig.savefig(OUT_DIR / "fig06_lambda_sweep.pdf", bbox_inches="tight")
    fig.savefig(OUT_DIR / "fig06_lambda_sweep.png", bbox_inches="tight", dpi=150)
    print("Saved:", OUT_DIR / "fig06_lambda_sweep.pdf")


if __name__ == "__main__":
    main()
