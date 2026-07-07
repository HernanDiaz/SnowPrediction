"""
Figure 8 - Effect of adding meteorological channels.

Compares the 22-channel ResUNet++ (topography + persistence) with the 26-channel
configuration that adds four scalar meteorological forcings (T2m 7/15/30 d,
accumulated precipitation), across the spatial-loss weight lambda, 3 seeds.
Two panels: R^2 and SPAEF vs lambda. The meteo variant is not better on the
central metrics and shows larger seed-to-seed variance.

Sources:
  22ch : results/resunetpp_v4_ms_sx200_hpo[_spXX] + lambda_sweep_hpo/...s{123,7}
  26ch : results/lambda_sweep_meteo/resunetpp_meteo_spXX_s{42,123,7}

Output: paper_computers_geosciences/figures/fig08_meteo.pdf (+ .png)
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

C22 = "#1f77b4"   # 22-ch (reference)
C26 = "#ff7f0e"   # 26-ch (meteo)
plt.rcParams.update({"font.size": 9, "axes.titlesize": 10})


def _paths_22(tag):
    base = "resunetpp_v4_ms_sx200_hpo" if tag == "sp00" \
        else f"resunetpp_v4_ms_sx200_hpo_{tag}"
    p = [_REPO / f"results/{base}/{base}_metrics.json"]
    for s in (123, 7):
        b = f"resunetpp_hpo_{tag}_s{s}"
        p.append(_REPO / f"results/lambda_sweep_hpo/{b}/{b}_metrics.json")
    return p


def _paths_26(tag):
    p = []
    for s in (42, 123, 7):
        b = f"resunetpp_meteo_{tag}_s{s}"
        p.append(_REPO / f"results/lambda_sweep_meteo/{b}/{b}_metrics.json")
    return p


def collect(path_fn):
    lams, r2m, r2s, spm, sps = [], [], [], [], []
    for tag, lam in LAM.items():
        r2, sp = [], []
        for p in path_fn(tag):
            d = json.load(open(p))
            r2.append(d["R2"]); sp.append(d["SPAEF"])
        lams.append(lam)
        r2m.append(np.mean(r2)); r2s.append(np.std(r2, ddof=1))
        spm.append(np.mean(sp)); sps.append(np.std(sp, ddof=1))
    o = np.argsort(lams)
    a = lambda x: np.array(x)[o]
    return a(lams), a(r2m), a(r2s), a(spm), a(sps)


def main():
    l22, r22, r22s, s22, s22s = collect(_paths_22)
    l26, r26, r26s, s26, s26s = collect(_paths_26)

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.2), sharex=True)
    for ax, (m22, e22, m26, e26, ylab) in zip(
            axes,
            [(r22, r22s, r26, r26s, "$R^2$"),
             (s22, s22s, s26, s26s, "SPAEF")]):
        ax.errorbar(l22, m22, yerr=e22, marker="o", color=C22, capsize=3,
                    lw=1.5, label="22 ch (topo+pers.)")
        ax.errorbar(l26 + 0.012, m26, yerr=e26, marker="s", color=C26,
                    capsize=3, lw=1.5, label="26 ch (+ meteo)")
        ax.set_xlabel(r"Spatial-loss weight $\lambda$")
        ax.set_ylabel(ylab)
        ax.set_xticks(list(LAM.values()))
        ax.tick_params(axis="x", labelrotation=45)
        ax.grid(True, alpha=0.3)
    axes[0].legend(frameon=False, loc="lower center", fontsize=8)

    fig.tight_layout()
    fig.savefig(OUT_DIR / "fig08_meteo.pdf", bbox_inches="tight")
    fig.savefig(OUT_DIR / "fig08_meteo.png", bbox_inches="tight", dpi=150)
    print("Saved:", OUT_DIR / "fig08_meteo.pdf")


if __name__ == "__main__":
    main()
