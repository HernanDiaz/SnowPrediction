"""
Figure - Generality of the hybrid spatial loss across architectures.

Spatial-loss weight sweep (lambda) for the plain U-Net (GroupNorm) and for
ResUNet++, 3 seeds each, on the temporal split. Two panels: R^2 and SPAEF vs
lambda. The spatial loss clearly helps ResUNet++ (optimum at lambda=0.4) but the
U-Net stays near R^2 = 0 for every lambda, with only a marginal, noisy SPAEF
gain -- i.e. the loss helps only when paired with a capable architecture.

Sources:
  U-Net  : results/unet_gn_lambda/<tag>_s<seed>/...metrics.json
  ResUNet++ : results/resunetpp_v4_ms_sx200_hpo[_spXX] (s42) +
              results/lambda_sweep_hpo/resunetpp_hpo_<tag>_s{123,7}

Output: paper_computers_geosciences/figures/fig10_loss_generality.pdf (+ .png)
"""

from pathlib import Path
import json
import numpy as np
import matplotlib.pyplot as plt

_REPO = Path(__file__).resolve().parents[2]
OUT = _REPO / "paper_computers_geosciences" / "figures"
OUT.mkdir(parents=True, exist_ok=True)

LAM = {"sp00": 0.0, "sp01": 0.1, "sp025": 0.25, "sp04": 0.4,
       "sp05": 0.5, "sp06": 0.6, "sp075": 0.75, "sp10": 1.0}
SEEDS = (42, 123, 7)

C_UNET = "#ff7f0e"
C_RES = "#1f77b4"
plt.rcParams.update({"font.size": 9, "axes.titlesize": 10})


def _load(p):
    d = json.load(open(p))
    return d["R2"], d["SPAEF"]


def unet_paths(tag):
    return [_REPO / f"results/unet_gn_lambda/{tag}_s{s}/unet_gn_{tag}_s{s}_metrics.json"
            for s in SEEDS]


def res_paths(tag):
    base = "resunetpp_v4_ms_sx200_hpo" if tag == "sp00" \
        else f"resunetpp_v4_ms_sx200_hpo_{tag}"
    p = [_REPO / f"results/{base}/{base}_metrics.json"]
    for s in (123, 7):
        b = f"resunetpp_hpo_{tag}_s{s}"
        p.append(_REPO / f"results/lambda_sweep_hpo/{b}/{b}_metrics.json")
    return p


def collect(path_fn):
    lams, r2m, r2s, spm, sps = [], [], [], [], []
    for tag, lam in LAM.items():
        r2, sp = [], []
        for p in path_fn(tag):
            if p.exists():
                a, b = _load(p); r2.append(a); sp.append(b)
        if not r2:
            continue
        lams.append(lam)
        r2m.append(np.mean(r2)); r2s.append(np.std(r2, ddof=1))
        spm.append(np.mean(sp)); sps.append(np.std(sp, ddof=1))
    o = np.argsort(lams)
    arr = lambda x: np.array(x)[o]
    return arr(lams), arr(r2m), arr(r2s), arr(spm), arr(sps)


def main():
    lu, r2u, r2us, spu, spus = collect(unet_paths)
    lr, r2r, r2rs, spr, sprs = collect(res_paths)

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.2), sharex=True)
    for ax, (mu, eu, mr, er, ylab) in zip(
            axes,
            [(r2u, r2us, r2r, r2rs, "$R^2$"),
             (spu, spus, spr, sprs, "SPAEF")]):
        ax.axhline(0, color="0.7", lw=0.6)
        ax.errorbar(lr, mr, yerr=er, marker="o", color=C_RES, capsize=3,
                    lw=1.5, label="ResUNet++")
        ax.errorbar(lu + 0.012, mu, yerr=eu, marker="s", color=C_UNET,
                    capsize=3, lw=1.5, label="U-Net (GroupNorm)")
        ax.set_xlabel(r"Spatial-loss weight $\lambda$")
        ax.set_ylabel(ylab)
        ax.set_xticks(list(LAM.values()))
        ax.tick_params(axis="x", labelrotation=45)
        ax.grid(True, alpha=0.3)
    axes[0].legend(frameon=False, loc="upper right", fontsize=8)

    fig.tight_layout()
    fig.savefig(OUT / "fig10_loss_generality.pdf", bbox_inches="tight")
    fig.savefig(OUT / "fig10_loss_generality.png", bbox_inches="tight", dpi=150)
    print("Saved:", OUT / "fig10_loss_generality.pdf")


if __name__ == "__main__":
    main()
