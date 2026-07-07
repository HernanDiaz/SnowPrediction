"""
Figure 7 - Channel-group ablation (leave-one-group-out).

ResUNet++ (HPO architecture, MSE loss) trained with the full 22-channel stack and
with one input group removed at a time, 3 seeds (42, 123, 7). Bars show mean R^2
and SPAEF on the 2025 test set; the dashed lines mark the full-stack reference.

Sources:
  full s42 :  results/resunetpp_v4_ms_sx200_hpo/...metrics.json
  others   :  results/ablation/resunetpp_ablation_<group>_s<seed>/...metrics.json

Output: paper_computers_geosciences/figures/fig07_ablation.pdf (+ .png)
"""

from pathlib import Path
import json
import numpy as np
import matplotlib.pyplot as plt

_REPO = Path(__file__).resolve().parents[2]
OUT_DIR = _REPO / "paper_computers_geosciences/figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# display order: full first, then by decreasing damage to R^2
GROUPS = [
    ("full",      "Full (22 ch)"),
    ("sin_pers",  "\u2212 Persistence"),
    ("sin_topo5", "\u2212 Topo 5 m"),
    ("sin_sx",    "\u2212 Sx"),
    ("sin_topo1", "\u2212 Topo 1 m"),
    ("sin_sce",   "\u2212 SCE"),
]
SEEDS = (42, 123, 7)

C_R2 = "#1f77b4"
C_SP = "#d62728"
plt.rcParams.update({"font.size": 9, "axes.titlesize": 10})


def path_for(group, seed):
    # Isolated OLD-normalisation reference set (verified reproducible, 18 runs).
    return _REPO / f"results/paper_oldnorm/ablation/{group}_s{seed}/metrics.json"


def collect():
    labels, r2m, r2s, spm, sps = [], [], [], [], []
    for g, lab in GROUPS:
        r2, sp = [], []
        for s in SEEDS:
            d = json.load(open(path_for(g, s)))
            r2.append(d["R2"]); sp.append(d["SPAEF"])
        labels.append(lab)
        r2m.append(np.mean(r2)); r2s.append(np.std(r2, ddof=1))
        spm.append(np.mean(sp)); sps.append(np.std(sp, ddof=1))
    return labels, r2m, r2s, spm, sps


def main():
    labels, r2m, r2s, spm, sps = collect()
    x = np.arange(len(labels))
    w = 0.38

    fig, ax = plt.subplots(figsize=(6.2, 3.4))
    ax.bar(x - w / 2, r2m, w, yerr=r2s, capsize=3, color=C_R2, label="$R^2$")
    ax.bar(x + w / 2, spm, w, yerr=sps, capsize=3, color=C_SP, label="SPAEF")

    # full-stack reference lines
    ax.axhline(r2m[0], color=C_R2, ls="--", lw=0.8, alpha=0.6)
    ax.axhline(spm[0], color=C_SP, ls="--", lw=0.8, alpha=0.6)
    ax.axhline(0, color="0.5", lw=0.6)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_ylabel("Test score")
    ax.legend(frameon=False, ncol=2, loc="upper right")
    ax.grid(True, axis="y", alpha=0.3)

    fig.tight_layout()
    fig.savefig(OUT_DIR / "fig07_ablation.pdf", bbox_inches="tight")
    fig.savefig(OUT_DIR / "fig07_ablation.png", bbox_inches="tight", dpi=150)
    print("Saved:", OUT_DIR / "fig07_ablation.pdf")


if __name__ == "__main__":
    main()
