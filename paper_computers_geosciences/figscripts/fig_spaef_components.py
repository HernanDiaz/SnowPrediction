"""
Figure - SPAEF component decomposition by model.

SPAEF = 1 - sqrt((rho-1)^2 + (alpha-1)^2 + (beta-1)^2), where
    rho   = spatial Pearson correlation (pattern co-location),
    alpha = ratio of coefficients of variation (CV_sim / CV_obs),
    beta  = histogram intersection (value-distribution overlap).
A perfect field has rho = alpha = beta = 1.

We compute each component per tile on the 2025 temporal test set and average
over tiles (and over the 3 seeds for the CNNs; the RF is near-deterministic and
uses seed 42). The figure shows why SPAEF reorders the models: the per-pixel RF
attains a decent correlation but strongly under-disperses (alpha << 1, an
over-smoothed field), whereas the CNNs keep alpha near 1; ResUNet++ leads on all
three components.

Output: paper_computers_geosciences/figures/fig_spaef_components.pdf (+ .png)
"""

from pathlib import Path
import sys
import numpy as np
import torch
import joblib
import pandas as pd
import matplotlib.pyplot as plt

_REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO))
from models.unet import build_model            # noqa: E402

DS = _REPO / "dataset_v4_ms_sx200"
MODELS = _REPO / "Articulo 1/Models"
OUT = _REPO / "paper_computers_geosciences/figures"
OUT.mkdir(parents=True, exist_ok=True)

UNET_CFG = {"model": {"architecture": "unet_gn", "in_channels": 22,
                      "out_channels": 1, "features": [48, 96, 192, 384],
                      "dropout_p": 0.0012, "num_groups": 8}}
RES_CFG = {"model": {"architecture": "resunetpp", "in_channels": 22,
                     "out_channels": 1, "features": [64, 128, 256, 512],
                     "dropout_p": 0.077, "num_groups": 8}}

RF_PATH = _REPO / "results/rf_v6_s42/rf_v6_s42.joblib"
UNET_W = [_REPO / f"results/unet_gn/weights/unet_gn_s{s}.pth" for s in (42, 123, 7)]
RES_W = [MODELS / "resunetpp_v4_ms_sx200_hpo_sp04.pth",
         MODELS / "resunetpp_hpo_sp04_s123.pth",
         MODELS / "resunetpp_hpo_sp04_s7.pth"]

C = {"Random Forest": "#7f7f7f", "U-Net (GroupNorm)": "#ff7f0e",
     "ResUNet++": "#1f77b4"}
plt.rcParams.update({"font.size": 9, "axes.titlesize": 10})


def normalize(img):
    X = img[:22].copy().astype(np.float32)
    X[X == -9999] = 0.0
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    X[0] = (X[0] - 2100.0) / 1000.0
    X[1] = X[1] / 90.0
    X[4] = np.clip(X[4] / 9200.0, -1.0, 1.0)
    X[5] = (X[5] > 5).astype(np.float32)
    X[6:14] = np.clip(X[6:14] / 90.0, -1.0, 1.0)
    X[17] = (X[17] - 2100.0) / 1000.0
    X[18] = X[18] / 90.0
    X[21] = np.clip(X[21] / 9200.0, -1.0, 1.0)
    return X


def components(o, s, n_bins=100):
    s = np.maximum(s, 0.0)
    if len(o) < 10:
        return None
    rho = np.corrcoef(o, s)[0, 1]
    mo, ms = o.mean(), s.mean()
    if mo == 0 or ms == 0 or o.std() == 0:
        return None
    alpha = (s.std() / ms) / (o.std() / mo)
    lo, hi = min(o.min(), s.min()), max(o.max(), s.max())
    if hi <= lo:
        return None
    b = np.linspace(lo, hi, n_bins + 1)
    ho, _ = np.histogram(o, bins=b); hs, _ = np.histogram(s, bins=b)
    ho = ho / (ho.sum() + 1e-10); hs = hs / (hs.sum() + 1e-10)
    beta = np.minimum(ho, hs).sum()
    if np.isnan(rho):
        return None
    return rho, alpha, beta


def cnn_models(cfg, weights, dev):
    ms = []
    for w in weights:
        m = build_model(cfg).to(dev)
        m.load_state_dict(torch.load(w, map_location=dev)); m.eval()
        ms.append(m)
    return ms


def main():
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    rf = joblib.load(RF_PATH)
    unets = cnn_models(UNET_CFG, UNET_W, dev)
    ress = cnn_models(RES_CFG, RES_W, dev)

    df = pd.read_csv(DS / "dataset_v4_ms_sx200.csv")
    test = df[df.exp_temporal_split == "test"]

    acc = {"Random Forest": [], "U-Net (GroupNorm)": [], "ResUNet++": []}
    for _, r in test.iterrows():
        ip, mp = DS / "images" / r.tile_id, DS / "masks" / r.tile_id
        if not ip.exists():
            continue
        im = np.load(ip).astype(np.float32)
        mk = np.nan_to_num(np.load(mp).astype(np.float32), nan=0.0); mk[mk <= -100] = 0
        v = mk > 0.01
        if v.sum() < 10:
            continue
        Xn = normalize(im); o = mk[v]
        # RF (seed 42)
        fp = np.maximum(rf.predict(Xn.reshape(22, -1).T), 0).reshape(mk.shape)[v]
        c = components(o, fp)
        if c:
            acc["Random Forest"].append(c)
        # CNNs: each seed model
        t = torch.from_numpy(Xn).unsqueeze(0).to(dev)
        with torch.no_grad():
            for m in unets:
                up = np.maximum(m(t).squeeze().cpu().numpy(), 0)[v]
                c = components(o, up)
                if c:
                    acc["U-Net (GroupNorm)"].append(c)
            for m in ress:
                rp = np.maximum(m(t).squeeze().cpu().numpy(), 0)[v]
                c = components(o, rp)
                if c:
                    acc["ResUNet++"].append(c)

    names = ["Random Forest", "U-Net (GroupNorm)", "ResUNet++"]
    comp_labels = [r"$\rho$ (correlation)", r"$\alpha$ (CV ratio)",
                   r"$\beta$ (histogram)"]
    means = {n: np.array(acc[n]).mean(axis=0) for n in names}
    for n in names:
        a = np.array(acc[n])
        print(f"{n:18s} rho={a[:,0].mean():+.3f} alpha={a[:,1].mean():.3f} "
              f"beta={a[:,2].mean():.3f}  (n={len(a)})")

    x = np.arange(3)
    w = 0.26
    fig, ax = plt.subplots(figsize=(5.6, 3.4))
    for i, n in enumerate(names):
        ax.bar(x + (i - 1) * w, means[n], w, color=C[n], label=n,
               edgecolor="white", linewidth=0.4)
    ax.axhline(1.0, color="0.35", ls="--", lw=0.9)
    ax.text(2.43, 1.02, "ideal = 1", fontsize=7, color="0.35", ha="right")
    ax.set_xticks(x)
    ax.set_xticklabels(comp_labels)
    ax.set_ylabel("Component value")
    ax.set_ylim(0, 1.25)
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / "fig_spaef_components.pdf", bbox_inches="tight")
    fig.savefig(OUT / "fig_spaef_components.png", bbox_inches="tight", dpi=150)
    print("Saved:", OUT / "fig_spaef_components.pdf")


if __name__ == "__main__":
    main()
