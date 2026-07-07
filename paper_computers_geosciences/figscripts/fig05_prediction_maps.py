"""
Figure 5 - Predicted snow-depth maps for one 2025 (test) tile:
    Ground truth | Random Forest | U-Net | ResUNet++

Runs inference with the three trained models (seed 42) on a single test tile and
plots them side by side with a shared colour scale.

Model checkpoints (Articulo 1/Models):
  - RF v6:    results/optuna_rf_v6/rf_v6_best.joblib
  - U-Net:    results/optA_oldnorm/weights/unet_gn_oldnorm_s42.pth (GroupNorm, OLD norm)
  - ResUNet++: resunetpp_v4_ms_sx200_hpo_sp04.pth  (best-validation, lpear=0.4)

Output: paper_computers_geosciences/figures/fig05_prediction_maps.pdf (+ .png)
"""

from pathlib import Path
import sys
import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor

_REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO))
from models.unet import build_model            # noqa: E402
from utils.metrics import compute_spaef        # noqa: E402
from data.dataset import load_splits           # noqa: E402

DS = _REPO / "dataset_v4_ms_sx200"
CSV = DS / "dataset_v4_ms_sx200.csv"
MODELS = _REPO / "Articulo 1/Models"
OUT_DIR = _REPO / "paper_computers_geosciences/figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)

TILE = "20250327_lidar_tile_1024_2048.npy"
SEED = 42
RF_PARAMS = dict(n_estimators=300, max_depth=15, min_samples_leaf=1,
                 max_features="log2", min_samples_split=2)
RF_MAX_PIXELS = 2_000_000
UNET_PATH = _REPO / "results/optA_oldnorm/weights/unet_gn_oldnorm_s42.pth"
RES_PATH = MODELS / "resunetpp_v4_ms_sx200_hpo_sp04.pth"

UNET_CFG = {"model": {"architecture": "unet_gn", "in_channels": 22,
                      "out_channels": 1, "features": [48, 96, 192, 384],
                      "dropout_p": 0.0012, "num_groups": 8}}
RES_CFG = {"model": {"architecture": "resunetpp", "in_channels": 22,
                     "out_channels": 1, "features": [64, 128, 256, 512],
                     "dropout_p": 0.077, "num_groups": 8}}

plt.rcParams.update({"font.size": 9, "axes.titlesize": 10})


def normalize(img):
    """Replicates SnowDataset._normalize for the 22-channel stack -> (22,H,W)."""
    X = img[:22].copy().astype(np.float32)
    X[X == -9999] = 0.0
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    X[0] = (X[0] - 2100.0) / 1000.0
    X[1] = X[1] / 90.0
    X[4] = np.clip(X[4] / 9200.0, -1.0, 1.0)
    X[5] = (X[5] > 5).astype(np.float32)
    X[6:14] = np.clip(X[6:14] / 90.0, -1.0, 1.0)
    # OLD normalisation (paper reference): 5 m-topography channels 17-21 stay RAW.
    return X


RF_SAVE = _REPO / "results/rf_v6_s42/rf_v6_s42.joblib"


def train_rf():
    """Load the saved RF v6 (22ch) if present, else retrain on train+val with the
    Optuna params (seed 42) and persist it, matching compute_spaef_rf_v6.py."""
    import joblib
    if RF_SAVE.exists():
        print(f"Loading saved RF from {RF_SAVE}")
        return joblib.load(RF_SAVE)
    train_df, val_df, _ = load_splits(str(CSV), source="lidar",
                                      split_type="temporal")
    import pandas as pd
    df = pd.concat([train_df, val_df], ignore_index=True)
    X_list, y_list = [], []
    for row in df.itertuples():
        ip, mp = DS / "images" / row.tile_id, DS / "masks" / row.tile_id
        if not ip.exists() or not mp.exists():
            continue
        im = np.load(ip).astype(np.float32)
        mk = np.nan_to_num(np.load(mp).astype(np.float32), nan=0.0)
        mk[mk <= -100] = 0.0
        v = mk > 0.01
        if v.sum() == 0:
            continue
        X_list.append(normalize(im).reshape(22, -1).T[v.flatten()])
        y_list.append(mk[v])
    X = np.vstack(X_list); y = np.concatenate(y_list)
    if X.shape[0] > RF_MAX_PIXELS:
        idx = np.random.RandomState(SEED).choice(X.shape[0], RF_MAX_PIXELS,
                                                 replace=False)
        X, y = X[idx], y[idx]
    print(f"Training RF on {X.shape[0]} pixels...")
    rf = RandomForestRegressor(random_state=SEED, n_jobs=-1, **RF_PARAMS)
    rf.fit(X, y)
    RF_SAVE.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(rf, RF_SAVE, compress=3)
    print(f"Saved RF to {RF_SAVE}")
    return rf


def cnn_predict(cfg, weights, x, device):
    model = build_model(cfg).to(device)
    model.load_state_dict(torch.load(weights, map_location=device))
    model.eval()
    with torch.no_grad():
        t = torch.from_numpy(x).unsqueeze(0).to(device)   # (1,22,H,W)
        y = model(t).squeeze().cpu().numpy()
    return np.maximum(y, 0.0)


def metrics(y_true, y_pred, valid):
    yt, yp = y_true[valid], y_pred[valid]
    ss_res = np.sum((yt - yp) ** 2)
    ss_tot = np.sum((yt - yt.mean()) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan
    sp = compute_spaef(yt, yp)
    return r2, sp


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    img = np.load(DS / "images" / TILE).astype(np.float32)
    mask = np.load(DS / "masks" / TILE).astype(np.float32)
    mask = np.nan_to_num(mask, nan=0.0)
    mask[mask <= -100] = 0.0
    valid = mask > 0.01

    Xn = normalize(img)                          # (22,H,W)
    H, W = mask.shape

    rf = train_rf()
    rf_pred = np.maximum(rf.predict(Xn.reshape(22, -1).T), 0).reshape(H, W)
    unet_pred = cnn_predict(UNET_CFG, UNET_PATH, Xn, device)
    res_pred = cnn_predict(RES_CFG, RES_PATH, Xn, device)

    preds = {"Random Forest": rf_pred, "U-Net": unet_pred, "ResUNet++": res_pred}
    for name, p in preds.items():
        r2, sp = metrics(mask, p, valid)
        print(f"{name:14s} tile R2={r2:+.3f}  SPAEF={sp:+.3f}")

    # Shared colour scale from the ground truth
    vmax = float(np.percentile(mask[valid], 99))
    panels = [("Ground truth", mask)] + list(preds.items())

    fig, axes = plt.subplots(1, 4, figsize=(6.8, 2.2))
    for ax, (title, data) in zip(axes, panels):
        disp = np.where(valid, data, np.nan)
        im = ax.imshow(disp, cmap="viridis", vmin=0, vmax=vmax, origin="upper")
        ax.set_title(title)
        ax.set_xticks([]); ax.set_yticks([])

    fig.subplots_adjust(left=0.01, right=0.90, top=0.86, bottom=0.10, wspace=0.08)
    cax = fig.add_axes([0.915, 0.12, 0.014, 0.72])
    cb = fig.colorbar(im, cax=cax)
    cb.set_label("Snow depth (m)", fontsize=8)
    cb.ax.tick_params(labelsize=7)

    fig.savefig(OUT_DIR / "fig05_prediction_maps.pdf", bbox_inches="tight", dpi=300)
    fig.savefig(OUT_DIR / "fig05_prediction_maps.png", bbox_inches="tight", dpi=150)
    print("Saved:", OUT_DIR / "fig05_prediction_maps.pdf")


if __name__ == "__main__":
    main()
