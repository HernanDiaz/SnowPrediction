"""
ResUNet++ architecture diagram (compact, publication-ready).

Reproduces the visual style of the MAPunet `architecture_svg_compact.py`
(pastel encoder/decoder/bottleneck boxes with title bars, pool boxes, dashed
skip connections and a bottom legend) but rendered with matplotlib so it exports
straight to PDF for the LaTeX manuscript.

Architecture (from models/resunet.py, features=[64,128,256,512], 22 input ch):
  Input 256x256x22
  Enc1 ResBlock+SE 64   (256) -> MaxPool
  Enc2 ResBlock+SE 128  (128) -> MaxPool
  Enc3 ResBlock+SE 256  (64)  -> MaxPool
  ASPP bottleneck 512   (32)  [ResBlock + ASPP r=1,6,12,18 + ResBlock]
  Dec3 up + AttGate(e3) -> 256 (64)
  Dec2 up + AttGate(e2) -> 128 (128)
  Dec1 up + AttGate(e1) -> 64  (256)
  Output head Conv 1x1 (linear) -> HS 256x256x1

Output: paper_computers_geosciences/figures/fig_architecture.pdf (+ .png)
"""

from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle

OUT_DIR = Path(__file__).resolve().parents[1] / "figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── palette (same as the MAPunet SVG) ──────────────────────────────────────
C_ENC, T_ENC = "#D6EAF8", "#C5D9E8"
C_DEC, T_DEC = "#FDEBD0", "#F0C9A0"
C_BN,  T_BN  = "#D5F5E3", "#C9E8C5"
C_HEAD, T_HEAD = "#FADBD8", "#E8C5C5"
C_POOL = "#EAECEE"
C_IO   = "#F2F3F4"
C_EDGE = "#2C3E50"
C_SKIP = "#7F8C8D"
C_ARR  = "#2C3E50"
C_DIM  = "#555555"

# ── geometry (points) ──────────────────────────────────────────────────────
BW, TH, LH, POOL = 66, 15, 14, 12
BODY = 2 * LH
BH = TH + BODY            # 43
STEP = BW + 20            # column pitch
ROWGAP = 40

FS_T, FS_B, FS_P, FS_H, FS_LEG = 7.0, 6.3, 5.8, 8.5, 6.0

plt.rcParams.update({"font.family": "serif",
                     "font.serif": ["Times New Roman", "DejaVu Serif"]})


def box(ax, left, top, title, ops, out, fill, tfill, pool=None):
    """Draw a compact block anchored at its TOP-left; return bottom y."""
    def rect(x, y_top, h, fc):
        ax.add_patch(Rectangle((x, y_top - h), BW, h, facecolor=fc,
                               edgecolor=C_EDGE, linewidth=0.6, zorder=2))
    rect(left, top, TH, tfill)
    ax.text(left + BW / 2, top - TH / 2, title, ha="center", va="center",
            fontsize=FS_T, fontweight="bold", zorder=3)
    rect(left, top - TH, LH, fill)
    ax.text(left + BW / 2, top - TH - LH / 2, ops, ha="center", va="center",
            fontsize=FS_B, zorder=3)
    rect(left, top - TH - LH, LH, fill)
    ax.text(left + BW / 2, top - TH - 1.5 * LH, out, ha="center", va="center",
            fontsize=FS_B, zorder=3)
    bottom = top - BH
    if pool:
        rect(left, bottom, POOL, C_POOL)
        ax.text(left + BW / 2, bottom - POOL / 2, pool, ha="center",
                va="center", fontsize=FS_P, zorder=3)
        bottom -= POOL
    return bottom


def arrow(ax, x1, y1, x2, y2, color=C_ARR, lw=0.9, dashed=False):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle="-|>", color=color, lw=lw,
                                shrinkA=0, shrinkB=0,
                                linestyle="--" if dashed else "-",
                                mutation_scale=8), zorder=1)


def elbow(ax, pts, color=C_ARR, lw=0.9):
    for (xa, ya), (xb, yb) in zip(pts[:-1], pts[1:-1] + [pts[-1]]):
        pass  # placeholder (unused)


def main():
    fig, ax = plt.subplots(figsize=(6.0, 3.0))

    # column left-x positions
    cx = [i * STEP for i in range(5)]   # c0..c4
    enc_top = 0.0
    dec_top = enc_top - (BH + POOL) - ROWGAP

    enc_cy = enc_top - BH / 2
    dec_cy = dec_top - BH / 2

    # ── Input box (c0, encoder row) ────────────────────────────────────────
    io_h = 18
    io_top = enc_cy + io_h / 2
    ax.add_patch(Rectangle((cx[0], io_top - io_h), BW, io_h, facecolor=C_IO,
                           edgecolor=C_EDGE, linewidth=0.6, zorder=2))
    ax.text(cx[0] + BW / 2, enc_cy, "Input\n256\u00d7256\u00d722",
            ha="center", va="center", fontsize=FS_B, zorder=3)

    # ── Encoder blocks (c1..c3) ────────────────────────────────────────────
    enc_bottoms = {}
    enc_specs = [
        (1, "Enc 1", "ResBlock+SE 64", "256\u00b2\u00d764", "MaxPool 2\u00d72"),
        (2, "Enc 2", "ResBlock+SE 128", "128\u00b2\u00d7128", "MaxPool 2\u00d72"),
        (3, "Enc 3", "ResBlock+SE 256", "64\u00b2\u00d7256", "MaxPool 2\u00d72"),
    ]
    for c, title, ops, out, pool in enc_specs:
        enc_bottoms[c] = box(ax, cx[c], enc_top, title, ops, out,
                             C_ENC, T_ENC, pool=pool)

    # ── ASPP bottleneck (c4) ───────────────────────────────────────────────
    bn_bottom = box(ax, cx[4], enc_top, "ASPP",
                    "ResBlk+ASPP 512", "32\u00b2\u00d7512", C_BN, T_BN)

    # ── Decoder blocks (c3..c1) ────────────────────────────────────────────
    dec_specs = [
        (3, "Dec 3", "\u2191 + AG(e3) 256", "64\u00b2\u00d7256"),
        (2, "Dec 2", "\u2191 + AG(e2) 128", "128\u00b2\u00d7128"),
        (1, "Dec 1", "\u2191 + AG(e1) 64", "256\u00b2\u00d764"),
    ]
    for c, title, ops, out in dec_specs:
        box(ax, cx[c], dec_top, title, ops, out, C_DEC, T_DEC)

    # ── Output head (c0, decoder row) ──────────────────────────────────────
    box(ax, cx[0], dec_top, "Output", "Conv 1\u00d71 (lin.)",
        "HS 256\u00b2\u00d71", C_HEAD, T_HEAD)

    # ── arrows: input -> enc1 -> enc2 -> enc3 -> aspp ──────────────────────
    arrow(ax, cx[0] + BW, enc_cy, cx[1], enc_cy)
    arrow(ax, cx[1] + BW, enc_cy, cx[2], enc_cy)
    arrow(ax, cx[2] + BW, enc_cy, cx[3], enc_cy)
    arrow(ax, cx[3] + BW, enc_cy, cx[4], enc_cy)

    # ── decoder arrows: dec3 -> dec2 -> dec1 -> head (right to left) ────────
    arrow(ax, cx[3], dec_cy, cx[2] + BW, dec_cy)
    arrow(ax, cx[2], dec_cy, cx[1] + BW, dec_cy)
    arrow(ax, cx[1], dec_cy, cx[0] + BW, dec_cy)

    # ── ASPP -> Dec3 elbow (down then into right edge of dec3) ──────────────
    xa = cx[4] + BW / 2
    ax.plot([xa, xa, cx[3] + BW + 6],
            [bn_bottom, dec_cy, dec_cy], color=C_ARR, lw=0.9, zorder=1)
    arrow(ax, cx[3] + BW + 6, dec_cy, cx[3] + BW, dec_cy)

    # ── skip connections enc -> dec with attention gate (dashed) ───────────
    for c, lbl in [(1, "e1"), (2, "e2"), (3, "e3")]:
        sx = cx[c] + BW / 2
        arrow(ax, sx, enc_bottoms[c], sx, dec_top, color=C_SKIP, lw=0.9,
              dashed=True)
        ax.text(sx + 5, (enc_bottoms[c] + dec_top) / 2,
                f"AG\u00b7{lbl}", ha="left", va="center",
                fontsize=5.4, color=C_DIM, zorder=3)

    # ── section labels ─────────────────────────────────────────────────────
    ax.text((cx[0] + cx[4] + BW) / 2, enc_top + 13,
            "RESUNET++ ENCODER  (residual + squeeze-excitation)",
            ha="center", va="center", fontsize=FS_H, fontweight="bold")
    ax.text((cx[0] + cx[3] + BW) / 2, dec_top + 13,
            "ATTENTION DECODER",
            ha="center", va="center", fontsize=FS_H, fontweight="bold")

    # ── legend ─────────────────────────────────────────────────────────────
    leg = [(C_ENC, "Encoder"), (C_DEC, "Decoder"),
           (C_BN, "ASPP bottleneck"), (C_HEAD, "Output head"),
           (C_POOL, "MaxPool"), ("skip", "Skip + att. gate")]
    x0 = cx[0]
    total_w = cx[4] + BW - cx[0]
    slot = total_w / len(leg)
    ly = dec_top - BH - 16
    for k, (c, lbl) in enumerate(leg):
        lx = x0 + k * slot
        if c == "skip":
            ax.plot([lx, lx + 9], [ly, ly], color=C_SKIP, lw=1.0, ls="--")
        else:
            ax.add_patch(Rectangle((lx, ly - 4), 9, 8, facecolor=c,
                                   edgecolor="#888888", linewidth=0.4))
        ax.text(lx + 12, ly, lbl, ha="left", va="center", fontsize=FS_LEG,
                color="#333333")

    # ── finalize ───────────────────────────────────────────────────────────
    ax.set_xlim(cx[0] - 8, cx[4] + BW + 8)
    ax.set_ylim(ly - 12, enc_top + 22)
    ax.set_aspect("equal")
    ax.axis("off")
    fig.savefig(OUT_DIR / "fig_architecture.pdf", bbox_inches="tight")
    fig.savefig(OUT_DIR / "fig_architecture.png", bbox_inches="tight", dpi=200)
    print("Saved:", OUT_DIR / "fig_architecture.pdf")


if __name__ == "__main__":
    main()
