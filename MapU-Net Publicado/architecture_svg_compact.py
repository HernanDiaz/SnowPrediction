"""
architecture_svg_compact.py — DocVerify compact architecture, publication-ready.

Designed for IEEE double-column (≈7 in wide).  All layout constants are in
typographic points (1 pt = 1/72 in), so font-size="7" prints as exactly 7 pt
when the figure is placed at its specified physical width.

Each encoder / decoder block is a single compact box (title + ops summary +
output resolution), suitable for the main body of a Q1 journal paper.

NOTE: arrows use explicit inline <polygon> arrowheads (svglib does not
support SVG <marker> elements, so marker-end="url(#ah)" produces invisible
arrowheads when converted via svglib/reportlab).

Run standalone:
    python architecture_svg_compact.py
Output:
    paper_figures/output/fig0_architecture_compact.svg
"""

from pathlib import Path

# ── Layout constants (all in typographic points) ──────────────────────────────
BW       = 49      # block width (pt)
TITLE_H  = 13      # title bar height inside block
LINE_H   = 12      # one content-line height
POOL_H   = 10      # pool/upsample box below applicable enc blocks
IO_H     = 14      # I/O box height (input only)
HGAP     = 6       # horizontal gap between blocks (arrow zone)
ROW_GAP  = 38      # encoder-row → decoder-row vertical gap
MX       = 2       # left/right canvas margin (tight — no wasted whitespace)
MY       = 18      # top margin (just enough for section labels at MY-11)
BH       = TITLE_H + 2 * LINE_H   # block height without pool  (= 37 pt)

FONT = "'Times New Roman', Georgia, serif"
FS_T  = 6.8    # title bar font size
FS_B  = 6.2    # body (ops / out lines)
FS_P  = 5.8    # pool-box label
FS_H  = 8.5    # section header
FS_IO = 7.0    # I/O box text

# ── Colour palette ─────────────────────────────────────────────────────────────
C_ENC  = "#D6EAF8"; C_DEC  = "#FDEBD0"; C_CLS  = "#FADBD8"; C_BN   = "#D5F5E3"
C_POOL = "#EAECEE"; C_IO   = "#F2F3F4"; C_EDGE = "#2C3E50"
C_SKIP = "#7F8C8D"; C_ARR  = "#2C3E50"; C_DIM  = "#555555"

# ── Arrowhead dimensions (pt) ─────────────────────────────────────────────────
AH_LEN   = 4.8   # length of arrowhead along shaft
AH_HW    = 2.0   # half-width of arrowhead base
AH_LEN_S = 3.5   # skip-connection arrowhead (smaller)
AH_HW_S  = 1.5

# ── Architecture data ──────────────────────────────────────────────────────────
ENC = [
    ("Enc Block 1", "Conv 8ch, k3",    "→ 224×224×8",   None,          None),
    ("Enc Block 2", "2×Conv 16ch, k3", "→ 224×224×16",  "AvgPool 2×2", "s224"),
    ("Enc Block 3", "3×Conv 32ch, k3", "→ 112×112×32",  "AvgPool 2×2", "s112"),
    ("Enc Block 4", "4×Conv 64ch, k3", "→ 56×56×64",    "AvgPool 2×2", "s56"),
    ("Enc Block 5", "Conv 128ch, k5",  "→ 28×28×128",   "MaxPool 2×2", "s28"),
    ("Enc Block 6", "Conv 256ch, k5",  "→ 14×14×256",   "MaxPool 2×2", "s14"),
]
BN  = ("Bottleneck", "AdaptAvgPool", "→ 7×7×256")
DEC = [
    ("Dec Block 1", "↑ + s14  (384ch)", "→ 14×14×128"),
    ("Dec Block 2", "↑ + s28  (256ch)", "→ 28×28×64"),
    ("Dec Block 3", "↑ + s56  (128ch)", "→ 56×56×32"),
    ("Dec Block 4", "↑ + s112 (64ch)",  "→ 112×112×16"),
    ("Dec Block 5", "↑ + s224 (32ch)",  "→ 224×224×8"),
]
CLS = ("Cls Head",  "4×FC + Dropout",   "→ Score (σ)")
SEG = ("Seg Head",  "Conv 1ch, k1×1",   "→ Mask 224×224")

LEG_ITEMS = [
    (C_ENC,  "Encoder block"),
    (C_DEC,  "Decoder block"),
    (C_BN,   "Bottleneck"),
    (C_CLS,  "Head (CLS / SEG)"),
    (C_POOL, "Pool / Upsample"),
    ("skip",  "Skip connection"),
]


# ── SVG primitives ────────────────────────────────────────────────────────────

def _r(x, y, w, h, fill, stroke=C_EDGE, sw=0.6, rx=0):
    """Rectangle — rx=0 = square corners."""
    return (f'<rect x="{x:.1f}" y="{y:.1f}" width="{w:.1f}" height="{h:.1f}" '
            f'rx="{rx}" fill="{fill}" stroke="{stroke}" stroke-width="{sw}"/>')


def _t(x, y, s, size=FS_B, weight="normal", fill="#111111", anchor="middle"):
    s = s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    return (f'<text x="{x:.1f}" y="{y:.1f}" text-anchor="{anchor}" '
            f'dominant-baseline="middle" font-size="{size}" font-weight="{weight}" '
            f'fill="{fill}" font-family="{FONT}">{s}</text>')


def _arrowhead(x, y, direction, color=C_ARR, ln=AH_LEN, hw=AH_HW):
    """Inline filled triangle arrowhead at tip (x,y).

    direction: 'r'=pointing right, 'l'=pointing left,
               'u'=pointing up,    'd'=pointing down.
    The triangle base is ln pt behind the tip, hw pt on each side.
    """
    if direction == 'r':
        pts = f"{x-ln:.1f},{y-hw:.1f} {x:.1f},{y:.1f} {x-ln:.1f},{y+hw:.1f}"
    elif direction == 'l':
        pts = f"{x+ln:.1f},{y-hw:.1f} {x:.1f},{y:.1f} {x+ln:.1f},{y+hw:.1f}"
    elif direction == 'd':
        pts = f"{x-hw:.1f},{y-ln:.1f} {x:.1f},{y:.1f} {x+hw:.1f},{y-ln:.1f}"
    elif direction == 'u':
        pts = f"{x-hw:.1f},{y+ln:.1f} {x:.1f},{y:.1f} {x+hw:.1f},{y+ln:.1f}"
    else:
        raise ValueError(f"Unknown direction: {direction!r}")
    return f'<polygon points="{pts}" fill="{color}"/>'


def _ah(x1, y, x2, color=C_ARR, lw=0.8):
    """Horizontal arrow (handles both L→R and R→L).

    Returns a string with the shaft line + inline arrowhead polygon.
    """
    if x2 >= x1:
        direction = 'r'
        xe = x2 - AH_LEN          # shaft stops AH_LEN pt before tip
    else:
        direction = 'l'
        xe = x2 + AH_LEN
    line = (f'<line x1="{x1:.1f}" y1="{y:.1f}" x2="{xe:.1f}" y2="{y:.1f}" '
            f'stroke="{color}" stroke-width="{lw}"/>')
    ah = _arrowhead(x2, y, direction, color)
    return line + "\n" + ah


def _av(x, y1, y2, color=C_ARR, lw=0.8, dashed=False):
    """Vertical arrow (handles both up and down).

    Returns a string with the shaft line + inline arrowhead polygon.
    """
    d = ' stroke-dasharray="4,2.5"' if dashed else ''
    if y2 > y1:
        direction = 'd'
        ye = y2 - AH_LEN
    else:
        direction = 'u'
        ye = y2 + AH_LEN
    line = (f'<line x1="{x:.1f}" y1="{y1:.1f}" x2="{x:.1f}" y2="{ye:.1f}" '
            f'stroke="{color}" stroke-width="{lw}"{d}/>')
    ah = _arrowhead(x, y2, direction, color)
    return line + "\n" + ah


def _seg3(x1, y1, x2, y2, x3, y3, color=C_ARR, lw=0.8):
    """3-segment path (two elbows) with arrowhead at (x3,y3).
    The final segment (x2,y2)→(x3,y3) must be axis-aligned.
    """
    dy = 1 if y3 > y2 else (-1 if y3 < y2 else 0)
    dx = 1 if x3 > x2 else (-1 if x3 < x2 else 0)
    direction = {(1, 0): 'r', (-1, 0): 'l', (0, 1): 'd', (0, -1): 'u'}[(dx, dy)]
    xe = x3 - dx * AH_LEN
    ye = y3 - dy * AH_LEN
    polyline = (f'<polyline points="{x1:.1f},{y1:.1f} {x2:.1f},{y2:.1f} '
                f'{xe:.1f},{ye:.1f}" '
                f'fill="none" stroke="{color}" stroke-width="{lw}"/>')
    ah = _arrowhead(x3, y3, direction, color)
    return polyline + "\n" + ah


def _seg4(x1, y1, x2, y2, x3, y3, x4, y4, color=C_ARR, lw=0.8):
    """4-point path with arrowhead at (x4,y4). Last segment must be axis-aligned."""
    dy = 1 if y4 > y3 else (-1 if y4 < y3 else 0)
    dx = 1 if x4 > x3 else (-1 if x4 < x3 else 0)
    direction = {(1, 0): 'r', (-1, 0): 'l', (0, 1): 'd', (0, -1): 'u'}[(dx, dy)]
    xe = x4 - dx * AH_LEN
    ye = y4 - dy * AH_LEN
    polyline = (f'<polyline '
                f'points="{x1:.1f},{y1:.1f} {x2:.1f},{y2:.1f} '
                f'{x3:.1f},{y3:.1f} {xe:.1f},{ye:.1f}" '
                f'fill="none" stroke="{color}" stroke-width="{lw}"/>')
    ah = _arrowhead(x4, y4, direction, color)
    return polyline + "\n" + ah


# ── Block drawing ─────────────────────────────────────────────────────────────

def _block(x, y, title, ops, out, fill, pool=None):
    """Draw a compact block. Returns (svg_parts, bottom_y)."""
    tc = ("#C5D9E8" if fill == C_ENC else
          "#F0C9A0" if fill == C_DEC else
          "#C9E8C5" if fill == C_BN  else "#E8C5C5")
    ps, cy = [], y
    ps += [_r(x, cy, BW, TITLE_H, tc, sw=0.6),
           _t(x + BW/2, cy + TITLE_H/2, title, size=FS_T, weight="bold")]
    cy += TITLE_H
    ps += [_r(x, cy, BW, LINE_H, fill, sw=0.6),
           _t(x + BW/2, cy + LINE_H/2, ops, size=FS_B)]
    cy += LINE_H
    ps += [_r(x, cy, BW, LINE_H, fill, sw=0.6),
           _t(x + BW/2, cy + LINE_H/2, out, size=FS_B)]
    cy += LINE_H
    if pool:
        ps += [_r(x, cy, BW, POOL_H, C_POOL, sw=0.6),
               _t(x + BW/2, cy + POOL_H/2, pool, size=FS_P)]
        cy += POOL_H
    return ps, cy


# ── Main SVG builder ──────────────────────────────────────────────────────────

def build_svg() -> str:
    parts = []

    # ── X positions ────────────────────────────────────────────────────────
    STEP  = BW + HGAP                                 # 59 pt per slot
    enc_x = [MX + i * STEP for i in range(6)]        # enc[0..5]
    dec_x = [enc_x[5 - i] for i in range(5)]         # dec[0]=enc5 .. dec[4]=enc1
    SEG_X = dec_x[4] - STEP                           # leftmost column (= MX)
    CLS_X = enc_x[5] + STEP + 3                        # right-most column

    # ── Row Y positions ─────────────────────────────────────────────────────
    max_enc_h = BH + POOL_H             # tallest encoder block = 47 pt
    R1_TOP    = MY                      # = 20
    R1_BOT    = R1_TOP + max_enc_h + 4  # = 71  (bottom of encoder row)
    R2_TOP    = R1_BOT + ROW_GAP        # = 109 (top of decoder row)
    R2_BOT    = R2_TOP + BH             # = 146

    # Arrow y levels
    bn_arr_y  = R1_TOP + BH / 2         # = 38.5  (encoder + bottleneck row)
    dec_arr_y = R2_TOP + BH / 2         # = 127.5 (decoder + cls head row)

    # ── Bottleneck (same height as encoder row) ──────────────────────────────
    bn_parts, bn_bot = _block(CLS_X, R1_TOP, *BN, C_BN)   # bn_bot = 57

    # ── Cls head (same height as decoder row) ────────────────────────────────
    cls_parts, cls_bot = _block(CLS_X, R2_TOP, *CLS, C_CLS)  # cls_bot = 146

    # ── Seg head ─────────────────────────────────────────────────────────────
    seg_parts, _ = _block(SEG_X, R2_TOP, *SEG, C_CLS)

    # ── Canvas size ──────────────────────────────────────────────────────────
    CW          = CLS_X + BW + MX
    content_bot = max(R2_BOT, cls_bot)          # = 146
    sep_y       = content_bot + 6               # separator line y
    leg_y       = sep_y + 11                    # legend items centre y
    CH          = leg_y + 9
    CW_IN = f"{CW / 72:.3f}in"
    CH_IN = f"{CH / 72:.3f}in"

    # ── SVG header (no <defs> needed — arrowheads are inline polygons) ───────
    parts.append(f'''<?xml version="1.0" encoding="UTF-8"?>
<svg xmlns="http://www.w3.org/2000/svg"
     width="{CW_IN}" height="{CH_IN}"
     viewBox="0 0 {CW:.1f} {CH:.1f}">
  <rect width="{CW:.1f}" height="{CH:.1f}" fill="white"/>
''')

    # ── ENCODER blocks + arrows ─────────────────────────────────────────────
    enc_bot = []
    for i, (title, ops, out, pool, _) in enumerate(ENC):
        ps, bot = _block(enc_x[i], R1_TOP, title, ops, out, C_ENC, pool=pool)
        parts.extend(ps)
        enc_bot.append(bot)
        x_next = enc_x[i + 1] if i < 5 else CLS_X
        parts.append(_ah(enc_x[i] + BW, bn_arr_y, x_next))

    # ── BOTTLENECK ─────────────────────────────────────────────────────────
    parts.extend(bn_parts)

    # ── Bottleneck → Cls Head: long vertical arrow ───────────────────────────
    # Runs along the right side of the dual-heads column
    cls_arr_x = CLS_X + BW * 0.65
    parts.append(_av(cls_arr_x, bn_bot, R2_TOP))

    # ── CLS HEAD ────────────────────────────────────────────────────────────
    parts.extend(cls_parts)

    # ── DECODER blocks + arrows ─────────────────────────────────────────────
    for i, (title, ops, out) in enumerate(DEC):
        ps, _ = _block(dec_x[i], R2_TOP, title, ops, out, C_DEC)
        parts.extend(ps)
        if i < 4:
            parts.append(_ah(dec_x[i], dec_arr_y, dec_x[i + 1] + BW))

    # ── SEGMENTATION HEAD ──────────────────────────────────────────────────
    parts.append(_ah(dec_x[4], dec_arr_y, SEG_X + BW))
    parts.extend(seg_parts)

    # ── Bottleneck → Dec[0]: drawn LAST so arrowhead renders on top ──────────
    # Route (5 segments, arrives HORIZONTALLY at right edge of dec[0]):
    #   1. Drop 26 pt from bottleneck bottom  → y_jog
    #   2. Turn left (horizontal) → x_turn
    #   3. Continue down to dec_arr_y
    #   4. Final horizontal left into right edge of dec[0]  ← arriving segment
    bn_dec_x  = CLS_X + BW * 0.3             # departure x, left of cls_arr_x
    y_jog     = bn_bot + 26                  # drop 26 pt before first turn
    old_h_len = bn_dec_x - (CLS_X - 2)
    x_slip    = bn_dec_x - old_h_len * 1.5  # 1.5× longer horizontal
    x_turn    = x_slip + 8
    # arrowhead tip sits AH_LEN pt past the right edge of dec[0]
    tip_x = dec_x[0] + BW + 1               # arrowhead tip x (pointing left)
    shaft_xe = tip_x + AH_LEN               # polyline ends here (just before tip)
    parts.append(
        f'<polyline points="'
        f'{bn_dec_x:.1f},{bn_bot:.1f} '
        f'{bn_dec_x:.1f},{y_jog:.1f} '
        f'{x_turn:.1f},{y_jog:.1f} '
        f'{x_turn:.1f},{dec_arr_y:.1f} '
        f'{shaft_xe:.1f},{dec_arr_y:.1f}" '
        f'fill="none" stroke="{C_ARR}" stroke-width="0.8"/>'
    )
    parts.append(_arrowhead(tip_x, dec_arr_y, 'l', C_ARR))

    # ── INPUT box — below Enc Block 1, vertical upward arrow ───────────────
    inp_y = enc_bot[0] + 18
    parts += [_r(enc_x[0], inp_y, BW, IO_H, C_IO),
              _t(enc_x[0] + BW/2, inp_y + IO_H/2, "Input 224×224×3", size=FS_IO)]
    parts.append(_av(enc_x[0] + BW/2, inp_y, enc_bot[0]))

    # ── SKIP CONNECTIONS  enc[1..5] → dec[4..0] ────────────────────────────
    for ei, di in [(1, 4), (2, 3), (3, 2), (4, 1), (5, 0)]:
        sx = enc_x[ei] + BW / 2
        y1 = enc_bot[ei]
        y2 = R2_TOP
        # shaft stops AH_LEN_S pt before tip; arrowhead polygon draws the rest
        parts.append(
            f'<line x1="{sx:.1f}" y1="{y1:.1f}" '
            f'x2="{sx:.1f}" y2="{y2 - AH_LEN_S:.1f}" '
            f'stroke="{C_SKIP}" stroke-width="0.8" stroke-dasharray="3.5,2"/>'
        )
        parts.append(_arrowhead(sx, y2, 'd', C_SKIP, ln=AH_LEN_S, hw=AH_HW_S))
        lbl = ENC[ei][4]
        if lbl:
            parts.append(_t(sx + 5, (y1 + y2) / 2, lbl,
                            size=5.4, fill=C_DIM, anchor="start"))

    # ── SECTION LABELS — centred on the full canvas width ────────────────────
    label_cx = CW / 2
    parts.append(_t(label_cx, R1_TOP - 11, "PATEL CNN ENCODER",
                    size=FS_H, weight="bold", fill="#111111"))
    parts.append(_t(label_cx + 7, R2_TOP - 13, "U-NET DECODER",
                    size=FS_H, weight="bold", fill="#111111"))

    # ── SEPARATOR LINE ─────────────────────────────────────────────────────
    parts.append(
        f'<line x1="{MX:.1f}" y1="{sep_y:.1f}" '
        f'x2="{CW - MX:.1f}" y2="{sep_y:.1f}" '
        f'stroke="#BBBBBB" stroke-width="0.5"/>'
    )

    # ── HORIZONTAL LEGEND ─────────────────────────────────────────────────
    n      = len(LEG_ITEMS)
    slot_w = (CW - 2 * MX) / n
    sw_sz  = 8

    for k, (c, lbl) in enumerate(LEG_ITEMS):
        sx = MX + k * slot_w + slot_w * 0.18
        sy = leg_y - sw_sz / 2
        if c == "skip":
            parts.append(
                f'<line x1="{sx:.1f}" y1="{leg_y:.1f}" '
                f'x2="{sx + sw_sz:.1f}" y2="{leg_y:.1f}" '
                f'stroke="{C_SKIP}" stroke-width="0.9" stroke-dasharray="3,2"/>'
            )
        else:
            parts.append(_r(sx, sy, sw_sz, sw_sz, c, stroke="#888888", sw=0.4))
        parts.append(_t(sx + sw_sz + 3, leg_y, lbl,
                        size=5.8, anchor="start", fill="#333333"))

    parts.append("</svg>")
    return "\n".join(parts)


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    out = Path(__file__).parent.parent / "output" / "fig0_architecture_compact.svg"
    out.parent.mkdir(exist_ok=True)
    svg = build_svg()
    out.write_text(svg, encoding="utf-8")
    print(f"[saved] {out}")

    import re
    boxes     = len(re.findall(r"<rect ", svg))
    polygons  = len(re.findall(r"<polygon ", svg))
    m_w = re.search(r'width="([\d.]+in)"', svg)
    m_h = re.search(r'height="([\d.]+in)"', svg)
    if m_w and m_h:
        print(f"  boxes={boxes}  arrowhead-polygons={polygons}  "
              f"physical size: {m_w.group(1)} × {m_h.group(1)}")
