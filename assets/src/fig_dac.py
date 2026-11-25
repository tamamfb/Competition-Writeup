"""Gambar untuk writeup DAC Face Anti-Spoofing. Angka dari DAC-Face-Anti-Spoofing/README.md,
struktur model dari notebook.ipynb (GAC dan blok CLIP memakai pre-norm: LayerNorm → sublayer → +)."""
from pathlib import Path

import numpy as np

from paperfig import *

OUT = Path(__file__).resolve().parents[1] / "dac"


def caption_c(fig, s, fs=12.5, gap=0.04):
    """Seperti paperfig.caption, tetapi dipusatkan pada lebar semua sumbu beserta labelnya."""
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    bbs = [ax.get_tightbbox(r) for ax in fig.axes]
    x0, x1 = min(b.x0 for b in bbs), max(b.x1 for b in bbs)
    y0 = min(b.y0 for b in bbs)
    fig.text((x0 + x1) / 2 / fig.bbox.width, y0 / fig.bbox.height - gap, s, ha="center",
             va="top", fontsize=fs, family=SERIF)


def prenorm(d, cx, ln_y, w, text, fill, rx, fs=11, ln_h=1.4, main_h=2.2):
    """Satu sublayer pre-norm: LayerNorm → sublayer → (+), dengan jalur residual di sisi rx.

    ln_y adalah tepi bawah kotak LayerNorm. Panah masuk digambar oleh pemanggil.
    Mengembalikan (kotak sublayer, y pusat lingkaran +).
    """
    ln = d.box(cx - w / 2, ln_y, w, ln_h, "LayerNorm", YELLOW, fs=fs - 0.5)
    main = d.box(cx - w / 2, ln.y + ln_h + 0.7, w, main_h, text, fill, fs=fs)
    oy = main.y + main_h + 0.85
    r = 0.42
    d.oplus(cx, oy, r=r)
    d.arrow(ln.top, main.bottom)
    d.arrow(main.top, (cx, oy - r))
    yb = ln_y - 0.45
    side = r if rx > cx else -r
    d.arrow((cx, yb), (rx, yb), (rx, oy), (cx + side, oy))
    return main, oy


# --------------------------------------------------------------------------
# Figure 1: arsitektur
# --------------------------------------------------------------------------
def model():
    d = Diagram(43.5, 43.4, scale=0.21)
    fs = 11
    r = 0.42
    xl, xr, w = 12.5, 31.0, 10.0

    # ---- kolom kiri: CLIP ViT-L/14@336px ----
    d.text(xl, 0.9, "Gambar 336 × 336", fs=fs + 0.5)
    pe = d.box(xl - 5.5, 2.4, 11.0, 2.5, "Patch embedding\n14 × 14 + token CLS", PINK, fs=fs - 0.5)
    d.arrow((xl, 1.5), pe.bottom)
    pos_y = 6.1
    d.oplus(xl, pos_y, r=r)
    d.arrow(pe.top, (xl, pos_y - r))
    d.wave(4.6, pos_y)
    d.arrow((4.6 + 0.85, pos_y), (xl - r, pos_y))
    d.text(4.6, 4.95, "Positional\nembedding", fs=10, va="top")

    lc = d.container(5.6, 7.2, 13.8, 13.6)
    d.text(3.3, lc.cy + 1.3, "24×", fs=13)
    d.text(3.3, lc.cy - 1.0, "CLIP\nViT-L/14", fs=10, style="italic")
    rx_l = 6.6
    sa1, o1 = prenorm(d, xl, 8.2, w, "Multi-head\nself-attention", PEACH, rx_l, fs=fs)
    d.arrow((xl, pos_y + r), (xl, 8.2))
    ff1, o2 = prenorm(d, xl, o1 + r + 0.95, w, "Feed forward", BLUE, rx_l, fs=fs)
    d.arrow((xl, o1 + r), (xl, o1 + r + 0.95))

    # ---- kolom kanan: Enhanced GAC ----
    pj = d.box(24.0, 2.4, 6.0, 2.6, "Proyeksi\nlinear", LAVENDER, fs=fs)
    qq = d.box(32.0, 2.4, 6.0, 2.6, "96 query\ndipelajari", PINK, fs=fs)
    d.text(pj.cx, 1.45, "24 token CLS", fs=10)
    rc = d.container(23.6, 7.0, 15.8, 20.1)
    d.text(41.5, rc.cy + 1.3, "4×", fs=13)
    d.text(41.5, rc.cy - 1.0, "Enhanced\nGAC", fs=10, style="italic")
    rx_r = 37.7
    merge = 6.2
    ln0 = 8.1
    d.arrow(pj.top, (pj.cx, merge), (xr, merge), (xr, ln0))
    d.arrow(qq.top, (qq.cx, merge), (xr, merge), (xr, ln0))
    sa, g1 = prenorm(d, xr, ln0, w, "Multi-head\nself-attention", PEACH, rx_r, fs=fs)
    ln1 = g1 + r + 0.95
    d.arrow((xr, g1 + r), (xr, ln1))
    ca, g2 = prenorm(d, xr, ln1, w, "Cross-attention\nke token patch", PEACH, rx_r, fs=fs)
    ln2 = g2 + r + 0.95
    d.arrow((xr, g2 + r), (xr, ln2))
    ff, g3 = prenorm(d, xr, ln2, w, "Feed forward", BLUE, rx_r, fs=fs)

    # ---- sambungan antar kolom ----
    split = 22.2
    # token CLS dari setiap blok (forward hook) → proyeksi linear
    d.arrow((xl + r, o2), (20.7, o2), (20.7, pj.cy), pj.left, ls=(0, (4, 2)))
    # 576 token patch → cross-attention
    d.arrow((xl, o2 + r), (xl, split), (22.0, split), (22.0, ca.cy), ca.left)
    d.text(17.0, split + 0.75, "576 token patch", fs=10)
    # CLS akhir → fusi
    fuse_y = 30.5
    d.arrow((xl, o2 + r), (xl, split), (9.0, split), (9.0, fuse_y), (xr - r, fuse_y))
    d.text(8.5, 26.3, "CLS\nakhir", fs=10, ha="right")

    # ---- classifier ----
    pool = d.box(xr - w / 2, 28.0, w, 1.4, "Rata-rata token", YELLOW, fs=fs - 0.5)
    d.arrow((xr, g3 + r), pool.bottom)
    d.oplus(xr, fuse_y, r=r)
    d.arrow(pool.top, (xr, fuse_y - r))
    ln = d.box(xr - w / 2, 31.6, w, 1.4, "LayerNorm", YELLOW, fs=fs - 0.5)
    d.arrow((xr, fuse_y + r), ln.bottom)

    by, bh, br = 34.6, 2.5, 33.8
    mlp = d.box(23.1, by, 9.0, bh, "MLP\n1024 → 512 → 256 → 6", BLUE, fs=fs - 0.5)
    sc = d.box(33.2, by, 5.2, bh, "Linear\n1024 → 6", LAVENDER, fs=fs - 0.5)
    pr = d.box(xl - 5.5, by, 11.0, bh, "Head proyeksi\n256-d, norm L2", LAVENDER, fs=fs)
    for b in (mlp, sc, pr):
        d.arrow(ln.top, (xr, br), (b.cx, br), b.bottom)
    out_y = 38.3
    d.oplus(xr, out_y, r=r)
    d.arrow(mlp.top, (mlp.cx, out_y), (xr - r, out_y))
    d.arrow(sc.top, (sc.cx, out_y), (xr + r, out_y))
    sm = d.box(xr - w / 2, 39.4, w, 1.4, "Softmax", GREEN, fs=fs - 0.5)
    d.arrow((xr, out_y + r), sm.bottom)
    d.arrow(sm.top, (xr, 41.9))
    d.text(xr, 42.6, "6 kelas (Focal + CE berbobot)", fs=fs)
    d.arrow(pr.top, (xl, 38.6))
    d.text(xl, 39.3, "SupCon ($\\tau$ = 0,07)", fs=fs)

    d.caption("Figure 1: Arsitektur model. Token CLS dari 24 blok CLIP ViT-L/14 dan 96 query "
              "dipelajari\ndiagregasi oleh 4 lapis Enhanced GAC yang melakukan cross-attention "
              "ke 576 token patch.", y=-0.4, fs=14)
    d.save(OUT / "fig1_model.png")


# --------------------------------------------------------------------------
# Figure 2: macro-F1 validasi per versi
# --------------------------------------------------------------------------
def versions():
    vers = ["v1\n224 px\nbackbone beku", "v2\n336 px\nGAC 2 lapis", "v3\nSupCon\n3 fase",
            "v4\n224 px\nzero-shot + kNN", "v4b\nGAC 3 lapis\nGPU cloud", "v5 (final)\nGAC 4 lapis\npseudo-label"]
    f1 = [0.9944, 0.9926, 0.9874, 0.9833, 0.9948, 0.9888]
    inside = ["n = 330", "n = 248", "hanya P1", "OOM kNN", "n = 165", "n = 165"]
    crashed = [False, False, True, True, False, False]
    fills = [BLUE, BLUE, BLUE, BLUE, BLUE, PEACH]

    fig, ax = new_chart(6.8, 3.6)
    x = np.arange(len(vers))
    for xi, v, c, bad in zip(x, f1, fills, crashed):
        ax.bar(xi, v, 0.62, color=c, edgecolor="black", lw=1.0, zorder=3,
               hatch="///" if bad else None)
    for xi, v, s, bad in zip(x, f1, inside, crashed):
        ax.text(xi, v + 0.0004, f"{v:.4f}", ha="center", va="bottom", fontsize=9.5, zorder=4)
        ax.text(xi, 0.9807, s, ha="center", va="bottom", fontsize=8.5, zorder=4,
                bbox=dict(boxstyle="square,pad=0.15", fc="white", ec="none") if bad else None)
    ax.set_xticks(x)
    ax.set_xticklabels(vers, fontsize=9)
    ax.set_ylabel("Macro-F1 validasi")
    ax.set_ylim(0.98, 1.0)
    ax.set_xlim(-0.55, len(vers) - 0.45)
    ax.grid(axis="x", visible=False)
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(fc=BLUE, ec="black", label="run selesai"),
                       Patch(fc=BLUE, ec="black", hatch="///", label="run terhenti (kehabisan memori)"),
                       Patch(fc=PEACH, ec="black", label="model final")],
              loc="upper right", fontsize=9, ncol=3, handlelength=1.6, columnspacing=1.0)
    caption_c(fig, "Figure 2: Macro-F1 validasi per versi. Ukuran split validasi berbeda antarversi,\n"
                   "dan pada 165 gambar satu gambar setara dengan sekitar 0,006.")
    save_chart(fig, OUT / "fig2_versions.png")


if __name__ == "__main__":
    model()
    versions()
