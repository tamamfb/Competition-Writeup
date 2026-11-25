"""Gambar untuk writeup ROGII Wellbore Geology.

Angka diambil dari ROGII-Wellbore-Geology/README.md. Gambar contoh sumur memakai
sample_wells_predictions.csv (prediksi tiap estimator pada 3 sumur contoh, dihitung
dengan sumur itu dikeluarkan dari semua aset).

Urutan nomor Figure mengikuti urutan gambar muncul di README: spatial CV (Figure 1),
pipeline (Figure 2), contoh sumur (Figure 3).
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from paperfig import *

OUT = Path(__file__).resolve().parents[1] / "rogii"
CSV = Path(r"C:\Users\tamam\AppData\Local\Temp\claude\8c2c7c9dfeb40828"
           r"\508159f6-00e3-4e2c-b47a-40847ebb475a\scratchpad\rogii\sample_wells_predictions.csv")
WELL = "00bbac68"


def caption_mid(fig, s, fs=12.5, gap=0.04):
    """Seperti paperfig.caption, tetapi titik tengahnya mengikuti lebar isi gambar
    (sumbu + label), bukan lebar kanvas, supaya tetap di tengah setelah bbox_inches='tight'."""
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    bbs = [ax.get_tightbbox(r) for ax in fig.axes]
    x0, x1 = min(b.x0 for b in bbs), max(b.x1 for b in bbs)
    y0 = min(b.y0 for b in bbs)
    fig.text((x0 + x1) / 2 / fig.bbox.width, y0 / fig.bbox.height - gap, s,
             ha="center", va="top", fontsize=fs, family=SERIF)


def pipeline():
    d = Diagram(48.0, 33.4, scale=0.21)
    fs = 11

    # ---- kolom kanan: alur per sumur test, dari bawah ke atas ----
    test = d.box(24.5, 0.6, 12.0, 3.6, "Sumur test\nMD, X, Y, Z, GR, TVT s.d. PS\n+ typewell (TVT, GR)",
                 PINK, fs=fs - 0.5)
    look = d.box(24.5, 6.0, 12.0, 2.8, "Lookup fingerprint", YELLOW, fs=fs)
    d.arrow(test.top, look.bottom)

    copy = d.box(39.8, 6.0, 7.6, 2.8, "Salin TVT\ntrain", GREEN, fs=fs - 0.5)
    d.arrow(look.right, copy.left)
    d.text((look.x + look.w + copy.x) / 2, look.cy + 0.35, "cocok", fs=9.5, va="bottom", style="italic")

    ec = d.container(18.9, 11.0, 23.6, 6.8)
    ests = [("nndrift\ndrift U tetangga\n(radius 1.500 ft)", 19.6),
            ("struct-kNN\nbatas formasi\nIDW 25 titik", 27.3),
            ("struct-global\nbatas formasi\ndari LightGBM", 35.0)]
    eboxes = [d.box(x, 11.7, 6.8, 4.4, t, PEACH, fs=fs - 1) for t, x in ests]
    d.arrow(look.top, (look.cx, ec.y))
    d.text(look.cx - 0.5, (look.y + look.h + ec.y) / 2, "tidak cocok", fs=9.5, ha="right", style="italic")

    blend = d.box(18.9, 20.2, 23.6, 2.8, "Blend linear", YELLOW, fs=fs)
    for b, wgt in zip(eboxes, ["0,5", "0,25", "0,25"]):
        d.arrow(b.top, (b.cx, blend.y))
        d.text(b.cx + 0.45, (ec.y + ec.h + blend.y) / 2, f"× {wgt}", fs=9.5, ha="left", style="italic")

    fb = d.box(23.7, 24.9, 13.6, 3.4, "Koreksi forward-backward\n(posterior mean)", YELLOW, fs=fs - 0.5)
    d.arrow(blend.top, (blend.cx, fb.y))
    out = d.box(25.5, 30.2, 10.0, 2.8, "Prediksi TVT", GREEN, fs=fs)
    d.arrow((fb.cx, fb.y + fb.h), (fb.cx, out.y))
    # jalur sumur yang sudah ada di train
    jx = copy.x + copy.w * 0.75
    d.arrow((jx, copy.y + copy.h), (jx, out.cy), out.right)

    # ---- kolom kiri: aset global dari sumur train ----
    train = d.box(2.6, 0.6, 9.8, 3.6, "Sumur train\n(773 sumur)", PINK, fs=fs)
    ac = d.container(0.5, 5.3, 14.0, 16.7)
    d.arrow(train.top, (train.cx, ac.y))
    ax_, aw = 1.2, 12.6
    fp = d.box(ax_, look.y, aw, look.h, "Fingerprint MD5\nlintasan sumur", YELLOW, fs=fs - 0.5)
    udb = d.box(ax_, 10.7, aw, 3.0, "U-database (U = TVT + Z)\nKD-tree", BLUE, fs=fs - 0.5)
    mdb = d.box(ax_, 14.5, aw, 3.0, "Batas formasi: KD-tree\n+ 6 LightGBM (X, Y)", BLUE, fs=fs - 0.5)
    disc = d.box(ax_, 18.3, aw, 3.0, "Discriminator GR\n(LightGBM, 20 fitur)", LAVENDER, fs=fs - 0.5)
    d.text(ac.cx, ac.y + ac.h + 0.75, "Aset global (dibangun dari train)", fs=fs, style="italic")

    d.arrow(fp.right, look.left)
    d.arrow(udb.right, (ec.x, udb.cy))
    d.arrow(mdb.right, (ec.x, mdb.cy))
    kx = 16.7
    d.arrow(disc.right, (kx, disc.cy), (kx, fb.cy), fb.left)
    d.text((kx + fb.x) / 2, fb.cy + 0.35, "emisi", fs=9.5, va="bottom", style="italic")

    d.caption("Figure 2: Pipeline v4. Aset global dibangun dari sumur train; setiap sumur test melewati lookup\n"
              "fingerprint, tiga estimator struktural, blend, lalu koreksi forward-backward.", y=-0.3, fs=14)
    d.save(OUT / "fig2_pipeline.png")


def spatial_cv():
    groups = ["Sumur itu sendiri", "Seluruh cluster"]
    methods = ["Konstan", "nndrift", "struct-kNN", "Blend"]
    vals = {"Konstan": [18.74, 18.74], "nndrift": [15.97, 18.70],
            "struct-kNN": [17.21, 18.68], "Blend": [14.46, 18.67]}
    fills = [LAVENDER, BLUE, PEACH, GREEN]
    fig, ax = new_chart(6.0, 3.4)
    x = np.arange(len(groups))
    w = 0.19
    for j, (m, c) in enumerate(zip(methods, fills)):
        xx = x + (j - 1.5) * w
        ax.bar(xx, vals[m], w, color=c, edgecolor="black", lw=1, label=m, zorder=3)
        for xi, v in zip(xx, vals[m]):
            ax.text(xi, v + 0.3, f"{v:.2f}", ha="center", va="bottom", fontsize=8.5)
    ax.set_xticks(x)
    ax.set_xticklabels(groups)
    ax.set_xlabel("Data yang dikeluarkan dari aset")
    ax.set_ylabel("RMSE (ft), lebih kecil lebih baik")
    ax.set_ylim(0, 24)
    ax.set_xlim(-0.55, 1.55)
    ax.grid(axis="x", visible=False)
    ax.tick_params(axis="x", length=0)
    ax.legend(loc="upper center", ncol=4, fontsize=9.5, columnspacing=1.2, handlelength=1.6)
    caption_mid(fig, "Figure 1: RMSE pada 144 sumur yang sama. Saat seluruh cluster spasial dikeluarkan,\n"
                     "semua metode turun ke level prediksi konstan.")
    save_chart(fig, OUT / "fig1_spatial_cv.png")


def example_well():
    df = pd.read_csv(CSV)
    w = df[df.wid == WELL].reset_index(drop=True)
    pred = w.TVT_input.isna()
    ps = int(pred.idxmax())
    md_ps = w.MD[ps]
    view = w.MD >= md_ps - 500

    def rmse(col):
        return float(np.sqrt(((w.loc[pred, col] - w.loc[pred, "TVT_true"]) ** 2).mean()))

    fig, (ax, gx) = new_chart(6.8, 5.0, nrows=2, sharex=True,
                              gridspec_kw={"height_ratios": [3, 1], "hspace": 0.08})
    v = w[view]
    p = w[pred]
    ax.axvspan(md_ps, v.MD.max(), color=GREY, zorder=0)
    ax.plot(v.MD, v.TVT_true, color="black", lw=1.8, label="TVT sebenarnya", zorder=5)
    # konstan digambar paling atas (putus-putus) karena nndrift jatuh ke nilai konstan
    # saat tidak ada tetangga, sehingga kedua garis berimpit di bagian akhir lateral
    series = [("const", "Konstan", DARK["grey"], (0, (5, 3)), 1.6, 4.6),
              ("nndrift", "nndrift", DARK["blue"], "-", 1.4, 4.2),
              ("struct_knn", "struct-kNN", DARK["peach"], "-", 1.4, 4.1),
              ("final_v4", "Final v4", DARK["pink"], "-", 2.1, 4.4)]
    for col, name, c, ls, lw, z in series:
        ax.plot(p.MD, p[col], color=c, ls=ls, lw=lw, zorder=z, label=f"{name} (RMSE {rmse(col):.2f})")
    ax.invert_yaxis()
    ax.set_ylabel("TVT (ft)")
    lo, hi = ax.get_ylim()
    ax.axvline(md_ps, color="black", lw=1.2, ls=(0, (4, 2)), zorder=6)
    ax.text(md_ps + 60, hi + (lo - hi) * 0.04, "Prediction Start", ha="left", va="top", fontsize=10,
            style="italic")
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.03), ncol=3, fontsize=9,
              columnspacing=1.2, handlelength=2.2)

    gx.plot(v.MD, v.GR, color=DARK["green"], lw=0.5)
    gx.axvspan(md_ps, v.MD.max(), color=GREY, zorder=0)
    gx.axvline(md_ps, color="black", lw=1.2, ls=(0, (4, 2)))
    gx.set_ylabel("GR")
    gx.set_xlabel("MD (ft)")
    gx.set_xlim(v.MD.min(), v.MD.max())
    caption_mid(fig, f"Figure 3: Prediksi tiap estimator pada sumur {WELL} (sumur itu dikeluarkan dari semua aset).\n"
                     "Area abu-abu adalah bagian yang diprediksi; sumbu TVT dibalik (ke bawah makin dalam).")
    save_chart(fig, OUT / "fig3_example_well.png")


if __name__ == "__main__":
    pipeline()
    spatial_cv()
    example_well()
