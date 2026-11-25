"""Gambar untuk writeup Data Slayer 4.0. Angka diambil dari Dataslayer-4.0/README.md."""
from pathlib import Path

import numpy as np

from paperfig import *

OUT = Path(__file__).resolve().parents[1] / "dataslayer"


def caption_c(fig, s, fs=12.5):
    """Seperti paperfig.caption, tetapi di tengah seluruh isi gambar (termasuk label sumbu)."""
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    bb = fig.get_tightbbox(r)
    cx = (bb.x0 + bb.x1) / 2 / fig.get_figwidth()
    bottom = min(ax.get_tightbbox(r).y0 for ax in fig.axes) / fig.bbox.height - 0.04
    fig.text(cx, bottom, s, ha="center", va="top", fontsize=fs, family=SERIF)


def pipeline():
    d = Diagram(52.8, 33.2, scale=0.2)
    fs = 10.5

    # kolom
    xA, wA = 0.4, 6.6
    xB, wB = 8.8, 7.0
    xC, wC = 17.6, 8.4
    xD, wD = 28.0, 7.0
    xE, wE = 37.4, 6.4
    xF, wF = 45.4, 7.2
    y1 = 29.0        # baris struktur test
    ym = 14.8        # garis tengah baris model

    # ---- baris 1: memulihkan struktur test
    test = d.box(xA, y1 - 2.2, wA, 4.4, "1.515 klip test\n(1 detik, diacak)", PINK, fs=fs)
    cam = d.box(xB, y1 - 2.4, wB, 4.8, "Klasifikasi\nkamera\ndepan / samping", YELLOW, fs=fs)
    sti = d.box(xC, y1 - 2.4, wC, 4.8, "Stitching\nHungarian penerus\nτ = 6× noise\n→ 96 video", PEACH, fs=fs)
    grp = d.box(xD, y1 - 2.4, wD + 1.4, 4.8, "Grouping\nWard + Hungarian\nberkapasitas\n→ 12 grup × 8 video",
                YELLOW, fs=fs - 0.5)
    d.arrow(test.right, cam.left)
    d.arrow(cam.right, sti.left)
    d.arrow(sti.right, grp.left)

    # ---- baris 2: frame train, fitur, head
    train = d.box(xA, ym - 2.2, wA, 4.4, "416 video train\n(26 subjek)", PINK, fs=fs)
    par = d.box(xB, ym - 2.4, wB, 4.8, "Frame parity\nOpenCV 640×360\nJPEG q95", YELLOW, fs=fs)
    d.arrow(train.right, par.left)

    fc = d.container(xC, ym - 8.0, wC, 16.0)
    sA = d.box(xC + 0.6, ym + 1.6, wC - 1.2, 4.6, "Stream A\nDINOv2 ViT-L\n10 fps, 29 token", BLUE, fs=fs)
    sB = d.box(xC + 0.6, ym - 6.2, wC - 1.2, 4.6, "Stream B\nV-JEPA 2 ViT-L\n8 irisan × 17 token", BLUE, fs=fs)
    d.text(fc.cx, fc.y - 0.5, "Fitur beku", fs=fs, va="top", style="italic")
    d.arrow(par.right, (fc.x, ym))
    d.arrow(sti.bottom, (sti.cx, fc.y + fc.h))

    heads = []
    for s, lab_dy in ((sA, 1), (sB, -1)):
        c = d.container(xD, s.cy - 2.4, wD, 4.8)
        h = d.box(xD + 0.6, s.cy - 1.8, wD - 1.2, 3.6, "Transformer\ntemporal\nd = 256", LAVENDER, fs=fs)
        d.text(c.cx, (c.y + c.h + 0.8) if lab_dy > 0 else (c.y - 0.8), "2×", fs=13)
        d.arrow(s.right, h.left)
        heads.append(h)

    gm = d.box(xE, ym - 2.3, wE, 4.6, "Rata-rata\ngeometrik:\nprob. 8 perilaku", YELLOW, fs=fs)
    d.arrow(heads[0].right, (gm.cx, heads[0].cy), gm.top)
    d.arrow(heads[1].right, (gm.cx, heads[1].cy), gm.bottom)

    # ---- kolom kanan: assignment, onset, submission
    s2 = d.box(xF, ym - 2.4, wF, 4.8, "S2: Hungarian\nper grup,\n8 perilaku ↔\n8 video", PEACH, fs=fs)
    d.arrow(gm.right, s2.left)
    d.arrow(grp.right, (s2.cx, y1), s2.top)
    ons = d.box(xF, s2.y - 6.0, wF, 4.4, "Aturan onset:\ndetik ≥ 7 di video\ncurang = 1", YELLOW, fs=fs)
    sub = d.box(xF, ons.y - 4.3, wF, 2.8, "submission.csv", GREEN, fs=fs + 0.5)
    d.arrow(s2.bottom, ons.top)
    d.arrow(ons.bottom, sub.top)

    d.caption("Figure 1: Pipeline Data Slayer 4.0. Klip test disusun ulang menjadi video dan grup (kamera, subjek);\n"
              "Hungarian per grup membagi 8 perilaku ke 8 video, lalu aturan onset memilih detik berlabel curang.",
              y=0.6, fs=14)
    d.save(OUT / "fig1_pipeline.png")


def tiers():
    tiers_ = ["S0\nper klip", "S1\nper video", "S2\nHungarian per grup\n(dipakai)", "S3\nS2 + fusi kamera"]
    vals = {
        "A (DINOv2)": [0.5678, 0.9418, 0.9808, 0.9853],
        "B (V-JEPA 2)": [0.6212, 0.9028, 0.9734, 0.9783],
        "A + B": [0.6086, 0.9627, 0.9947, 0.9898],
    }
    fills = [BLUE, PEACH, GREEN]
    fig, ax = new_chart(7.4, 3.6)
    x = np.arange(len(tiers_))
    w = 0.26
    for i, ((name, v), c) in enumerate(zip(vals.items(), fills)):
        xs = x + (i - 1) * w
        ax.bar(xs, v, w, color=c, edgecolor="black", lw=1, label=name, zorder=3)
        for xi, vi in zip(xs, v):
            bold = name == "A + B" and vi == 0.9947
            ax.text(xi, vi + 0.006, f"{vi:.4f}", ha="center", va="bottom", fontsize=8,
                    fontweight="bold" if bold else "normal")
    ax.set_xticks(x)
    ax.set_xticklabels(tiers_, fontsize=10)
    ax.set_ylabel("Macro-F1 CV (level video)")
    ax.set_ylim(0.5, 1.04)
    ax.grid(axis="x", visible=False)
    ax.legend(loc="upper left")
    caption_c(fig, "Figure 2: Macro-F1 CV per tingkat keputusan untuk tiap stream dan gabungannya.")
    save_chart(fig, OUT / "fig2_tiers.png")


def onset():
    sec = np.arange(1, 14)
    cnt = [6, 7, 4, 1, 10, 27, 152, 114, 19, 11, 7, 3, 3]
    rules = ["Semua detik (S2)", "Onset per video", "Posisi ≥ 8", r"Posisi ≥ $\lfloor n/2 \rfloor$",
             r"Posisi ≥ $\lfloor (n-1)/2 \rfloor$", "Posisi ≥ 7"]
    lb = [0.7507271, 0.9700344, 0.9672316, 0.9783611, 0.9919781, 1.0000000]

    fig, (a1, a2) = new_chart(8.6, 3.4, ncols=2, gridspec_kw={"width_ratios": [1.15, 1], "wspace": 0.55})
    fills = [PEACH if s == 7 else BLUE for s in sec]
    a1.bar(sec, cnt, 0.72, color=fills, edgecolor="black", lw=1, zorder=3)
    for s, c in zip(sec, cnt):
        a1.text(s, c + 2.5, str(c), ha="center", va="bottom", fontsize=8.5)
    a1.annotate("modus = 7\n(152 dari 364)", (6.64, 125), textcoords="offset points", xytext=(-58, 0),
                ha="center", va="center", fontsize=9.5,
                arrowprops=dict(arrowstyle="-|>", lw=0.9, color="black", shrinkA=2, shrinkB=3))
    a1.set_xticks(sec)
    a1.set_xlim(0.3, 13.7)
    a1.set_ylim(0, 172)
    a1.grid(axis="x", visible=False)
    a1.set_xlabel("Detik mulai aksi (floor) di train")
    a1.set_ylabel("Jumlah video")
    a1.set_title("(a) Onset aksi di 364 video train", fontsize=11)

    hbar(a2, rules, lb, highlight=len(rules) - 1, fmt="{:.4f}", xlim=(0.70, 1.06), fontsize=9,
         xlabel="Public LB (macro-F1)")
    a2.tick_params(axis="y", labelsize=9.5)
    a2.set_title("(b) Aturan label di public LB", fontsize=11)

    caption_c(fig, "Figure 3: Waktu mulai aksi di train berpusat di detik ke-7; aturan posisi ≥ 7 memberi "
                   "public LB tertinggi.")
    save_chart(fig, OUT / "fig3_onset.png")


if __name__ == "__main__":
    pipeline()
    tiers()
    onset()
