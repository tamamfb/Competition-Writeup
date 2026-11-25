"""Gambar untuk writeup Name That LLM. Semua angka diambil dari Name-That-LLM/README.md."""
from pathlib import Path

import numpy as np

from paperfig import *

OUT = Path(__file__).resolve().parents[1] / "name-that-llm"


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
    d = Diagram(47.5, 28.5, scale=0.21)
    fs = 11

    # input
    inp = d.box(0.5, 8.0, 6.0, 4.4, "Teks test\n(prompt baru)", PINK, fs=fs + 0.5)
    d.text(3.5, 6.9, "≈ 4.000 teks\n40–250 kata", fs=9.5, va="top")

    # rekonstruksi grup prompt (atas)
    cl = d.container(9.0, 21.4, 23.0, 4.6)
    bw, bh, by = 6.2, 3.2, 22.1
    xs = [9.7, 9.7 + bw + 1.5, 9.7 + 2 * (bw + 1.5)]
    c1 = d.box(xs[0], by, bw, bh, "Cosine TF-IDF\nkata + char 4–5", BLUE, fs=fs - 0.5)
    c2 = d.box(xs[1], by, bw, bh, "Average linkage\npotong di 0,94", YELLOW, fs=fs - 0.5)
    c3 = d.box(xs[2], by, bw, bh, "Pecah cluster\n> 18 teks", YELLOW, fs=fs - 0.5)
    d.arrow(c1.right, c2.left)
    d.arrow(c2.right, c3.left)
    d.text(cl.cx, 27.2, "Rekonstruksi grup prompt (clustering)", fs=fs, style="italic")

    # tiga keluarga model unary
    mc = d.container(9.0, 1.6, 23.0, 15.8)
    rows = [12.9, 7.9, 2.9]
    rh = 3.8
    lx, lw_ = 9.7, 10.0
    rx, rw = 21.4, 9.9
    left = [("TF-IDF char 1–5,\nkata 1–2 + 442\nfitur stilometri", BLUE),
            ("Qwen3 0.6B / 1.7B / 4B\nBase, beku\n(mean-pool lapisan)", PEACH),
            ("DeBERTa-v3-large\nfine-tune sadar grup\n(2 grup per batch)", PEACH)]
    right = ["Kernel centering\n→ logistic regression\n(+ 0,2 × NB)",
             "Prompt-centering\n→ PCA 1024\n→ LDA shrinkage",
             "Head ter-center\n+ head raw\n(+ column loss)"]
    for (lt, lc), rt, y in zip(left, right, rows):
        a = d.box(lx, y, lw_, rh, lt, lc, fs=fs - 0.5)
        b = d.box(rx, y, rw, rh, rt, LAVENDER, fs=fs - 0.5)
        d.arrow(a.right, b.left)
    d.text(mc.cx, 0.75, "3 keluarga model unary", fs=fs, style="italic")

    # input ke clustering dan ke model
    trunk = 7.8
    d.arrow(inp.right, (trunk, inp.cy), (trunk, c1.cy), (cl.x, c1.cy))
    d.arrow(inp.right, (trunk, inp.cy), (mc.x, inp.cy))

    # ID grup turun ke centering
    gx = rx + rw / 2
    d.arrow((gx, cl.y), (gx, mc.y + mc.h), ls=(0, (4, 2)))
    d.text(gx + 0.6, (cl.y + mc.y + mc.h) / 2, "ID grup", fs=9.5, ha="left", style="italic")

    # ensemble + decoding: kolom kanan, alur dari bawah ke atas
    ex, ew, eh, gap = 34.6, 10.4, 2.9, 1.15
    names = [("Geometric mean\nlog-prob (bobot sama)", YELLOW),
             ("Campur ter-center & raw\nuntuk cluster kecil", YELLOW),
             ("Sinkhorn per cluster\n(tiap model ≤ 1 kali)", YELLOW),
             ("Mean-field pasangan\n(15 iterasi)", YELLOW),
             ("Argmax marginal", YELLOW),
             ("submission.parquet", GREEN)]
    col = []
    for i, (t, c) in enumerate(names):
        col.append(d.box(ex, 1.6 + i * (eh + gap), ew, eh, t, c, fs=fs - 0.5))
    for a, b in zip(col[:-1], col[1:]):
        d.arrow(a.top, b.bottom)
    d.arrow((mc.x + mc.w, col[0].cy), col[0].left)

    # ID grup juga dipakai decoding
    jx = 33.3
    d.arrow((cl.x + cl.w, c3.cy), (jx, c3.cy), (jx, col[2].cy), col[2].left, ls=(0, (4, 2)))

    # tanda kurung samping: Ensemble / Decoding per grup
    bx = ex + ew + 0.5

    def bracket(y0, y1, label):
        d.line((bx, y0), (bx + 0.35, y0), (bx + 0.35, y1), (bx, y1), radius=0.2, lw=1.4)
        d.text(bx + 0.95, (y0 + y1) / 2, label, fs=10, rotation=90, style="italic")

    bracket(col[0].y, col[1].y + eh, "Ensemble")
    bracket(col[2].y, col[4].y + eh, "Decoding per grup")

    d.caption("Figure 1: Pipeline solusi Name That LLM. Grup prompt di test direkonstruksi lewat clustering,\n"
              "lalu prediksi tiga keluarga model di-ensemble dan di-decode bersama per grup.", y=-0.3, fs=14)
    d.save(OUT / "fig1_pipeline.png")


def components():
    comps = ["Baseline TF-IDF LR", "TF-IDF + stilometri", "DeBERTa-v3-large", "Qwen3-0.6B",
             "Qwen3-1.7B", "Qwen3-4B\n(LDA hierarkis)", "Ensemble final\n(8 komponen)"]
    # angka persis seperti di tabel README (string dipakai untuk label agar digitnya sama)
    table = {
        "Argmax": ["0.3519", "0.4734", "0.4583", "0.5627", "0.5750", "0.5976", "0.619"],
        "Hungarian": ["0.4664", "0.5282", "0.5353", "0.6512", "0.6720", "0.6927", "0.7161"],
        "Sinkhorn + pair": ["0.5135", "0.5835", "0.5802", "0.6809", "0.6972", "0.7073", "0.7365"],
    }
    fills = [BLUE, PEACH, GREEN]
    fig, ax = new_chart(6.6, 5.6)
    n = len(comps)
    y = np.arange(n)[::-1] * 1.0
    h = 0.27
    for j, ((name, vals), c) in enumerate(zip(table.items(), fills)):
        v = [float(s) for s in vals]
        yy = y + (1 - j) * h
        ax.barh(yy, v, h, color=c, edgecolor="black", lw=1.0, label=name, zorder=3)
        for yi, vi, s in zip(yy, v, vals):
            bold = yi > y[-1] - 0.5 and yi < y[-1] + 0.5
            ax.text(vi + 0.006, yi, s, va="center", ha="left", fontsize=8.5,
                    fontweight="bold" if bold else "normal", zorder=4)
    ax.set_yticks(y)
    ax.set_yticklabels(comps)
    ax.get_yticklabels()[-1].set_fontweight("bold")
    ax.tick_params(axis="y", length=0)
    ax.grid(axis="y", visible=False)
    ax.set_xlim(0.3, 0.80)
    ax.set_ylim(-0.6, n - 0.4)
    ax.set_xlabel("Balanced accuracy CV (lebih besar lebih baik)")
    ax.legend(loc="upper right", title="Decoding", title_fontsize=10)
    fig.subplots_adjust(left=0.27, right=0.97)
    caption_mid(fig, "Figure 2: Balanced accuracy CV tiap komponen dengan tiga cara decoding.")
    save_chart(fig, OUT / "fig2_components.png")


def group_size():
    ks = [2, 3, 4, 6, 9, 18]
    series = [
        ("Argmax", [0.465, 0.466, 0.467, 0.465, 0.467, 0.460], "-o", DARK["blue"], BLUE),
        ("Eksklusi ala notebook publik 0.522", [0.480, 0.483, 0.490, 0.487, 0.471, 0.460], "--s",
         DARK["peach"], PEACH),
        ("Centering kNN (K = 17, cos > 0,88, a = 0,5)", [0.470, 0.494, 0.509, 0.511, 0.509, 0.522], "-^",
         DARK["green"], GREEN),
    ]
    # posisi label (offset titik) per seri per k, diatur manual agar tidak bertumpuk
    offs = [
        [(0, -13), (0, -13), (0, -13), (0, -13), (0, -13), (0, -13)],
        [(0, 6), (0, -13), (0, -13), (0, -13), (0, 6), None],
        [(-8, -3), (0, 6), (0, 6), (0, 6), (0, 6), (0, 6)],
    ]
    fig, (top, ax) = new_chart(6.6, 4.4, nrows=2, sharex=True,
                               gridspec_kw={"height_ratios": [1, 3.2], "hspace": 0.06})
    x = np.arange(len(ks))
    for (name, v, st, lc, fc), off in zip(series, offs):
        ax.plot(x, v, st, color=lc, lw=1.7, ms=6.5, mfc=fc, mec="black", label=name, zorder=3)
        for xi, vi, o in zip(x, v, off):
            if o is None:
                continue
            ha = "right" if o[0] < 0 else "center"
            ax.annotate(f"{vi:.3f}", (xi, vi), textcoords="offset points", xytext=o, ha=ha,
                        va="bottom" if o[1] >= 0 else "top" if o[1] < -4 else "center", fontsize=8.5)

    ax.axhline(0.434, color="black", lw=1.3, ls=(0, (5, 3)), zorder=2)
    ax.text(x[-1] + 0.35, 0.434 + 0.002, "Public LB 0.434", ha="right", va="bottom", fontsize=9.5)
    top.axhline(0.7365, color="black", lw=1.3, ls=(0, (5, 3)), zorder=2)
    top.text(x[-1] + 0.35, 0.7365 + 0.003, "CV grup lengkap 0.7365", ha="right", va="bottom", fontsize=9.5)

    ax.set_ylim(0.425, 0.532)
    top.set_ylim(0.722, 0.752)
    top.set_yticks([0.73, 0.74, 0.75])
    ax.set_xlim(-0.55, len(ks) - 0.45)
    ax.set_xticks(x)
    ax.set_xticklabels([str(k) for k in ks])
    ax.set_xlabel("Teks per prompt (k)")
    ax.set_ylabel("Balanced accuracy")
    ax.yaxis.set_label_coords(-0.085, 0.67)

    # sumbu terputus
    top.spines["bottom"].set_visible(False)
    ax.spines["top"].set_visible(False)
    top.tick_params(axis="x", which="both", bottom=False, labelbottom=False)
    ax.tick_params(axis="x", which="both", top=False)
    kw = dict(color="black", clip_on=False, lw=1.0)
    dx, dy = 0.012, 0.03
    for xx in (0, 1):
        top.plot([xx - dx, xx + dx], [-dy * 3.2, dy * 3.2], transform=top.transAxes, **kw)
        ax.plot([xx - dx, xx + dx], [1 - dy, 1 + dy], transform=ax.transAxes, **kw)

    handles, labels = ax.get_legend_handles_labels()
    order = [0, 2, 1]
    top.legend([handles[i] for i in order], [labels[i] for i in order], loc="lower center",
               bbox_to_anchor=(0.5, 1.06), ncol=2, fontsize=9.5, handlelength=2.6, columnspacing=1.4)
    caption_mid(fig, "Figure 3: Simulasi ulang dengan k teks per prompt. Pada k kecil semua varian\n"
                 "tertahan di 0.46–0.52, jauh dari CV grup lengkap.")
    save_chart(fig, OUT / "fig3_group_size.png")


if __name__ == "__main__":
    pipeline()
    components()
    group_size()
