"""Gambar untuk writeup DATASCAPE 2026. Angka diambil dari Datascape-2026/README.md."""
from pathlib import Path

import numpy as np

from paperfig import *

OUT = Path(__file__).resolve().parents[1] / "datascape"


def nx_label(d, x, y, s, fs=13):
    """Label pengulangan "N×" di samping kontainer abu-abu."""
    d.text(x, y, s, fs=fs)


def caption_c(fig, s, fs=12.5):
    """Seperti paperfig.caption, tetapi di tengah seluruh isi gambar (termasuk label sumbu)."""
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    bb = fig.get_tightbbox(r)
    cx = (bb.x0 + bb.x1) / 2 / fig.get_figwidth()
    bottom = min(ax.get_tightbbox(r).y0 for ax in fig.axes) / fig.bbox.height - 0.04
    fig.text(cx, bottom, s, ha="center", va="top", fontsize=fs, family=SERIF)


def pipeline():
    d = Diagram(52.2, 32.4, scale=0.2)
    fs = 11

    # garis tengah: jalur gambar mentah ke head fine-tune
    y_img = 16.8

    inp = d.box(0.5, y_img - 2.2, 5.4, 4.4, "Gambar\ntrain + test", PINK, fs=fs + 0.5)
    d.text(inp.cx, inp.y - 0.7, "12.224 + 3.057\ngambar", fs=9.5, va="top")
    dec = d.box(7.4, y_img - 2.4, 6.0, 4.8, "Decode,\nresize 448,\ncache", YELLOW, fs=fs)
    d.arrow(inp.right, dec.left)

    # ---- head (kolom E): posisi celah antar kotak menentukan baris lain
    hx, hw, hh, gap, b0 = 34.7, 7.2, 3.2, 1.6, 8.0
    ys = [b0 + i * (hh + gap) for i in range(4)]          # bawah -> atas
    g_fold = b0 + hh + gap / 2                               # celah PE-L / ConvNeXt
    g_emb = b0 + 3 * hh + 2.5 * gap                          # celah kNN / LR+MLP
    hc = d.container(hx - 0.7, b0 - 0.8, hw + 1.4, 4 * hh + 3 * gap + 1.6)
    heads = [("PE-Core-L\nfine-tune", ys[0]), ("ConvNeXt-B\nfine-tune", ys[1]),
             ("kNN seimbang\nkelas, k = 20", ys[2]), ("LR + MLP\n(+ anchor test)", ys[3])]
    for t, y in heads:
        d.box(hx, y, hw, hh, t, LAVENDER, fs=fs - 0.5)

    cl = d.box(hx - 0.4, hc.y + hc.h + 1.8, hw + 0.8, 3.2,
               "Confident Learning:\n399 label → bobot 0,3", YELLOW, fs=fs - 1)
    d.arrow((hx + 1.9, hc.y + hc.h), (hx + 1.9, cl.y))
    d.arrow((hx + hw - 1.9, cl.y), (hx + hw - 1.9, hc.y + hc.h))
    ymid = (hc.y + hc.h + cl.y) / 2
    d.text(hx + 1.5, ymid, "OOF", fs=9.5, ha="right", style="italic")
    d.text(hx + hw - 1.5, ymid, "latih ulang", fs=9.5, ha="left", style="italic")

    # ---- jalur embedding (atas)
    ec = d.container(16.6, g_emb - 2.6, 8.0, 5.2)
    enc = d.box(17.3, g_emb - 1.8, 6.6, 3.6, "Encoder ViT\nbeku + TTA flip", PEACH, fs=fs)
    nx_label(d, 15.55, ec.cy, "8×")
    d.text(ec.cx, ec.y + ec.h + 0.5, "PE-Core L/G · SigLIP2 · EVA02\nCLIP L/H · DINOv2 · DINOv3",
           fs=9.5, va="bottom", style="italic")
    emb = d.box(26.4, g_emb - 1.7, 5.6, 3.4, "Embedding\n8.576 dim", BLUE, fs=fs)
    d.arrow(dec.right, (enc.cx, y_img), (enc.cx, enc.y))
    d.arrow(enc.right, emb.left)
    d.arrow(emb.right, (hc.x, g_emb))

    # ---- gambar mentah ke head fine-tune
    d.arrow(dec.right, (hc.x, y_img))

    # ---- forensik duplikat (bawah)
    fc = d.container(16.6, 1.0, 8.0, 12.2)
    cand = d.box(17.3, 9.0, 6.6, 3.4, "pHash / dHash\n+ DINOv2 kNN", YELLOW, fs=fs - 0.5)
    orb = d.box(17.3, 1.7, 6.6, 3.8, "Verifikasi ORB\n→ pasangan\nSTRICT", YELLOW, fs=fs - 0.5)
    d.text(fc.cx, fc.y + fc.h + 0.4, "Forensik duplikat", fs=fs - 0.5, va="bottom", style="italic")
    d.arrow(dec.right, (14.6, y_img), (14.6, cand.cy), cand.left)
    d.arrow(cand.bottom, orb.top)

    uf = d.box(26.4, orb.cy - 1.8, 5.6, 3.6, "Union-find\n6.432 grup", YELLOW, fs=fs)
    sg = d.box(26.4, g_fold - 1.8, 5.6, 3.6, "Stratified\nGroupKFold\n5 fold", YELLOW, fs=fs - 0.5)
    d.arrow(orb.right, uf.left)
    d.arrow(uf.top, sg.bottom)
    d.arrow(sg.right, (hc.x, g_fold))

    tr = d.box(hx, uf.y, hw, uf.h, "Transfer label\nsalinan (grup)", BLUE, fs=fs - 0.5)
    d.arrow(uf.right, tr.left)

    # ---- ensemble dan post-processing (kolom F, mengalir ke atas)
    fx, fw = 44.6, 7.0
    ens = d.box(fx, g_fold - 2.0, fw, 4.0, "Ensemble\nlog-prob", YELLOW, fs=fs)
    d.arrow((hc.x + hc.w, g_fold), ens.left)
    d.arrow(tr.right, (ens.cx, tr.cy), ens.bottom)
    sc = d.box(fx, ens.y + ens.h + 2.0, fw, 4.0, "Koreksi jalan\npintas (ghtk,\nisocyt)", YELLOW, fs=fs - 0.5)
    la = d.box(fx, sc.y + sc.h + 1.8, fw, 4.6, "Perluasan Lainnya\nLP + P(Lainnya),\ntop 55", YELLOW, fs=fs - 1)
    sub = d.box(fx, la.y + la.h + 1.8, fw, 3.0, "submission.csv", GREEN, fs=fs)
    d.arrow(ens.top, sc.bottom)
    d.arrow(sc.top, la.bottom)
    d.arrow(la.top, sub.bottom)

    d.caption("Figure 1: Pipeline solusi DATASCAPE 2026. Forensik duplikat memberi grup CV dan term transfer;\n"
              "hasil ensemble dikoreksi oleh dua aturan post-processing untuk kelas Rusak dan Lainnya.",
              y=0.1, fs=14)
    d.save(OUT / "fig1_pipeline.png")


def lb_progress():
    steps = ["v1", "+ jalan\npintas", "+ Lainnya", "+ LP,\ntransfer", "+ ConvNeXt",
             "+ CL", "+ CNX-CL", "+ grup", "final"]
    pub = [0.83382, 0.88794, 0.96044, 0.97577, 0.97673, 0.97770, 0.97770, 0.97867, 0.98075]
    prv = [0.78781, 0.84264, 0.95579, 0.96241, 0.96415, 0.96502, 0.96585, 0.96627, 0.96760]
    fig, ax = new_chart(7.0, 3.7)
    x = np.arange(len(steps))
    ax.plot(x[:-1], pub[:-1], "-o", color=DARK["blue"], lw=1.8, ms=6, mfc=BLUE, mec="black",
            label="Public LB", zorder=3)
    ax.plot(x[:-1], prv[:-1], "--s", color=DARK["peach"], lw=1.5, ms=6, mfc=PEACH, mec="black",
            label="Private LB", zorder=3)
    ax.plot(x[-2:], pub[-2:], "-", color=DARK["blue"], lw=1.8, zorder=2)
    ax.plot(x[-2:], prv[-2:], "--", color=DARK["peach"], lw=1.5, zorder=2)
    ax.plot([x[-1]] * 2, [pub[-1], prv[-1]], "*", ms=14, mfc=GREEN, mec="black", ls="none",
            label="Final (private peringkat 1)", zorder=4)
    pub_off = {1: (-16, 4), 2: (-20, 3)}
    prv_off = {1: (16, -6), 2: (22, -17)}
    for xi, v in zip(x, pub):
        ax.annotate(f"{v:.5f}", (xi, v), textcoords="offset points", xytext=pub_off.get(xi, (0, 7)),
                    ha="center", va="bottom", fontsize=8.5)
    for xi, v in zip(x, prv):
        lead = dict(arrowstyle="-", lw=0.6, color="black", shrinkA=1, shrinkB=4) if xi == 2 else None
        ax.annotate(f"{v:.5f}", (xi, v), textcoords="offset points", xytext=prv_off.get(xi, (0, -8)),
                    ha="center", va="top", fontsize=8.5, arrowprops=lead)
    ax.set_xticks(x)
    ax.set_xticklabels(steps, fontsize=9.5)
    ax.set_ylabel("Macro-F1")
    ax.set_ylim(0.76, 1.0)
    ax.set_xlim(-0.5, len(steps) - 0.5)
    ax.legend(loc="lower right")
    caption_c(fig, "Figure 2: Skor public dan private leaderboard per langkah. LP = label propagation,\n"
                   "CL = Confident Learning, CNX-CL = ConvNeXt-CL, grup = propagasi koreksi ke anggota grup.")
    save_chart(fig, OUT / "fig2_lb_progress.png")


def encoders():
    names = ["PE-Core-L", "PE-Core-G", "SigLIP2 So400m", "EVA02-L CLIP", "CLIP ViT-L/14 (OpenAI)",
             "CLIP ViT-H/14 (LAION-2B)", "DINOv3 ViT-L", "DINOv2 ViT-L reg4"]
    f1 = [0.8701, 0.8676, 0.8590, 0.8434, 0.8432, 0.8414, 0.8270, 0.7840]
    order = np.argsort(f1)[::-1]
    names = [names[i] for i in order]
    f1 = [f1[i] for i in order]
    fig, ax = new_chart(6.4, 3.4)
    hbar(ax, names, f1, highlight=0, xlabel="OOF macro-F1 (regresi logistik saja)",
         xlim=(0.76, 0.886), fontsize=9.5)
    caption_c(fig, "Figure 3: OOF macro-F1 regresi logistik di atas embedding tiap encoder beku.")
    save_chart(fig, OUT / "fig3_encoders.png")


if __name__ == "__main__":
    pipeline()
    lb_progress()
    encoders()
