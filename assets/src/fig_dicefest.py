"""Gambar untuk writeup DiceFest 2026. Angka diambil dari notebook final dan kaggle_writeup.md."""
from pathlib import Path

import numpy as np

from paperfig import *

OUT = Path(__file__).resolve().parents[1] / "dicefest"


def pipeline():
    d = Diagram(47, 27.5, scale=0.21)
    fs = 11

    inp = d.box(0.5, 10.6, 5.6, 4.4, "train.csv\ntest.csv", PINK, fs=fs + 1)
    d.text(3.3, 9.4, "429k + 184k baris\n201 fitur anonim", fs=9.5, va="top")
    pre = d.box(8.2, 10.6, 6.2, 4.4, "Preprocessing\nencode kategori,\nkode zone &\nsubzone", YELLOW, fs=fs - 0.5)
    d.arrow(inp.right, pre.left)

    # jalur target struktural
    d.text(23.5, 26.6, "Target struktural (eksak)", fs=fs, style="italic")
    clean = d.box(18.0, 23.0, 11.0, 2.6, "clean = 100·(1 − b13 / b01)", GREEN, fs=fs)
    eh = d.box(18.0, 19.4, 11.0, 2.8, "EH: lookup subzone,\nLightGBM untuk subzone baru", GREEN, fs=fs - 0.5)
    d.arrow(pre.right, (15.8, pre.cy), (15.8, clean.cy), clean.left)
    d.arrow(pre.right, (15.8, pre.cy), (15.8, eh.cy), eh.left)

    # jalur target kontinu
    fc = d.container(17.0, 2.6, 9.0, 13.4)
    feats = ["Set A · 496 kolom", "Set B · 426 kolom", "Fitur N\n(jumlah rumah tangga)", "Fitur spasial\n(K-NN climate cell)"]
    ys = [12.6, 9.6, 6.6, 3.4]
    fboxes = [d.box(17.7, y, 7.6, 2.5 if i < 3 else 2.6, t, BLUE, fs=fs - 1) for i, (t, y) in enumerate(zip(feats, ys))]
    d.text(21.5, 1.7, "Feature engineering", fs=fs - 0.5, style="italic")
    d.arrow(pre.right, (15.8, pre.cy), (15.8, 9.3), (fc.x, 9.3))

    mc = d.container(28.0, 2.6, 9.4, 13.4)
    models = [("LightGBM WLS\n(EUI, gas)", PEACH), ("LightGBM Poisson\nper-meter, linear tree", PEACH),
              ("XGBoost Poisson\n(GPU, base margin)", PEACH), ("MLP Poisson &\nMLP multi-task (DAE)", LAVENDER)]
    for (t, c), y in zip(models, ys):
        d.box(28.7, y, 8.0, 2.5 if y > 4 else 2.6, t, c, fs=fs - 1)
    d.text(32.7, 1.7, "16 model × 5 fold", fs=fs - 0.5, style="italic")
    d.arrow((fc.x + fc.w, 9.3), (mc.x, 9.3))

    nnls = d.box(39.4, 3.4, 7.2, 2.6, "NNLS blend\nper target", YELLOW, fs=fs - 0.5)
    ridge = d.box(39.4, 7.3, 7.2, 2.6, "Koreksi spasial\n(ridge)", YELLOW, fs=fs - 0.5)
    post = d.box(39.4, 11.2, 7.2, 2.8, "Clip, twin override,\ngas = EUI × area", YELLOW, fs=fs - 1)
    out = d.box(39.4, 15.6, 7.2, 2.6, "submission.csv\n(5 target)", GREEN, fs=fs - 0.5)
    d.arrow((mc.x + mc.w, nnls.cy), nnls.left)
    d.arrow(nnls.top, ridge.bottom)
    d.arrow(ridge.top, post.bottom)
    d.arrow(post.top, out.bottom)
    # fitur spasial juga masuk ke koreksi ridge
    d.arrow((21.5, fc.y + fc.h), (21.5, 16.8), (38.2, 16.8), (38.2, ridge.cy), ridge.left, ls=(0, (4, 2)))
    d.arrow(clean.right, (45.2, clean.cy), (45.2, out.y + out.h))
    d.arrow(eh.right, (40.8, eh.cy), (40.8, out.y + out.h))

    d.caption("Figure 1: Pipeline solusi DiceFest 2026. Dua target diturunkan dari identitas data,\n"
              "tiga target kontinu dari 16 model yang di-blend per target.", y=0.4, fs=14)
    d.save(OUT / "fig1_pipeline.png")


def ablation():
    steps = ["v1", "+ MLP\nmulti-task", "+ gas\nper-meter", "+ koreksi\nspasial (v2)", "+ XGB, seed bag,\nfitur N (v3)"]
    cv = [0.25329, 0.25223, 0.25175, 0.25128, 0.25107]
    lb_idx, lb = [0, 3, 4], [0.25035, 0.24809, 0.24798]
    fig, ax = new_chart(6.6, 3.5)
    x = np.arange(len(steps))
    ax.plot(x, cv, "-o", color=DARK["blue"], lw=1.8, ms=6, mfc=BLUE, mec="black", label="CV 5-fold", zorder=3)
    ax.plot(lb_idx, lb, "--s", color=DARK["peach"], lw=1.5, ms=6, mfc=PEACH, mec="black", label="Public LB", zorder=3)
    ax.plot([4], [0.24772], "*", ms=13, mfc=GREEN, mec="black", label="Private LB (peringkat 6)", zorder=4)
    for xi, v in zip(x, cv):
        ax.annotate(f"{v:.5f}", (xi, v), textcoords="offset points", xytext=(0, 7), ha="center", fontsize=9)
    for xi, v, off in zip(lb_idx, lb, [(0, -14), (0, -14), (-34, 3)]):
        ax.annotate(f"{v:.5f}", (xi, v), textcoords="offset points", xytext=off, ha="center", fontsize=9)
    ax.annotate("0.24772", (4, 0.24772), textcoords="offset points", xytext=(0, -16), ha="center", fontsize=9)
    ax.set_xticks(x)
    ax.set_xticklabels(steps, fontsize=9.5)
    ax.set_ylabel("NRMSE (lebih kecil lebih baik)")
    ax.set_ylim(0.2470, 0.2542)
    ax.set_xlim(-0.4, 4.4)
    ax.legend(loc="upper right")
    caption(fig, "Figure 2: Kontribusi tiap tambahan terhadap CV dan leaderboard.")
    save_chart(fig, OUT / "fig2_ablation.png")


def seen_unseen():
    targets = ["gas", "elec", "EUI", "EH"]
    seen = [0.23684, 0.30558, 0.64308, 0.0]
    unseen = [0.28245, 0.37131, 0.67355, 0.13802]
    fig, ax = new_chart(6.0, 3.2)
    x = np.arange(len(targets))
    w = 0.36
    ax.bar(x - w / 2, seen, w, color=BLUE, edgecolor="black", lw=1, label="Subzone dikenal", zorder=3)
    ax.bar(x + w / 2, unseen, w, color=PEACH, edgecolor="black", lw=1, label="Subzone baru", zorder=3)
    for xi, a, b in zip(x, seen, unseen):
        ax.text(xi - w / 2, a + 0.01, f"{a:.3f}", ha="center", fontsize=8.5)
        ax.text(xi + w / 2, b + 0.01, f"{b:.3f}", ha="center", fontsize=8.5)
    ax.set_xticks(x)
    ax.set_xticklabels(targets)
    ax.set_ylabel("NRMSE OOF")
    ax.set_ylim(0, 0.78)
    ax.grid(axis="x", visible=False)
    ax.legend(loc="upper left")
    caption(fig, "Figure 3: Error OOF pada baris yang subzone-nya muncul di train vs tidak.")
    save_chart(fig, OUT / "fig3_seen_unseen.png")


if __name__ == "__main__":
    pipeline()
    ablation()
    seen_unseen()
