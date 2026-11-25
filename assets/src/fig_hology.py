"""Gambar untuk writeup Hology 2026 (HoloMine). Semua angka diambil dari Hology/README.md."""
from pathlib import Path

import numpy as np
from matplotlib.ticker import FuncFormatter

from paperfig import *

OUT = Path(__file__).resolve().parents[1] / "hology"


def ribuan(v):
    """Format angka bulat dengan pemisah ribuan gaya Indonesia (titik)."""
    return f"{v:,.0f}".replace(",", ".")


def caption_tengah(fig, s):
    """caption() dari paperfig, lalu digeser ke tengah isi grafik (label sumbu ikut dihitung)."""
    caption(fig, s)
    renderer = fig.canvas.get_renderer()
    boxes = [ax.get_tightbbox(renderer) for ax in fig.axes]
    x0, x1 = min(b.x0 for b in boxes), max(b.x1 for b in boxes)
    fig.texts[-1].set_x((x0 + x1) / 2 / fig.bbox.width)


# --------------------------------------------------------------------------
# Task 1: mammogram
# --------------------------------------------------------------------------
def task1_pipeline():
    # tinggi kanvas dihitung dari isi: kolom kanan 2.6 → 23.3, baris Normal + label di atasnya
    d = Diagram(52.6, 28.4, scale=0.2)
    fs = 11
    cy = 11.15  # sumbu baris utama (pusat kontainer tim)

    inp = d.box(0.5, cy - 2.3, 6.0, 4.6, "Mammogram\nJPEG", PINK, fs=fs + 0.5)
    d.text(inp.cx, inp.y - 0.7, "212 train\n54 test", fs=10, va="top")

    fmt = d.box(8.4, cy - 2.3, 6.4, 4.6, "Aturan format\n(mode warna)", YELLOW, fs=fs)
    d.arrow(inp.right, fmt.left)

    pre = d.box(17.4, cy - 3.2, 6.8, 6.4,
                "Preprocessing\nmask payudara,\ncrop, flip,\nwindowing,\n2048 px", YELLOW, fs=fs - 0.5)
    d.arrow(fmt.right, pre.left)
    d.text((fmt.x + fmt.w + pre.x) / 2, cy + 0.75, "RGB", fs=10, style="italic")

    pair = d.box(26.1, cy - 2.3, 6.0, 4.6, "Pairing view\ntanpa label", BLUE, fs=fs - 0.5)
    d.text(pair.cx, pair.y - 0.7, "133 kasus × 2 view", fs=10, va="top")
    d.arrow(pre.right, pair.left)

    # lima tim zero-shot
    teams = ["RSNA-2023 1st", "RSNA-2023 3rd", "Theo Viel", "MAM-CLIP", "Mergen1"]
    bh, gap, pad = 2.5, 0.75, 0.8
    ch = len(teams) * bh + (len(teams) - 1) * gap + 2 * pad
    cont = d.container(34.0, cy - ch / 2, 8.4, ch)
    for i, t in enumerate(teams):
        y = cont.y + cont.h - pad - bh - i * (bh + gap)
        d.box(cont.x + 0.6, y, cont.w - 1.2, bh, t, PEACH, fs=fs - 0.5)
    d.text(cont.cx, cont.y - 0.75, "5 tim zero-shot (bobot beku)", fs=10.5, style="italic")
    d.arrow(pair.right, cont.left)

    # kolom kanan, mengalir ke atas
    rx, rw, g = 44.2, 8.0, 1.2
    agg = d.box(rx, cont.y, rw, 4.4, "Agregasi per tim\nrerata fold & skala\n→ rerata 2 view\n→ z-score",
                YELLOW, fs=fs - 0.5)
    mean = d.box(rx, agg.y + agg.h + g, rw, 2.7, "Rerata 5 tim\n(bobot sama)", YELLOW, fs=fs - 0.5)
    platt = d.box(rx, mean.y + mean.h + g, rw, 2.7, "Platt seimbang\nkelas (2 param.)", LAVENDER, fs=fs - 0.5)
    dec = d.box(rx, platt.y + platt.h + g, rw, 3.4, "P(M) > 0,5\n→ Malignant,\nselain itu Benign", LAVENDER,
                fs=fs - 0.5)
    out = d.box(rx, dec.y + dec.h + g, rw, 2.7, "1 label per kasus\n(N / B / M)", GREEN, fs=fs - 0.5)
    d.arrow((cont.x + cont.w, agg.cy), agg.left)
    d.arrow(agg.top, mean.bottom)
    d.arrow(mean.top, platt.bottom)
    d.arrow(platt.top, dec.bottom)
    d.arrow(dec.top, out.bottom)

    # jalur grayscale: langsung Normal
    ny = out.y + out.h + 1.8
    normal = d.box(20.0, ny - 1.3, 10.0, 2.6, "Normal", GREEN, fs=fs)
    d.text(normal.cx, normal.y + normal.h + 0.45, "78 dari 78 JPEG grayscale di train berlabel Normal",
           fs=10.5, style="italic", va="bottom")
    d.arrow(fmt.top, (fmt.cx, normal.cy), normal.left)
    d.text(fmt.cx + 0.5, (fmt.y + fmt.h + normal.cy) / 2, "grayscale\n(mode L)", fs=10, style="italic",
           ha="left")
    d.arrow(normal.right, (out.cx, normal.cy), out.top)

    d.caption("Figure 1: Pipeline Task 1. Gambar grayscale langsung diberi label Normal, sedangkan kasus RGB\n"
              "dinilai lima tim zero-shot yang dibekukan lalu dikalibrasi dengan Platt dua parameter.",
              y=0.2, fs=14)
    d.save(OUT / "fig1_task1_pipeline.png")


def task1_leakage():
    enc = ["Mammo-CLIP B5", "Mammo-FM", "mammoscreen n0", "RSNA-2023 1st\n(fold 0 @2048)", "DINOv3 ViT-B"]
    leaky = np.array([0.844, 0.889, 0.845, 0.832, 0.714])
    safe = np.array([0.596, 0.671, 0.753, 0.819, 0.674])
    base = 0.5
    fig, ax = new_chart(6.4, 3.9)
    y = np.arange(len(enc))[::-1]
    h = 0.36
    ax.barh(y + h / 2, leaky - base, h, left=base, color=PINK, edgecolor="black", lw=1.0,
            label="Fold per gambar (bocor)", zorder=3)
    ax.barh(y - h / 2, safe - base, h, left=base, color=BLUE, edgecolor="black", lw=1.0,
            label="Fold aman-pasangan", zorder=3)
    for yi, a, b in zip(y, leaky, safe):
        ax.text(a + 0.004, yi + h / 2, f"{a:.3f}", va="center", ha="left", fontsize=9.5, zorder=4)
        ax.text(b + 0.004, yi - h / 2, f"{b:.3f}", va="center", ha="left", fontsize=9.5, zorder=4)
    ax.set_yticks(y)
    ax.set_yticklabels(enc)
    ax.tick_params(axis="y", length=0)
    ax.grid(axis="y", visible=False)
    ax.set_xlim(base, 0.95)
    ax.set_ylim(-0.6, len(enc) - 0.4)
    ax.set_xlabel("AUC Benign vs Malignant")
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.02), ncol=2)
    caption_tengah(fig, "Figure 2: AUC dengan fold per gambar (bocor) dibanding fold yang menjaga pasangan view.")
    save_chart(fig, OUT / "fig2_task1_leakage.png")


def task1_teams():
    labels = ["MAM-CLIP", "Theo Viel", "RSNA-2023 3rd", "Mergen1", "RSNA-2023 1st", "Rerata 5 tim (final)"]
    vals = [0.859, 0.816, 0.803, 0.772, 0.763, 0.833]
    fig, ax = new_chart(6.0, 3.4)
    hbar(ax, labels, vals, fmt="{:.3f}", highlight=5, xlabel="AUC kasus (Benign vs Malignant)",
         xlim=(0.5, 0.9), fontsize=10)
    ax.axhline(0.5, color="black", lw=0.8, ls="--", zorder=2)
    caption_tengah(fig, "Figure 3: AUC kasus tiap tim zero-shot dan ensemble final lima tim dengan bobot sama.")
    save_chart(fig, OUT / "fig3_task1_teams.png")


# --------------------------------------------------------------------------
# Task 2: harga properti
# --------------------------------------------------------------------------
def task2_pipeline():
    # tinggi kanvas dihitung dari isi: kolom kanan 3.1 → 24.5
    d = Diagram(49.0, 24.9, scale=0.2)
    fs = 11
    cy = 11.8

    inp = d.box(0.5, cy - 2.3, 5.2, 4.6, "Teks iklan\nproperti", PINK, fs=fs + 0.5)
    d.text(inp.cx, inp.y - 0.7, "14.640 train\n3.659 test", fs=10, va="top")

    tgt = d.box(7.6, cy - 3.0, 6.4, 6.0, "Target\nlog1p(harga)\n\n5-fold stratified\n(20 bin harga)",
                YELLOW, fs=fs - 1)
    d.arrow(inp.right, tgt.left)

    # fitur
    feats = ["Regex handcrafted\n(kamar, luas, tahun)", "Token lokasi +\ntarget enc. OOF",
             "TF-IDF kata + char\n→ SVD 256"]
    fbh, fgap, pad = 3.0, 0.8, 0.8
    fch = len(feats) * fbh + (len(feats) - 1) * fgap + 2 * pad
    fc = d.container(16.0, cy - fch / 2, 8.6, fch)
    for i, t in enumerate(feats):
        y = fc.y + fc.h - pad - fbh - i * (fbh + fgap)
        d.box(fc.x + 0.6, y, fc.w - 1.2, fbh, t, BLUE, fs=fs - 1)
    d.text(fc.cx, fc.y - 0.75, "Fitur", fs=10.5, style="italic")
    d.arrow(tgt.right, fc.left)

    # 37 model dasar
    models = ["Ridge / LinearSVR", "kNN cosine\n(k = 5–40)", "LightGBM\nL1 / L2 / Huber", "Ensemble ordinal",
              "DeBERTa-v3\nbase & large"]
    mbh, mgap = 2.6, 0.7
    mch = len(models) * mbh + (len(models) - 1) * mgap + 2 * pad
    mc = d.container(27.0, cy - mch / 2, 8.8, mch)
    for i, t in enumerate(models):
        y = mc.y + mc.h - pad - mbh - i * (mbh + mgap)
        d.box(mc.x + 0.6, y, mc.w - 1.2, mbh, t, PEACH, fs=fs - 1)
    d.text(mc.cx, mc.y - 0.75, "37 model dasar (OOF)", fs=10.5, style="italic")
    d.arrow(fc.right, mc.left)

    # kolom kanan, mengalir ke atas
    rx, rw, g = 38.4, 10.0, 1.2
    stack = d.box(rx, mc.y, rw, 4.6, "Stack level 2:\nLightGBM, XGBoost,\nCatBoost\npada 14.640 × 401",
                  PEACH, fs=fs - 1)
    blend = d.box(rx, stack.y + stack.h + g, rw, 4.0,
                  "Buang korelasi ≥ 0,995\n(42 → 34 kandidat), lalu\nblend QuantileRegressor",
                  LAVENDER, fs=fs - 1)
    dup = d.box(rx, blend.y + blend.h + g, rw, 2.7, "Override duplikat\n(cos ≥ 0,9)", YELLOW, fs=fs - 1)
    clip = d.box(rx, dup.y + dup.h + g, rw, 2.7, "Clip ke\n[\\$1.000, \\$90 juta]", YELLOW, fs=fs - 1)
    sub = d.box(rx, clip.y + clip.h + g, rw, 2.6, "submission.csv", GREEN, fs=fs)
    d.arrow((mc.x + mc.w, stack.y + 1.3), (stack.x, stack.y + 1.3))
    d.arrow(stack.top, blend.bottom)
    d.arrow(blend.top, dup.bottom)
    d.arrow(dup.top, clip.bottom)
    d.arrow(clip.top, sub.bottom)

    # fitur asli juga masuk ke stacker
    vx = (mc.x + mc.w + rx) / 2
    ty = mc.y + mc.h + 0.9
    d.arrow((fc.cx, fc.y + fc.h), (fc.cx, ty), (vx, ty), (vx, stack.y + stack.h - 1.0),
            (stack.x, stack.y + stack.h - 1.0), ls=(0, (4, 2)))
    d.text((fc.cx + vx) / 2, ty + 0.35, "fitur asli", fs=10, style="italic", va="bottom")

    d.caption("Figure 4: Pipeline Task 2. Prediksi OOF dari 37 model dasar ditumpuk di level 2,\n"
              "di-blend dengan QuantileRegressor, lalu dikoreksi dengan override duplikat.",
              y=0.2, fs=14)
    d.save(OUT / "fig4_task2_pipeline.png")


def task2_ladder():
    labels = ["Median konstan", "Ridge pada dolar mentah", "Ridge pada target log",
              "LightGBM Huber\n(regex + lokasi + SVD)", "DeBERTa-v3-base", "DeBERTa-v3-large",
              "Stack LightGBM L2\n(setelah DeBERTa-large)", "Blend QuantileRegressor",
              "Blend + override\nduplikat (final)"]
    vals = [550252, 516771, 346843, 330534, 291597, 283793, 271094, 267845, 267201]
    fig, ax = new_chart(6.6, 4.4)
    hbar(ax, labels, vals, fmt="{:,.0f}", highlight=len(vals) - 1,
         xlabel="OOF MAE dalam USD (lebih kecil lebih baik)", xlim=(0, 650000), fontsize=10)
    for t in ax.texts:
        t.set_text(t.get_text().replace(",", "."))
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: ribuan(v)))
    caption_tengah(fig, "Figure 5: OOF MAE tiap tahap, dari tebakan median sampai submission final.")
    save_chart(fig, OUT / "fig5_task2_ladder.png")


if __name__ == "__main__":
    task1_pipeline()
    task1_leakage()
    task1_teams()
    task2_pipeline()
    task2_ladder()
