"""Gambar untuk writeup IMITCOMP 2026. Angka diambil dari IMITCOMP-2026/README.md."""
from pathlib import Path

import numpy as np

from paperfig import *

OUT = Path(__file__).resolve().parents[1] / "imitcomp"


def pipeline():
    d = Diagram(46.5, 25.4, scale=0.21)
    fs = 11

    # kolom kiri: persiapan data, dari atas ke bawah
    steps = [
        ("credit_risk_dataset.csv\n32.581 pinjaman", PINK, fs - 1),
        ("Imputasi median,\nwinsorization IQR", YELLOW, fs),
        ("7 fitur turunan\n(DTI, skor grade, …)", BLUE, fs),
        ("Encoding\n→ 27 fitur", BLUE, fs),
        ("Split stratified\n80/20", YELLOW, fs),
        ("SMOTE pada\ndata train", YELLOW, fs),
    ]
    h0, gap, top = 2.8, 1.0, 25.2
    left = []
    for i, (t, c, f) in enumerate(steps):
        left.append(d.box(0.5, top - h0 - i * (h0 + gap), 8.6, h0, t, c, fs=f))
    for a, b in zip(left, left[1:]):
        d.arrow(a.bottom, b.top)
    smote = left[-1]

    # tengah: empat model + tuning XGBoost
    cont = d.container(12.0, 2.6, 10.8, 18.3)
    bx, bw = 12.7, 9.4
    tune = d.box(bx, 3.3, bw, 3.2, "Tuning XGBoost\n(RandomizedSearchCV,\n20 iterasi)", YELLOW, fs=fs - 1)
    xgb = d.box(bx, 7.6, bw, 2.4, "XGBoost", PEACH, fs=fs)
    lgb = d.box(bx, 11.0, bw, 2.4, "LightGBM", PEACH, fs=fs)
    rf = d.box(bx, 14.4, bw, 2.4, "Random forest", PEACH, fs=fs)
    lr = d.box(bx, 17.8, bw, 2.4, "Regresi logistik", PEACH, fs=fs)
    d.arrow(xgb.bottom, tune.top)
    d.text(cont.cx, 1.75, "4 model, CV 5 fold di train", fs=fs - 0.5, style="italic")
    d.arrow(smote.right, (10.55, smote.cy), (10.55, cont.cy), cont.left)

    # pemilihan model dan penjelasan
    pick = d.box(25.2, xgb.cy - 1.5, 8.0, 3.0, "Pilih XGBoost\nAUC-ROC 0.9496", LAVENDER, fs=fs)
    d.arrow(xgb.right, pick.left)
    shap = d.box(25.2, 12.6, 8.0, 3.6, "Penjelasan SHAP\n(TreeExplainer,\n500 baris test)", BLUE, fs=fs - 0.5)
    d.arrow(pick.top, shap.bottom)
    d.text(shap.cx, 17.0, "|SHAP| terbesar:\nskor grade 0.840\npendapatan 0.773\ntujuan usaha 0.717",
           fs=9.5, va="bottom", linespacing=1.25)

    # kolom kanan: skor -> pita -> keputusan (dari bawah ke atas)
    rx, rw = 35.8, 10.2
    score = d.box(rx, pick.y, rw, 3.0, "Skor risiko\n= 100 · P(gagal bayar)", LAVENDER, fs=fs - 0.5)
    d.arrow(pick.right, score.left)
    band_c = d.container(rx, 11.4, rw, 9.4)
    d.text(band_c.cx, 19.85, "Pita risiko", fs=fs - 0.5, style="italic")
    bands = [("Low (≤ 30)", GREEN), ("Medium (30–60)", YELLOW), ("High (60–80)", PEACH),
             ("Very High (> 80)", PINK)]
    for i, (t, c) in enumerate(bands):
        d.box(rx + 0.7, 12.0 + i * 1.85, rw - 1.4, 1.55, t, c, fs=fs - 1, radius=0.3)
    d.arrow(score.top, band_c.bottom)
    dec = d.box(rx, 22.2, rw, 3.0, "Keputusan: setujui,\ntinjau manual, tolak", GREEN, fs=fs - 0.5)
    d.arrow(band_c.top, dec.bottom)

    d.caption("Figure 1: Pipeline notebook IMITCOMP 2026, dari data mentah sampai model terpilih,\n"
              "skor risiko, dan rekomendasi keputusan.", y=0.4, fs=14)
    d.save(OUT / "fig1_pipeline.png")


def models():
    names = ["Regresi\nlogistik", "Random\nforest", "XGBoost\n(dipilih)", "LightGBM", "XGBoost\nsetelah tuning"]
    f1 = [0.6458, 0.7976, 0.8157, 0.8299, 0.8193]
    roc = [0.8710, 0.9193, 0.9496, 0.9494, 0.9467]
    pr = [0.7234, 0.8687, 0.9065, 0.9049, 0.9024]
    fig, ax = new_chart(6.8, 3.7)
    x = np.arange(len(names))
    w = 0.26
    ax.axvspan(2 - 0.47, 2 + 0.47, color=PEACH, alpha=0.7, zorder=0, lw=0)
    series = [("F1", f1, BLUE), ("AUC-ROC", roc, LAVENDER), ("AUC-PR", pr, GREEN)]
    for k, (lab, vals, c) in enumerate(series):
        xs = x + (k - 1) * w
        ax.bar(xs, vals, w, color=c, edgecolor="black", lw=1, label=lab, zorder=3)
        for xi, v in zip(xs, vals):
            ax.text(xi, v - 0.02, f"{v:.4f}", ha="center", va="top", rotation=90, fontsize=8.5, zorder=4)
    ax.set_xticks(x)
    ax.set_xticklabels(names)
    for i, t in enumerate(ax.get_xticklabels()):
        if i == 2:
            t.set_fontweight("bold")
    ax.set_ylabel("Skor di set test")
    ax.set_ylim(0, 1.12)
    ax.set_yticks(np.arange(0, 1.01, 0.2))
    ax.set_xlim(-0.55, len(names) - 0.45)
    ax.grid(axis="x", visible=False)
    ax.legend(loc="upper center", ncol=3, bbox_to_anchor=(0.5, 1.0))
    caption(fig, "Figure 2: F1, AUC-ROC, dan AUC-PR kelima konfigurasi di set test 20%.")
    save_chart(fig, OUT / "fig2_models.png")


def risk_bands():
    bands = ["Low\n(≤ 30)", "Medium\n(30–60)", "High\n(60–80)", "Very High\n(> 80)"]
    n = ["4.301", "1.009", "170", "1.037"]
    rate = [3.5, 17.8, 41.2, 98.6]
    fills = [GREEN, YELLOW, PEACH, PINK]
    fig, ax = new_chart(6.0, 3.4)
    x = np.arange(len(bands))
    ax.bar(x, rate, 0.6, color=fills, edgecolor="black", lw=1, zorder=3)
    for xi, v in zip(x, rate):
        ax.text(xi, v + 2, f"{v:.1f}%".replace(".", ","), ha="center", va="bottom", fontsize=10.5, zorder=4,
                bbox=dict(facecolor="white", edgecolor="none", pad=1.0))
    ax.axhline(21.8, color=DARK["grey"], ls="--", lw=1.2, zorder=2)
    ax.text(-0.42, 23.3, "rata-rata data 21,8%", fontsize=9.5, color=DARK["grey"], va="bottom")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{b}\nn = {k}" for b, k in zip(bands, n)])
    ax.set_ylabel("Gagal bayar aktual (%)")
    ax.set_ylim(0, 112)
    ax.set_yticks(np.arange(0, 101, 20))
    ax.set_xlim(-0.5, len(bands) - 0.5)
    ax.grid(axis="x", visible=False)
    caption(fig, "Figure 3: Tingkat gagal bayar aktual per pita skor risiko XGBoost di set test.")
    save_chart(fig, OUT / "fig3_risk_bands.png")


if __name__ == "__main__":
    pipeline()
    models()
    risk_bands()
