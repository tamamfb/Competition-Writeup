"""Gambar untuk writeup Datathon UI 2026. Angka diambil dari DATATHON-UI/README.md."""
from pathlib import Path

import matplotlib as mpl
import numpy as np
from matplotlib.patches import Circle

from paperfig import *

OUT = Path(__file__).resolve().parents[1] / "datathon-ui"


def sans_math():
    """Mathtext memakai Arial supaya subskrip di dalam kotak serasi dengan teks biasa."""
    mpl.rcParams.update({"mathtext.fontset": "custom", "mathtext.rm": "Arial",
                         "mathtext.it": "Arial:italic", "mathtext.bf": "Arial:bold",
                         "mathtext.sf": "Arial"})


def op_circle(d, x, y, sym, r=0.62, fs=13):
    """Lingkaran operasi (mis. gabungan himpunan) bergaya simbol (+) Transformer."""
    d.ax.add_patch(Circle((x, y), r, facecolor="white", edgecolor="black", lw=LW, zorder=4))
    d.ax.text(x, y, sym, ha="center", va="center", fontsize=fs, family=SANS, zorder=5)
    return Box(x - r, y - r, 2 * r, 2 * r)


# --------------------------------------------------------------------------
# Figure 1: playground
# --------------------------------------------------------------------------
def playground():
    fig, (ax, ax2) = new_chart(7.6, 3.0, ncols=2, gridspec_kw=dict(width_ratios=[2.3, 1]))
    fig.subplots_adjust(left=0.18, right=0.98, wspace=0.32)
    names = ["TabPFN (dipakai)", "XGBoost + TabPFN\n(50 : 50)", "XGBoost\n(Optuna, 25 trial)",
             "LightGBM\n(Optuna, 25 trial)"]
    f1 = [0.5578, 0.5398, 0.5054, 0.4965]
    sd = [0.0256, 0.0140, 0.0127, 0.0167]
    y = np.arange(len(names))[::-1]
    fills = [PEACH, BLUE, BLUE, BLUE]
    ax.barh(y, f1, xerr=sd, color=fills, edgecolor="black", lw=1, height=0.62, zorder=3,
            error_kw=dict(ecolor="black", elinewidth=1.1, capsize=3.5, capthick=1.1, zorder=4))
    for yi, v, s in zip(y, f1, sd):
        ax.text(v + s + 0.012, yi, f"{v:.4f} ± {s:.4f}", va="center", ha="left", fontsize=9.5)
    ax.set_yticks(y)
    ax.set_yticklabels(names)
    ax.tick_params(axis="y", length=0)
    ax.grid(axis="y", visible=False)
    ax.set_xlim(0, 0.83)
    ax.set_xlabel("Macro-F1 CV (5 fold, rata-rata ± std)")
    ax.set_title("(a) Perbandingan model", fontsize=11.5)

    x2 = np.arange(2)
    vals = [0.4570, 0.5054]
    ax2.bar(x2, vals, 0.58, color=[BLUE, PEACH], edgecolor="black", lw=1, zorder=3)
    for xi, v in zip(x2, vals):
        ax2.text(xi, v + 0.012, f"{v:.4f}", ha="center", va="bottom", fontsize=9.5)
    ax2.set_xticks(x2)
    ax2.set_xticklabels(["41\n(mentah)", "54\n(+13 baru)"])
    ax2.set_xlabel("Jumlah fitur")
    ax2.set_ylabel("Macro-F1 CV XGBoost")
    ax2.set_ylim(0, 0.62)
    ax2.set_xlim(-0.6, 1.6)
    ax2.grid(axis="x", visible=False)
    ax2.set_title("(b) Feature engineering", fontsize=11.5)
    caption(fig, "Figure 1: Macro-F1 CV model playground (a) dan efek 13 fitur ringkasan deret mingguan (b).")
    save_chart(fig, OUT / "fig1_playground.png")


# --------------------------------------------------------------------------
# Figure 2: arsitektur TrafficNet
# --------------------------------------------------------------------------
def task1_model():
    sans_math()
    d = Diagram(44.2, 48.6, scale=0.21)
    fs = 10.5

    # --- cabang input (bawah) ---
    feat_y, feat_h = 8.4, 2.6
    proc_y, proc_h = 3.6, 3.4
    cols = {"spd": (1.0, 9.6), "reg": (11.6, 6.4), "sta": (19.0, 6.6), "emb": (26.6, 6.0), "evt": (33.6, 9.6)}

    def col(k):
        x, w = cols[k]
        return x, w, x + w / 2

    x, w, cx = col("spd")
    d.text(cx, 1.4, "Riwayat kecepatan\n15 langkah × 1.260 ruas", fs=fs - 0.5)
    p_spd = d.box(x, proc_y, w, proc_h, "Nilai 0 = hilang,\nimputasi + mask,\nnormalisasi per ruas",
                  YELLOW, fs=fs - 0.5)
    f_spd = d.box(x, feat_y, w, feat_h, "Fitur temporal\n(51)", BLUE, fs=fs)
    d.arrow((cx, 2.4), p_spd.bottom)
    d.arrow(p_spd.top, f_spd.bottom)

    x, w, cx = col("reg")
    f_reg = d.box(x, feat_y, w, feat_h, "Proksi kondisi\njaringan (4)", BLUE, fs=fs)
    d.arrow(p_spd.right, (cx, p_spd.cy), f_reg.bottom)

    x, w, cx = col("sta")
    d.text(cx, 1.4, "Atribut ruas", fs=fs - 0.5)
    f_sta = d.box(x, feat_y, w, feat_h, "Fitur statis\nruas (37)", BLUE, fs=fs)
    d.arrow((cx, 2.0), f_sta.bottom)

    x, w, cx = col("emb")
    d.text(cx, 1.4, "ID ruas", fs=fs - 0.5)
    f_emb = d.box(x, feat_y, w, feat_h, "Embedding\nruas (64)", PINK, fs=fs)
    d.arrow((cx, 2.0), f_emb.bottom)

    x, w, cx = col("evt")
    d.text(cx, 1.4, "Teks kejadian\nper langkah", fs=fs - 0.5)
    p_evt = d.box(x, proc_y, w, proc_h, "Urai 6 jenis kejadian,\npetakan ke ruas,\nsebar di graf",
                  YELLOW, fs=fs - 0.5)
    f_evt = d.box(x, feat_y, w, feat_h, "Kanal kejadian\n(24)", BLUE, fs=fs)
    d.arrow((cx, 2.4), p_evt.bottom)
    d.arrow(p_evt.top, f_evt.bottom)

    cat = d.box(1.0, 12.4, 42.2, 1.8, "Concat  (51 + 4 + 37 + 64 + 24 = 180)", YELLOW, fs=fs)
    for f in (f_spd, f_reg, f_sta, f_emb, f_evt):
        d.arrow(f.top, (f.cx, cat.y))

    # --- tulang punggung ---
    sx, bw = 22.0, 13.0
    bx = sx - bw / 2
    enc = d.box(bx, 15.6, bw, 2.2, "Encoder MLP  180 → 192", BLUE, fs=fs + 0.5)
    d.arrow((sx, cat.y + cat.h), enc.bottom)

    c0 = 19.2
    cont = d.container(12.6, c0, 17.6, 14.6)
    ln = d.box(bx, c0 + 1.8, bw, 1.7, "LayerNorm", YELLOW, fs=fs + 0.5)
    mix = d.box(bx, c0 + 4.7, bw, 3.0,
                "[ $z$,  $A_{\\rm maju}z$,  $A_{\\rm mundur}z$,  $A_{\\rm adp}z$ ]\nagregasi graf: 4 × 192 = 768",
                PEACH, fs=fs + 0.5)
    ffn = d.box(bx, c0 + 8.9, bw, 1.9, "FFN  768 → 384 → 192", BLUE, fs=fs + 0.5)
    add = d.box(bx, c0 + 12.0, bw, 1.7, "Add", YELLOW, fs=fs + 0.5)
    d.arrow(enc.top, ln.bottom)
    d.arrow(ln.top, mix.bottom)
    d.arrow(mix.top, ffn.bottom)
    d.arrow(ffn.top, add.bottom)
    # jalur residual ala Add & Norm
    d.arrow((sx, c0 + 0.9), (14.0, c0 + 0.9), (14.0, add.cy), add.left)
    d.text(11.0, cont.cy, "3×", fs=17)

    gr = d.box(31.4, mix.cy - 2.1, 11.8, 4.2,
               "Graf jalan: $A_{\\rm maju}$, $A_{\\rm mundur}$\n"
               "$A_{\\rm adp}$ = softmax(relu($E_1E_2^{\\rm T}$))",
               PINK, fs=fs - 0.5, linespacing=1.4)
    d.arrow(gr.left, mix.right)

    # --- head dan keluaran ---
    hln = d.box(bx, 35.0, bw, 1.7, "LayerNorm", YELLOW, fs=fs + 0.5)
    mlp = d.box(bx, 37.9, bw, 2.0, "MLP head → residual 3 horizon", LAVENDER, fs=fs + 0.5)
    outb = d.box(bx, 41.1, bw, 2.0, "nilai terakhir + residual × σ ruas", GREEN, fs=fs + 0.5)
    rule = d.box(bx, 44.3, bw, 2.0, "ruas mati → 0,  clip [0, 145]", YELLOW, fs=fs + 0.5)
    d.arrow(add.top, hln.bottom)
    d.arrow(hln.top, mlp.bottom)
    d.arrow(mlp.top, outb.bottom)
    d.arrow(outb.top, rule.bottom)
    d.arrow(rule.top, (sx, 47.6))
    d.text(sx, 48.3, "Prakiraan kecepatan 20 / 40 / 60 menit", fs=fs + 1)
    d.text(8.4, outb.cy, "nilai terakhir dan\nσ ruas (dari riwayat)", fs=fs - 0.5, ha="right")
    d.arrow((8.8, outb.cy), outb.left)

    d.caption("Figure 2: Arsitektur TrafficNet (sekitar 1,38 juta parameter). Lima kelompok fitur\n"
              "digabung, lalu diproses 3 blok spasial residual dengan tiga matriks ketetanggaan.",
              y=-0.4, fs=14)
    d.save(OUT / "fig2_task1_model.png")


# --------------------------------------------------------------------------
# Figure 3: hasil dan ablation Task 1
# --------------------------------------------------------------------------
def task1_ablation():
    fig, (a1, a2) = new_chart(7.8, 3.1, ncols=2, gridspec_kw=dict(width_ratios=[1.1, 1]))
    fig.subplots_adjust(left=0.17, right=0.98, wspace=0.62)
    ladder = ["Persistence", "Ridge global", "TrafficNet pertama", "TrafficNet terbaik",
              "Rata-rata 4 jaringan\n+ ridge", "Blend NNLS"]
    mse = [50.89, 36.34, 30.56, 29.19, 28.31, 28.06]
    hbar(a1, ladder, mse, highlight=5, fmt="{:.2f}", xlabel="Val MSE (lebih kecil lebih baik)",
         xlim=(0, 60), fontsize=9.5)
    a1.set_title("(a) Dari baseline ke blend akhir", fontsize=11.5)

    variants = ["Lengkap", "Tanpa teks\n(Δ −0.91)", "Tanpa graf\n(Δ +3.85)", "Tanpa augmentasi\n(Δ +0.21)"]
    vals = [29.79, 28.88, 33.65, 30.00]
    hbar(a2, variants, vals, fills=[YELLOW, BLUE, BLUE, BLUE], fmt="{:.2f}",
         xlabel="Val MSE, rata-rata 2 seed", xlim=(0, 42), fontsize=9.5)
    a2.set_title("(b) Ablation TrafficNet", fontsize=11.5)
    caption(fig, "Figure 3: Val MSE Task 1. Graf paling berpengaruh; efek teks dan augmentasi lebih kecil\n"
                 "dari noise antar seed (sekitar 1,2 MSE).")
    save_chart(fig, OUT / "fig3_task1_ablation.png")


# --------------------------------------------------------------------------
# Figure 4: pipeline Task 2
# --------------------------------------------------------------------------
def task2_pipeline():
    sans_math()
    d = Diagram(51.4, 24.6, scale=0.21)
    fs = 10.5

    # baris atas: rekonstruksi graf
    y1, h1 = 17.8, 3.0
    d.text(0.5, 22.9, "Rekonstruksi graf dari screenshot", fs=fs + 0.5, style="italic", ha="left")
    b1 = d.box(0.5, y1, 5.4, h1, "4.604\nscreenshot", PINK, fs=fs)
    b2 = d.box(7.5, y1, 6.0, h1, "Mask biru,\npotong, hash", YELLOW, fs=fs)
    b3 = d.box(15.1, y1, 6.0, h1, "TrOCR +\ncocok judul", PEACH, fs=fs)
    b4 = d.box(22.7, y1, 5.4, h1, "links.csv", BLUE, fs=fs)
    b5 = d.box(29.7, y1, 5.6, h1, "Perbaikan", YELLOW, fs=fs)
    b6 = d.box(36.9, y1, 6.2, h1, "links_v2.csv", BLUE, fs=fs)
    un = op_circle(d, 44.6, y1 + h1 / 2, "$\\cup$", fs=11.5)
    gph = d.box(46.4, y1, 4.6, h1, "Graf\nartikel", GREEN, fs=fs)
    for a, b in [(b1, b2), (b2, b3), (b3, b4), (b4, b5), (b5, b6), (b6, un), (un, gph)]:
        d.arrow(a.right, b.left)
    d.text(un.cx, 23.0, "tepi dari\ntrain", fs=fs - 1)
    d.arrow((un.cx, 22.1), un.top)
    notes = [(b1, "900 px, 5 GB"), (b2, "181.662 anchor\n→ 11.102 unik"), (b3, "difflib ≥ 0,84"),
             (b4, "92.595 tepi"), (b5, "buang halusinasi,\npecah anchor gabungan"), (b6, "94.670 tepi")]
    for b, t in notes:
        d.text(b.cx, y1 - 0.5, t, fs=fs - 1, va="top")

    # baris bawah: model dua tahap
    y2, h2 = 4.2, 3.6
    d.text(0.5, 9.6, "Model dua tahap", fs=fs + 0.5, style="italic", ha="left")
    st = d.box(0.5, y2, 7.2, h2, "State\n(saat ini, target)", PINK, fs=fs)
    sa = d.box(10.0, y2, 8.0, h2, "Tahap A\nBFS terbalik\ndari target", PEACH, fs=fs)
    sb = d.box(21.5, y2, 8.0, h2, "Tahap B\nLightGBM\n(700 pohon)", PEACH, fs=fs)
    am = d.box(32.6, y2, 6.2, h2, "argmax\nper state", LAVENDER, fs=fs)
    out = d.box(41.8, y2, 7.4, h2, "Artikel\nberikutnya", GREEN, fs=fs)
    for a, b in [(st, sa), (sa, sb), (sb, am), (am, out)]:
        d.arrow(a.right, b.left)
    d.text(sa.cx, y2 - 0.5, "simpan tautan berjarak minimum:\n~21 → ~5 kandidat, batas atas 0,966",
           fs=fs - 1, va="top")
    fb = d.box(18.5, 9.4, 14.0, 3.2,
               "59 fitur relasional, termasuk nilai rute\n"
               "$V[v]$ = base($v$) + γ · max $V[u]$", BLUE, fs=fs, linespacing=1.35)
    d.arrow(fb.bottom, sb.top)

    # graf mengalir ke tahap A dan ke fitur
    lane = 14.6
    d.arrow(gph.bottom, (gph.cx, lane), (sa.cx, lane), sa.top)
    d.arrow(gph.bottom, (gph.cx, lane), (fb.cx, lane), fb.top)

    d.caption("Figure 4: Task 2. Graf tautan direkonstruksi dari screenshot, lalu kandidat dipangkas\n"
              "dengan BFS terbalik (tahap A) dan diperingkat LightGBM dengan fitur nilai rute (tahap B).",
              y=0.6, fs=14)
    d.save(OUT / "fig4_task2_pipeline.png")


# --------------------------------------------------------------------------
# Figure 5: progres Task 2
# --------------------------------------------------------------------------
def task2_progress():
    steps = ["Modus\nglobal", "Klik paling\nsering", "Kandidat\nterpopuler", "Heuristik\njarak min.",
             "Tahap B v1", "+ graf v2,\nkemiripan", "+ lookahead\n2 langkah", "+ nilai rute\n(final)"]
    cv = [0.1164, 0.2573, 0.4013, 0.7220, 0.7511, 0.7656, 0.8356, 0.8748]
    lb_idx, lb = [4, 5, 7], [0.750, 0.773, 0.890]
    fig, ax = new_chart(7.4, 3.8)
    x = np.arange(len(steps))
    ax.axvspan(6.6, 7.4, color=PEACH, alpha=0.6, lw=0, zorder=0)
    ax.axvline(3.5, color=DARK["grey"], ls=":", lw=1.2, zorder=1)
    ax.text(1.5, 1.035, "Baseline", ha="center", va="center", fontsize=10.5, style="italic")
    ax.text(5.5, 1.035, "Model dua tahap", ha="center", va="center", fontsize=10.5, style="italic")
    ax.plot(x, cv, "-o", color=DARK["blue"], lw=1.8, ms=6, mfc=BLUE, mec="black", label="CV GroupKFold",
            zorder=3)
    ax.errorbar([7], [0.8748], yerr=[0.0083], fmt="none", ecolor="black", elinewidth=1.1, capsize=3.5,
                zorder=4)
    ax.plot(lb_idx, lb, "--s", color=DARK["peach"], lw=1.5, ms=6, mfc=PEACH, mec="black",
            label="Public LB", zorder=3)
    cv_off = [(0, 8), (-4, 8), (-4, 8), (-4, 8), (0, 8), (6, -15), (0, 8), (0, -17)]
    cv_ha = ["center", "right", "right", "right", "center", "left", "center", "center"]
    for xi, v, off, ha in zip(x, cv, cv_off, cv_ha):
        ax.annotate(f"{v:.4f}", (xi, v), textcoords="offset points", xytext=off, ha=ha, fontsize=9)
    for xi, v, off in zip(lb_idx, lb, [(0, -15), (0, 8), (0, 9)]):
        ax.annotate(f"{v:.3f}", (xi, v), textcoords="offset points", xytext=off, ha="center", fontsize=9,
                    color=DARK["peach"])
    ax.set_xticks(x)
    ax.set_xticklabels(steps, fontsize=9.5)
    ax.get_xticklabels()[7].set_fontweight("bold")
    ax.set_ylabel("Akurasi")
    ax.set_ylim(0, 1.08)
    ax.set_yticks(np.arange(0, 1.01, 0.2))
    ax.set_xlim(-0.5, 7.5)
    ax.legend(loc="lower right")
    caption(fig, "Figure 5: Akurasi Task 2 dari baseline sampai model final (CV 0.8748 ± 0.0083, public LB 0.890).")
    save_chart(fig, OUT / "fig5_task2_progress.png")


if __name__ == "__main__":
    playground()
    task1_model()
    task1_ablation()
    task2_pipeline()
    task2_progress()
