"""Gambar untuk writeup ACTION 2026. Semua angka diambil dari ACTION-2026/README.md."""
from pathlib import Path

import numpy as np

from paperfig import *

OUT = Path(__file__).resolve().parents[1] / "action"


def note(d, x, y, s, ha="left", fs=10, **kw):
    """Catatan kecil di samping kotak."""
    d.text(x, y, s, fs=fs, ha=ha, va="center", **kw)


def caption_c(fig, s, fs=12.5, gap=0.04):
    """Seperti paperfig.caption, tetapi dipusatkan pada lebar semua sumbu beserta labelnya."""
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    bbs = [ax.get_tightbbox(r) for ax in fig.axes]
    x0, x1 = min(b.x0 for b in bbs), max(b.x1 for b in bbs)
    y0 = min(b.y0 for b in bbs)
    fig.text((x0 + x1) / 2 / fig.bbox.width, y0 / fig.bbox.height - gap, s, ha="center",
             va="top", fontsize=fs, family=SERIF)


# --------------------------------------------------------------------------
# Figure 1: pipeline penyisihan
# --------------------------------------------------------------------------
def penyisihan_pipeline():
    d = Diagram(44, 31.6, scale=0.21)
    fs = 11
    cx = 22.0

    inp = d.box(15.0, 0.6, 14.0, 2.4, "Komentar + caption postingan", PINK, fs=fs)
    note(d, 30.2, inp.cy, "train 23.297 komentar,\ntest 34.687 komentar")

    tn = d.box(15.0, 4.4, 14.0, 2.6, "textnorm\n(de-obfuscator)", YELLOW, fs=fs)
    note(d, 30.2, tn.cy, "NFKC, zero-width, homoglyph,\nfullwidth, leet, Caesar,\nelongasi, slang")
    note(d, 13.8, tn.cy, "kata di luar kosakata\n(train): 15,28% → 4,89%", ha="right")
    d.arrow(inp.top, tn.bottom)

    # model level 1
    by, bh, bw = 9.6, 3.0, 8.4
    xs = [1.0, 11.0, 21.0, 31.8]
    sail_c = d.container(xs[3] - 0.7, by - 0.7, bw + 1.4, bh + 1.4)
    d.text(sail_c.x + sail_c.w + 1.3, sail_c.cy, "5×", fs=13)
    l1 = [
        d.box(xs[0], by, bw, bh, "TF-IDF + LR\n& probe caption", BLUE, fs=fs),
        d.box(xs[1], by, bw, bh, "Gate none_tf\n(TF-IDF + LR)", BLUE, fs=fs),
        d.box(xs[2], by, bw, bh, "IndoBERT-\nlarge-p2", PEACH, fs=fs),
        d.box(xs[3], by, bw, bh, "Sailor2-3B\nLoRA (r = 16)", PEACH, fs=fs),
    ]
    bus_in = 8.0
    for b in l1:
        d.arrow(tn.top, (cx, bus_in), (b.cx, bus_in), b.bottom)

    st = d.box(15.0, 15.8, 14.0, 3.0, "LightGBM level 2\n(format panjang, 22 fitur)", LAVENDER, fs=fs)
    note(d, 30.2, st.cy, "logit tiap blok, rata-rata & rank\nper postingan, probe caption,\nnone_tf, panjang komentar, id label, dll.")
    note(d, 13.8, st.cy, "gain terbesar: Sailor2-3B\n(logit 48%, peluang 21%)", ha="right")
    bus_out = 14.5
    for b in l1:
        top = b.top if b is not l1[3] else (b.cx, b.y + b.h)
        d.arrow(top, (b.cx, bus_out), (cx, bus_out), st.bottom)

    sh = d.box(15.0, 20.2, 14.0, 2.6, "Shrinkage ke rata-rata\npostingan (w = 0,2)", YELLOW, fs=fs)
    note(d, 30.2, sh.cy, "rata-rata per postingan\ndari Sailor & IndoBERT")
    th = d.box(15.0, 24.2, 14.0, 2.2, "Threshold global 0,40", YELLOW, fs=fs)
    note(d, 13.8, th.cy, "lebih baik dari threshold\nper label (0,9367 vs 0,9300)", ha="right")
    out = d.box(15.0, 27.8, 14.0, 2.6, "9 label biner\n(relevan / tidak per pilar)", GREEN, fs=fs)
    d.arrow(st.top, sh.bottom)
    d.arrow(sh.top, th.bottom)
    d.arrow(th.top, out.bottom)

    d.caption("Figure 1: Pipeline penyisihan. Model level 1 (Sailor2-3B dengan lima seed) digabung oleh\n"
              "stacker LightGBM, disusutkan ke rata-rata postingan, lalu dipotong dengan threshold 0,40.",
              y=-0.4, fs=14)
    d.save(OUT / "fig1_penyisihan_pipeline.png")


# --------------------------------------------------------------------------
# Figure 2: riwayat versi penyisihan
# --------------------------------------------------------------------------
def penyisihan_history():
    vers = ["v1\nbaseline", "v2\ngate", "v3\nSailor-1B", "v4\nSailor-3B", "v5\nblok sendiri",
            "v6\n2 seed", "v7\n3 seed", "v8\nshrinkage", "v9\n5 seed"]
    cv = [0.9269, 0.9327, 0.9286, 0.9339, 0.9354, 0.9317, 0.9374, 0.9369, 0.9373]
    lb = [0.93278, 0.93483, 0.93649, 0.93764, 0.93609, 0.93761, 0.94137, 0.94282, 0.94451]
    priv = 0.94384
    fig, ax = new_chart(7.2, 4.0)
    x = np.arange(len(vers))
    ax.plot(x, cv, "-o", color=DARK["blue"], lw=1.8, ms=6, mfc=BLUE, mec="black",
            label="CV 5-fold per postingan", zorder=3)
    ax.plot(x, lb, "--s", color=DARK["peach"], lw=1.5, ms=6, mfc=PEACH, mec="black",
            label="Public LB", zorder=3)
    ax.plot([8], [priv], "*", ms=14, mfc=GREEN, mec="black", label="Private LB (v9)", zorder=4)
    # posisi label dipilih per titik supaya tidak tertimpa garis
    C, L = "center", "right"
    lb_off = [((0, 11), C), ((-4, 6), L), ((-4, 6), L), ((0, 7), C), ((-2, 11), C),
              ((-4, 6), L), ((-4, 6), L), ((-4, 6), L), ((0, 7), C)]
    cv_off = [((0, -14), C), ((0, 8), C), ((0, -14), C), ((5, -15), "left"), ((-3, -17), L),
              ((0, -14), C), ((0, 8), C), ((0, -14), C), ((0, -14), C)]
    for xi, v, (off, ha) in zip(x, lb, lb_off):
        ax.annotate(f"{v:.5f}", (xi, v), textcoords="offset points", xytext=off,
                    ha=ha, fontsize=8.5)
    for xi, v, (off, ha) in zip(x, cv, cv_off):
        ax.annotate(f"{v:.4f}", (xi, v), textcoords="offset points", xytext=off,
                    ha=ha, fontsize=8.5)
    ax.annotate(f"{priv:.5f}", (8, priv), textcoords="offset points", xytext=(0, -18),
                ha="center", fontsize=8.5)
    ax.set_xticks(x)
    ax.set_xticklabels(vers, fontsize=9.5)
    ax.set_ylabel("Rata-rata macro-F1 biner")
    ax.set_ylim(0.9245, 0.9475)
    ax.set_xlim(-0.5, 8.5)
    ax.legend(loc="upper left")
    caption_c(fig, "Figure 2: CV dan public leaderboard tiap versi penyisihan.")
    save_chart(fig, OUT / "fig2_penyisihan_history.png")


# --------------------------------------------------------------------------
# Figure 3: pipeline final dua tahap
# --------------------------------------------------------------------------
def final_pipeline():
    d = Diagram(44, 27.6, scale=0.21)
    fs = 11
    cx, xl, xr = 22.0, 11.5, 32.5

    inp = d.box(15.0, 0.6, 14.0, 2.8, "Komentar + caption\n(setelah textnorm)", PINK, fs=fs)

    # tahap 1 dan tahap 2
    cy0, ch = 5.6, 8.4
    lc = d.container(xl - 6.7, cy0, 13.4, ch)
    rc = d.container(xr - 6.7, cy0, 13.4, ch)
    d.text(lc.x - 0.5, lc.cy, "Tahap 1\nrelevansi", fs=fs, ha="right", style="italic")
    d.text(rc.x + rc.w + 0.5, rc.cy, "Tahap 2\nsentimen", fs=fs, ha="left", style="italic")
    bw = 11.6
    stk = d.box(xl - bw / 2, 6.4, bw, 2.8, "Stacker v9 + shrinkage\n(dari penyisihan)", PEACH, fs=fs)
    q = d.box(xl - bw / 2, 10.4, bw, 2.8, "Peluang relevan Q\nper sel pilar", BLUE, fs=fs)
    sai = d.box(xr - bw / 2, 6.4, bw, 2.8, "Sailor2-3B LoRA\n(20.429 komentar relevan)", PEACH, fs=fs - 0.5)
    sig = d.box(xr - bw / 2, 10.4, bw, 2.8, "3 sigmoid per komentar\n[Neg, Net, Pos]", LAVENDER, fs=fs)
    bus = 4.5
    d.arrow(inp.top, (cx, bus), (xl, bus), stk.bottom)
    d.arrow(inp.top, (cx, bus), (xr, bus), sai.bottom)
    d.arrow(stk.top, q.bottom)
    d.arrow(sai.top, sig.bottom)

    # penggabungan
    dec = d.box(xl - 4.6, 15.4, 9.2, 2.6, "Q < 0,40 ?", YELLOW, fs=fs + 0.5)
    am = d.box(xr - bw / 2 - 0.4, 15.4, bw + 0.8, 2.6, "argmax(log p + bias)\nbias = [0; 1,1; 0,9]", YELLOW, fs=fs)
    d.arrow(q.top, dec.bottom)
    d.arrow(sig.top, am.bottom)
    d.arrow(dec.right, am.left)
    d.text((dec.x + dec.w + am.x) / 2, dec.cy + 0.6, "tidak", fs=10.5, style="italic")

    tr = d.box(xl - 5.0, 19.8, 10.0, 2.2, "TidakRelevan", GREEN, fs=fs)
    sent = d.box(xr - bw / 2, 19.8, bw, 2.2, "Negatif / Netral / Positif", GREEN, fs=fs)
    d.arrow(dec.top, tr.bottom)
    d.text(xl + 0.4, (dec.y + dec.h + tr.y) / 2, "ya", fs=10.5, style="italic", ha="left")
    d.arrow(am.top, sent.bottom)

    out = d.box(cx - 8.0, 24.4, 16.0, 2.4, "9 pilar × 4 kelas", GREEN, fs=fs + 0.5)
    mid = 23.2
    d.arrow(tr.top, (xl, mid), (cx, mid), out.bottom)
    d.arrow(sent.top, (xr, mid), (cx, mid), out.bottom)

    d.caption("Figure 3: Pipeline final dua tahap. Relevansi diambil dari stacker penyisihan, "
              "sentimen diprediksi\nsekali per komentar, lalu keduanya digabung per sel pilar.",
              y=-0.4, fs=14)
    d.save(OUT / "fig3_final_pipeline.png")


# --------------------------------------------------------------------------
# Figure 4: tangga CV tahap sentimen + submission final
# --------------------------------------------------------------------------
def final_ladder():
    steps = ["Semua sel relevan → Negatif", "TF-IDF + LR", "TF-IDF + LR + bias",
             "IndoBERT-large-p2", "IndoBERT-large-p2 + bias", "Sailor2-3B LoRA",
             "Sailor2-3B LoRA + bias (final)"]
    vals = [0.4375, 0.6124, 0.6374, 0.6878, 0.6937, 0.7208, 0.7362]

    subs = ["2 encoder + stage 2", "IndoBERT + TF-IDF", "3 encoder",
            "Sailor2-3B LoRA (final)", "Sailor × 4 encoder"]
    cv = [0.6925, 0.6958, 0.7123, 0.7362, 0.7462]
    pub = [0.70092, 0.71206, 0.72040, 0.75788, 0.75659]
    prv = [0.70387, 0.71206, 0.72298, 0.76116, 0.75775]

    fig, (a1, a2) = new_chart(7.0, 6.6, nrows=2, gridspec_kw={"height_ratios": [1.0, 1.25],
                                                               "hspace": 0.42})
    hbar(a1, steps, vals, highlight=len(steps) - 1, xlim=(0.40, 0.79),
         xlabel="Macro-F1 4 kelas (CV)", fontsize=9.5)
    a1.set_title("(a) Tangga CV tahap sentimen", loc="left", fontsize=11.5)

    n = len(subs)
    y = np.arange(n)[::-1].astype(float)
    h = 0.25
    series = [(cv, BLUE, "CV"), (pub, PEACH, "Public LB"), (prv, GREEN, "Private LB")]
    for k, (vals_k, col, lab) in enumerate(series):
        yy = y + (1 - k) * h
        a2.barh(yy, vals_k, height=h, color=col, edgecolor="black", linewidth=0.9, zorder=3, label=lab)
        for yi, v in zip(yy, vals_k):
            a2.text(v + 0.0012, yi, f"{v:.4f}" if lab == "CV" else f"{v:.5f}", va="center",
                    ha="left", fontsize=8.5, zorder=4)
    a2.set_yticks(y)
    a2.set_yticklabels(subs)
    a2.tick_params(axis="y", length=0)
    a2.grid(axis="y", visible=False)
    a2.set_xlim(0.68, 0.79)
    a2.set_ylim(-0.6, n - 0.4)
    a2.set_xlabel("Macro-F1 4 kelas")
    a2.legend(loc="upper right", fontsize=9.5)
    a2.set_title("(b) Lima submission final", loc="left", fontsize=11.5)
    # tandai submission terpilih
    a2.get_yticklabels()[3].set_fontweight("bold")
    caption_c(fig, "Figure 4: Kenaikan CV tahap sentimen (a) dan skor lima submission final (b).", gap=0.03)
    save_chart(fig, OUT / "fig4_final_ladder.png")


if __name__ == "__main__":
    penyisihan_pipeline()
    penyisihan_history()
    final_pipeline()
    final_ladder()
