"""Gaya gambar ala paper untuk semua writeup di repo ini.

Diagram meniru gaya Figure 1 "Attention Is All You Need": kotak pastel bersudut
membulat dengan garis hitam tebal, panah hitam bersudut membulat, kontainer abu-abu
untuk blok yang diulang, dan caption serif di bawah gambar. Grafik memakai palet
yang sama dengan font serif dan tick ke dalam.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyBboxPatch, PathPatch, Polygon
from matplotlib.path import Path as MPath

# Palet diambil dari figur Transformer (Vaswani et al., 2017).
PINK = "#F7D3D7"      # embedding / input
PEACH = "#FCE2C2"     # attention / model utama
YELLOW = "#F2F5C4"    # norm / langkah ringan
BLUE = "#C6E8F6"      # feed forward / fitur
LAVENDER = "#DADCF2"  # linear / head
GREEN = "#CDEBD3"     # softmax / output
GREY = "#F1F1F1"      # kontainer
WHITE = "#FFFFFF"

# Versi gelap untuk garis dan titik di grafik.
DARK = {
    "pink": "#C9566A",
    "peach": "#D9822B",
    "yellow": "#A7A93A",
    "blue": "#2F86B8",
    "lavender": "#5F63B5",
    "green": "#3E9A5A",
    "grey": "#6E6E6E",
}
FILLS = [BLUE, PEACH, GREEN, PINK, LAVENDER, YELLOW]
LINES = [DARK["blue"], DARK["peach"], DARK["green"], DARK["pink"], DARK["lavender"], DARK["yellow"]]

SANS = "Arial"
SERIF = "Times New Roman"
LW = 2.0          # tebal garis kotak
ARROW_LW = 2.0    # tebal garis panah


def _save(fig, path, pad=0.08):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=200, bbox_inches="tight", pad_inches=pad, facecolor="white")
    plt.close(fig)
    print("saved", path)


# --------------------------------------------------------------------------
# Diagram
# --------------------------------------------------------------------------
@dataclass
class Box:
    x: float
    y: float
    w: float
    h: float

    @property
    def cx(self):
        return self.x + self.w / 2

    @property
    def cy(self):
        return self.y + self.h / 2

    @property
    def top(self):
        return (self.cx, self.y + self.h)

    @property
    def bottom(self):
        return (self.cx, self.y)

    @property
    def left(self):
        return (self.x, self.cy)

    @property
    def right(self):
        return (self.x + self.w, self.cy)

    def at(self, fx=0.5, fy=0.5):
        """Titik relatif di dalam kotak (0..1)."""
        return (self.x + fx * self.w, self.y + fy * self.h)


class Diagram:
    """Kanvas diagram dengan koordinat bebas (satuan = 1 'unit' kira-kira 0.25 inci)."""

    def __init__(self, width, height, scale=0.25):
        self.W, self.H = width, height
        self.fig = plt.figure(figsize=(width * scale, height * scale))
        self.ax = self.fig.add_axes([0, 0, 1, 1])
        self.ax.set_xlim(0, width)
        self.ax.set_ylim(0, height)
        self.ax.set_aspect("equal")
        self.ax.axis("off")
        self.scale = scale

    # ukuran font dalam pt, diskalakan terhadap ukuran kanvas
    def _fs(self, fs):
        return fs

    def box(self, x, y, w, h, text="", fill=WHITE, fs=11, radius=0.35, lw=LW, bold=False,
            color="black", ls="-", z=3, family=SANS, linespacing=1.15):
        patch = FancyBboxPatch(
            (x, y), w, h,
            boxstyle=f"round,pad=0,rounding_size={radius}",
            linewidth=lw, edgecolor="black", facecolor=fill, linestyle=ls, zorder=z,
        )
        self.ax.add_patch(patch)
        if text:
            self.ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs,
                         family=family, fontweight="bold" if bold else "normal", color=color,
                         zorder=z + 1, linespacing=linespacing)
        return Box(x, y, w, h)

    def container(self, x, y, w, h, fill=GREY, radius=0.8, lw=LW, ls="-", z=1):
        return self.box(x, y, w, h, fill=fill, radius=radius, lw=lw, ls=ls, z=z)

    def text(self, x, y, s, fs=12, ha="center", va="center", family=SANS, style="normal",
             color="black", weight="normal", z=6, rotation=0, linespacing=1.15):
        self.ax.text(x, y, s, fontsize=fs, ha=ha, va=va, family=family, style=style,
                     color=color, fontweight=weight, zorder=z, rotation=rotation,
                     linespacing=linespacing)

    def arrow(self, *pts, radius=0.45, lw=ARROW_LW, head=0.42, color="black", ls="-",
              both=False, z=2):
        """Polyline dengan sudut membulat dan kepala panah di titik terakhir."""
        pts = [tuple(p) for p in pts]
        verts, codes = [pts[0]], [MPath.MOVETO]
        for i in range(1, len(pts) - 1):
            p0, p1, p2 = pts[i - 1], pts[i], pts[i + 1]
            d1 = math.dist(p0, p1)
            d2 = math.dist(p1, p2)
            r = min(radius, d1 / 2, d2 / 2)
            a = (p1[0] + (p0[0] - p1[0]) * r / d1, p1[1] + (p0[1] - p1[1]) * r / d1)
            b = (p1[0] + (p2[0] - p1[0]) * r / d2, p1[1] + (p2[1] - p1[1]) * r / d2)
            verts += [a, p1, b]
            codes += [MPath.LINETO, MPath.CURVE3, MPath.CURVE3]
        # pendekkan segmen terakhir supaya ujung garis tidak menembus kepala panah
        end = pts[-1]
        prev = pts[-2]
        L = math.dist(prev, end)
        ux, uy = (end[0] - prev[0]) / L, (end[1] - prev[1]) / L
        stop = (end[0] - ux * head * 0.6, end[1] - uy * head * 0.6)
        verts.append(stop)
        codes.append(MPath.LINETO)
        self.ax.add_patch(PathPatch(MPath(verts, codes), fill=False, lw=lw, edgecolor=color,
                                    linestyle=ls, capstyle="butt", joinstyle="round", zorder=z))
        self._head(prev, end, head, color, z)
        if both:
            self._head(pts[1], pts[0], head, color, z)

    def _head(self, prev, end, size, color, z):
        L = math.dist(prev, end)
        ux, uy = (end[0] - prev[0]) / L, (end[1] - prev[1]) / L
        px, py = -uy, ux
        w = size * 0.5
        base = (end[0] - ux * size, end[1] - uy * size)
        tri = [end, (base[0] + px * w, base[1] + py * w), (base[0] - px * w, base[1] - py * w)]
        self.ax.add_patch(Polygon(tri, closed=True, facecolor=color, edgecolor=color, lw=0.8,
                                  zorder=z + 0.5))

    def line(self, *pts, radius=0.45, lw=ARROW_LW, color="black", ls="-", z=2):
        pts = [tuple(p) for p in pts]
        verts, codes = [pts[0]], [MPath.MOVETO]
        for i in range(1, len(pts) - 1):
            p0, p1, p2 = pts[i - 1], pts[i], pts[i + 1]
            d1, d2 = math.dist(p0, p1), math.dist(p1, p2)
            r = min(radius, d1 / 2, d2 / 2)
            a = (p1[0] + (p0[0] - p1[0]) * r / d1, p1[1] + (p0[1] - p1[1]) * r / d1)
            b = (p1[0] + (p2[0] - p1[0]) * r / d2, p1[1] + (p2[1] - p1[1]) * r / d2)
            verts += [a, p1, b]
            codes += [MPath.LINETO, MPath.CURVE3, MPath.CURVE3]
        verts.append(pts[-1])
        codes.append(MPath.LINETO)
        self.ax.add_patch(PathPatch(MPath(verts, codes), fill=False, lw=lw, edgecolor=color,
                                    linestyle=ls, zorder=z))

    def oplus(self, x, y, r=0.42, sym="+"):
        """Lingkaran operasi seperti simbol (+) pada positional encoding."""
        self.ax.add_patch(Circle((x, y), r, facecolor="white", edgecolor="black", lw=LW, zorder=4))
        k = r * 0.62
        if sym == "+":
            self.ax.plot([x - k, x + k], [y, y], color="black", lw=1.6, zorder=5)
            self.ax.plot([x, x], [y - k, y + k], color="black", lw=1.6, zorder=5)
        elif sym == "x":
            k *= 0.75
            self.ax.plot([x - k, x + k], [y - k, y + k], color="black", lw=1.6, zorder=5)
            self.ax.plot([x - k, x + k], [y + k, y - k], color="black", lw=1.6, zorder=5)
        else:
            self.ax.text(x, y, sym, ha="center", va="center", fontsize=10, family=SANS, zorder=5)
        return (x, y)

    def wave(self, x, y, r=0.85):
        """Lingkaran berisi gelombang sinus (ikon positional encoding)."""
        self.ax.add_patch(Circle((x, y), r, facecolor="white", edgecolor="black", lw=LW, zorder=4))
        import numpy as np
        t = np.linspace(-0.78, 0.78, 60)
        self.ax.plot(x + t * r, y + 0.42 * r * np.sin(t * math.pi / 0.78), color="black",
                     lw=1.5, zorder=5)
        return (x, y)

    def caption(self, s, y=None, fs=15):
        y = -1.2 if y is None else y
        self.ax.text(self.W / 2, y, s, ha="center", va="top", fontsize=fs, family=SERIF,
                     zorder=6)
        self.ax.set_ylim(min(self.ax.get_ylim()[0], y - 1.6), self.H)

    def save(self, path):
        _save(self.fig, path)


# --------------------------------------------------------------------------
# Grafik
# --------------------------------------------------------------------------
def chart_style():
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": [SERIF, "STIXGeneral", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "font.size": 11,
        "axes.titlesize": 12,
        "axes.labelsize": 11.5,
        "axes.linewidth": 1.0,
        "axes.edgecolor": "black",
        "axes.grid": True,
        "grid.color": "#D9D9D9",
        "grid.linewidth": 0.6,
        "grid.linestyle": "--",
        "axes.axisbelow": True,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.top": True,
        "ytick.right": True,
        "xtick.major.size": 4,
        "ytick.major.size": 4,
        "legend.frameon": True,
        "legend.edgecolor": "black",
        "legend.fancybox": False,
        "legend.framealpha": 1.0,
        "legend.fontsize": 10,
        "savefig.dpi": 200,
    })


def new_chart(w=6.4, h=3.6, ncols=1, nrows=1, **kw):
    chart_style()
    fig, ax = plt.subplots(nrows, ncols, figsize=(w, h), **kw)
    return fig, ax


def caption(fig, s, y=None, fs=12.5):
    """Caption serif di bawah grafik, ditempatkan di bawah semua label sumbu."""
    fig.canvas.draw()
    if y is None:
        renderer = fig.canvas.get_renderer()
        bottom = min(ax.get_tightbbox(renderer).y0 for ax in fig.axes)
        y = bottom / fig.bbox.height - 0.04
    fig.text(0.5, y, s, ha="center", va="top", fontsize=fs, family=SERIF)


def save_chart(fig, path):
    _save(fig, path, pad=0.12)


def hbar(ax, labels, values, fills=None, fmt="{:.4f}", highlight=None, xlabel="",
         xlim=None, fontsize=10):
    """Bar horizontal ala tabel ablation: batang pastel bertepi hitam + angka di ujung."""
    import numpy as np
    n = len(labels)
    y = np.arange(n)[::-1]
    fills = fills or [BLUE] * n
    if highlight is not None:
        fills = [PEACH if i == highlight else BLUE for i in range(n)]
    ax.barh(y, values, color=fills, edgecolor="black", linewidth=1.0, height=0.62, zorder=3)
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.tick_params(axis="y", length=0)
    ax.grid(axis="y", visible=False)
    if xlim:
        ax.set_xlim(*xlim)
    lo, hi = ax.get_xlim()
    for yi, v in zip(y, values):
        ax.text(v + (hi - lo) * 0.01, yi, fmt.format(v), va="center", ha="left",
                fontsize=fontsize, zorder=4)
    ax.set_xlabel(xlabel)
    return ax
