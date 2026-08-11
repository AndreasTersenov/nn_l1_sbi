"""fig:corner -- mean-observation posterior on (Om,s8,w0), 3 arms, filled getdist triangle.

Reads the committed dump <data_root>/derived/contour_nobnt_samples.npz (produced once by
dump_contour_samples.py). The joint l1 is the 3-compressor ensemble, same as everywhere.

Design (Andreas, 2026-06-26):
  - analytical statistics (l1+product, joint l1) drawn SOLID; the learned CNN drawn DASHED
    (contour outline + 1-D marginal), so the analytical/learned split reads at a glance.
  - CNN contour on top.
  - FoM3 bar inset (every configuration: auto / +product / joint l1 / CNN) in the top-right
    corner, legend moved into the middle-right cell -- the talk layout, no overlap.
CPU. Single column.
"""
import matplotlib
matplotlib.use("Agg")
import numpy as np
from pathlib import Path

from nn_l1_sbi.figures.style import use_style, ARMS, WIDTH_SINGLE, getdist_settings
from nn_l1_sbi.figures._data import DERIVED, ensure_output_dir, output_root, require
HERE = output_root() / "figures"    # $NN_L1_SBI_OUT/figures -- never the installed package
use_style()
DATA = DERIVED / "contour_nobnt_samples.npz"
GEN_CONTOUR = "nn_l1_sbi.figures.dump_contour_samples (one-time GPU; --basis nobnt)"
# draw order: widest under -> CNN last so it sits on top. l1_auto is drawn only once it has been
# merged into the dump (gen_auto_contour.sh); until then the figure is the 3-arm version.
ORDER_ALL = ["l1_auto", "l1_product", "jointl1", "cnn"]
LS = {"l1_auto": "-", "l1_product": "-", "jointl1": "-", "cnn": "--"}   # analytical solid, learned dashed
# committed FoM3 (no-BNT), every configuration -- the bar inset. Single-compressor headline
# (joint l1 is its 3-compressor ensemble); robust to ensembling to <1.2% (auto 2429, product 3009).
FOM = [("l1_auto", 2448), ("l1_product", 3045), ("jointl1", 3371), ("cnn", 3326)]
BAR_LAB = {"l1_auto": "auto", "l1_product": "+prod", "jointl1": "joint", "cnn": "CNN"}


def main():
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from getdist import MCSamples, plots
    ensure_output_dir("figures")    # create HERE on use, not as an import side effect
    z = np.load(require(DATA, GEN_CONTOUR, what="the no-BNT mean-observation contour dump"))
    ORDER = [a for a in ORDER_ALL if a in z.files]   # auto arm appears once merged into the dump
    truth = z["truth"]; names = ["Om", "s8", "w0"]; labels = [r"\Omega_{\rm m}", r"\sigma_8", "w_0"]
    mcs = [MCSamples(samples=z[a], names=names, labels=labels, label=ARMS[a]["label"]) for a in ORDER]
    colors = [ARMS[a]["color"] for a in ORDER]
    lss = [LS[a] for a in ORDER]

    g = plots.get_subplot_plotter(width_inch=WIDTH_SINGLE)
    getdist_settings(g, WIDTH_SINGLE)
    markers = {n: float(truth[i]) for i, n in enumerate(names)} if truth.size == 3 else None
    # filled contours in `colors`; per-arm linestyle on the 2-D edges (contour_ls) and the 1-D
    # marginals (line_args -- also forces the colours, which getdist otherwise drops for arm 3)
    g.triangle_plot(mcs, filled=True, colors=colors,
                    contour_ls=lss, contour_lws=[1.0, 1.0, 1.1],
                    line_args=[dict(color=c, lw=1.0, ls=l) for c, l in zip(colors, lss)],
                    markers=markers, legend_labels=[])
    try:
        if g.legend is not None:
            g.legend.remove()
    except Exception:
        pass

    # ---- manual legend in the middle-right empty cell (talk layout) ----
    handles = [Line2D([0], [0], color=ARMS[a]["color"], ls=LS[a], lw=2.0) for a in ORDER]
    g.fig.legend(handles, [ARMS[a]["label"] for a in ORDER], loc="center",
                 bbox_to_anchor=(0.835, 0.555), frameon=False, fontsize=6.8,
                 handlelength=1.8, labelspacing=0.45, borderaxespad=0.0)

    # ---- FoM3 bar inset in the top-right corner (talk layout) ----------
    ax = g.fig.add_axes([0.605, 0.745, 0.368, 0.205])
    x = np.arange(len(FOM)); w = 0.78
    vals = [v for _, v in FOM]
    bcol = [ARMS[a]["color"] for a, _ in FOM]
    ax.bar(x, vals, w, color=bcol, edgecolor="0.25", linewidth=0.4)
    for xi, (a, v) in zip(x, FOM):
        ax.text(xi, v + 95, f"{v}", ha="center", va="bottom", fontsize=5.3, color=ARMS[a]["color"])
    ax.set_xticks(x); ax.set_xticklabels([BAR_LAB[a] for a, _ in FOM], fontsize=5.5)
    ax.set_ylabel(r"$\mathrm{FoM}_3$", fontsize=7, labelpad=1.5)
    ax.xaxis.set_ticks_position("bottom"); ax.yaxis.set_ticks_position("left")
    ax.tick_params(which="both", top=False, right=False)
    ax.tick_params(axis="y", labelsize=5.5, length=2, pad=1)
    ax.tick_params(axis="x", length=0, pad=1.2)
    ax.set_ylim(0, max(vals) * 1.20)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)

    out = HERE / "fig_corner_cnn_vs_l1.pdf"
    g.fig.savefig(str(out)); g.fig.savefig(str(HERE / "_preview" / "fig_corner.png"), dpi=200)
    print("wrote", out)


if __name__ == "__main__":
    (HERE / "_preview").mkdir(parents=True, exist_ok=True)
    main()
