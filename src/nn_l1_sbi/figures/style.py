"""Shared style for the L1_vs_CNN_Tomographic_SBI paper figures.

One palette + one type system for every figure. Import and call use_style() at
the top of each figure script; pull per-arm colour/linestyle/marker from ARMS so
the colour of each summary is identical across the whole figure set.

    from nn_l1_sbi.figures.style import use_style, ARMS, WIDTH_SINGLE, WIDTH_DOUBLE
    use_style()
    c = ARMS["jointl1"]["color"]

A&A: single column 88 mm, double column 180 mm; serif; vector PDF; ticks inward;
Okabe-Ito colours paired with linestyle/marker for grayscale-safe print.
"""
from pathlib import Path
import matplotlib.pyplot as plt

# ---- A&A physical widths (inches) ------------------------------------
WIDTH_SINGLE = 88.0 / 25.4   # 3.465 in  — one-column figure
WIDTH_INTER = 120.0 / 25.4   # 4.724 in  — \sidecaption / intermediate
WIDTH_DOUBLE = 180.0 / 25.4  # 7.087 in  — full-width figure*

# ---- fixed per-arm style (used EVERYWHERE) ---------------------------
# Okabe-Ito colour-blind-safe set. Per Andreas (2026-06-24): lines are SOLID,
# distinguished by COLOUR (and a marker on point-plots), not by dash pattern.
# NOTE (A&A grayscale): colour-only line plots can lose a distinction in print;
# we mitigate with markers on point/line plots and distinct-luminance colours,
# and keep colour the online cue. Flagged in DESIGN_SPEC.md.
ARMS = {
    "l1_auto":    dict(label=r"$\ell_1$ auto-only",       color="#666666", ls="-", marker="v"),
    "l1_product": dict(label=r"$\ell_1$+product",         color="#D55E00", ls="-", marker="o"),
    "jointl1":    dict(label=r"joint $\ell_1$",            color="#009E73", ls="-", marker="s"),
    "cnn":        dict(label=r"CNN",                       color="#0072B2", ls="-", marker="^"),
}

# Optional operator-ablation extra arms (conv / both) reuse the l1 family hue
# but are only ever drawn as ladder nodes, not as separate posterior arms.
OPERATORS = ["auto-only", "+conv", "+product", "+both"]

_STYLE_PATH = Path(__file__).resolve().parent / "paper.mplstyle"


def use_style():
    """Load the in-repo A&A mplstyle baseline."""
    plt.style.use(str(_STYLE_PATH))


def arm(name):
    """Return the (color, ls, marker, label) dict for an arm key."""
    return ARMS[name]


def getdist_settings(g, width_inch=WIDTH_SINGLE):
    """Apply consistent getdist plotter settings (call before triangle_plot)."""
    g.settings.fig_width_inch = width_inch
    g.settings.legend_fontsize = 8
    g.settings.axes_fontsize = 8
    g.settings.lab_fontsize = 9
    g.settings.axis_marker_lw = 0.8
    g.settings.linewidth = 1.2
    return g
