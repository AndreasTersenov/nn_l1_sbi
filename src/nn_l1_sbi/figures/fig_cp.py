"""fig:cp -- per-patch distributions of sigma(Om,s8,w0) and FoM3 over 9000 obs, 3 arms.

Data: committed per_patch_metrics (_data.load_per_patch). CPU. Double column (1x4 violins).
"""
import matplotlib
matplotlib.use("Agg")
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from nn_l1_sbi.figures.style import use_style, ARMS, WIDTH_DOUBLE
from nn_l1_sbi.figures import _data
HERE = _data.output_root() / "figures"   # $NN_L1_SBI_OUT/figures -- never the installed package
use_style()
ORDER = ["l1_product", "jointl1", "cnn"]
PANELS = [(r"$\sigma(\Omega_{\rm m})$", "sig", 0), (r"$\sigma(\sigma_8)$", "sig", 1),
          (r"$\sigma(w_0)$", "sig", 2), (r"$\mathrm{FoM}_3$", "fom", None)]


def main():
    _data.ensure_output_dir("figures")   # create HERE on use, not as an import side effect
    data = {a: _data.load_per_patch(a) for a in ORDER}
    pos = list(range(len(ORDER)))
    fig, axes = plt.subplots(1, 4, figsize=(WIDTH_DOUBLE, 0.40 * WIDTH_DOUBLE))
    for ax, (title, kind, j) in zip(axes, PANELS):
        series = [(data[a]["fom3"] if kind == "fom" else data[a]["sigma"][:, j]) for a in ORDER]
        if kind == "fom":
            hi = max(np.percentile(v, 99) for v in series)
            series = [np.clip(v, None, hi) for v in series]
        parts = ax.violinplot(series, positions=pos, showextrema=False, widths=0.8)
        for body, a in zip(parts["bodies"], ORDER):
            body.set_facecolor(ARMS[a]["color"]); body.set_alpha(0.45)
            body.set_edgecolor(ARMS[a]["color"]); body.set_linewidth(0.8)
        meds = []
        for i, a in enumerate(ORDER):
            med = float(np.median(series[i]))
            ax.hlines(med, i - 0.3, i + 0.3, color=ARMS[a]["color"], lw=2.0, zorder=3)
            meds.append((i, a, med))
        # median value just above the median line, centred on the violin
        y0, y1 = ax.get_ylim(); off = 0.022 * (y1 - y0)
        for i, a, med in meds:
            label = f"{med:.0f}" if kind == "fom" else f"{med:.3f}"
            ax.text(i, med + off, label, ha="center", va="bottom", fontsize=5.6,
                    color=ARMS[a]["color"], fontweight="bold", zorder=4)
        ax.set_xticks(pos); ax.set_xticklabels([])
        ax.set_ylabel(title)
        ax.tick_params(axis="x", length=0)
    handles = [plt.Line2D([], [], color=ARMS[a]["color"], lw=5, alpha=0.55, label=ARMS[a]["label"])
               for a in ORDER]
    fig.legend(handles=handles, loc="outside lower center", ncol=3, fontsize=7.5)
    out = HERE / "fig_cp_fom3_bars.pdf"
    fig.savefig(out); fig.savefig(HERE / "_preview" / "fig_cp.png", dpi=150)
    print("wrote", out, "| medians FoM3:", {a: round(data[a]["med_fom3"]) for a in ORDER})


if __name__ == "__main__":
    (HERE / "_preview").mkdir(parents=True, exist_ok=True)
    main()
