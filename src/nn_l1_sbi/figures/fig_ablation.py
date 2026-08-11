"""fig:ablation -- the cross-map / statistic completeness ladder, one matched RealNVP frame.

All points are the VMIM 10-D -> sbi_lens RealNVP 4x128 median over 9000 obs. auto-only, +product and
joint l1 are the committed Table 1 values; +conv and +both are computed identically here
(operator_realnvp, validated by reproducing product 3045 -> 3058). The per-channel l1 climbs from the
autos through the cross-map operators to the full pairwise joint l1. CPU. Single column.
"""
import matplotlib
matplotlib.use("Agg")
import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from nn_l1_sbi.figures.style import use_style, ARMS, WIDTH_SINGLE
from nn_l1_sbi.figures._data import A, GEN_SWEEP, ensure_output_dir, output_root, require
HERE = output_root() / "figures"    # $NN_L1_SBI_OUT/figures -- never the installed package
use_style()
OPR = Path(A) / "operator_realnvp"


def opfom(arm):
    return json.load(open(require(OPR / arm / "n9000" / "median_summary.json", GEN_SWEEP,
                                  what=f"the operator-ladder {arm} sweep")))["fom3"]


def main():
    ensure_output_dir("figures")    # create HERE on use, not as an import side effect
    conv, both = opfom("conv"), opfom("both")
    # committed Table 1 values (match main.tex); auto/product single-compressor, joint l1 = ensemble;
    # conv/both newly computed via the same (single-compressor) operator pipeline
    nodes = [(r"$\ell_1$ auto", 2448, "l1_auto", "v"),
             (r"+conv", conv, "l1_product", "o"),
             (r"+product", 3045, "l1_product", "o"),
             (r"+both", both, "l1_product", "o"),
             (r"joint $\ell_1$", 3371, "jointl1", "s")]
    x = np.arange(len(nodes)); ys = [n[1] for n in nodes]

    ylo, yhi = 2300, 3550; yr = yhi - ylo
    fig, ax = plt.subplots(figsize=(WIDTH_SINGLE, 0.92 * WIDTH_SINGLE))
    ax.plot(x, ys, "-", color="0.6", lw=1.0, zorder=1)
    for i, (lab, y, key, mk) in enumerate(nodes):
        ax.plot(i, y, marker=mk, color=ARMS[key]["color"], ms=7, zorder=3)
        dx = -0.34 if i == len(nodes) - 1 else 0.04   # joint l1 value sits left of the rightmost marker
        ax.annotate(f"{y:.0f}", (i, y), (i + dx, y - 95), fontsize=7.5,
                    color=ARMS[key]["color"], fontweight="bold")
    # (3) step-to-step FoM3 gains, small + neutral, above each segment
    for i in range(len(nodes) - 1):
        ya, yb = ys[i], ys[i + 1]
        xlab = i + 0.5 - (0.18 if i == 1 else 0.0)    # nudge the +14% (conv->product) label left
        ax.text(xlab, (ya + yb) / 2 + 0.022 * yr, f"+{(yb - ya) / ya * 100:.0f}%",
                ha="center", va="bottom", fontsize=6, color="0.45")
    ax.set_xticks(x); ax.set_xticklabels([n[0] for n in nodes], fontsize=7.5, rotation=18, ha="right")
    ax.set_xlim(-0.4, 4.5); ax.set_ylim(ylo, yhi)
    ax.set_ylabel(r"$\mathrm{FoM}_3 = 1/\sqrt{\det \mathbf{C}_3}$")
    ax.tick_params(axis="x", length=0)
    out = HERE / "fig_ablation.pdf"
    fig.savefig(out); fig.savefig(HERE / "_preview" / "fig_ablation.png", dpi=150)
    print(f"wrote {out} | auto 2448 conv {conv:.0f} product 3045 both {both:.0f} joint 3371")


if __name__ == "__main__":
    (HERE / "_preview").mkdir(parents=True, exist_ok=True)
    main()
