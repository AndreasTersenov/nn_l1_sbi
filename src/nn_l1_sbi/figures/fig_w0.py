"""fig:w0 -- fiducial w0 offset vs patch size.

The l1 auto+cross population-mean w0 pull is -0.37 sigma at 20deg and shrinks to ~-0.10 at 10deg,
where it matches the CNN -> a flat-sky projection residual, not an l1 property. The 10deg means are
read from the committed per_patch_grid.csv; the 20deg -0.37 is the committed reference (Table C.2).
CPU. Single column.
"""
import matplotlib
matplotlib.use("Agg")
import csv
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from nn_l1_sbi.figures.style import use_style, ARMS, WIDTH_SINGLE
from nn_l1_sbi.figures._data import DERIVED, ensure_output_dir, output_root, require
HERE = output_root() / "figures"    # $NN_L1_SBI_OUT/figures -- never the installed package
use_style()
AN = DERIVED
GEN_GRID = ("no surviving generator (recon E section 9, tier D): the phase-C geometry per-patch "
            "grid, column pull_w_0 -- see docs/FIGURE_MAP.md, fig:w0")


def mean_pull(arm):
    rows = list(csv.DictReader(open(require(AN / "geometry" / arm / "per_patch_grid.csv", GEN_GRID,
                                            what=f"the {arm} 10deg per-patch grid"))))
    p = np.array([float(r["pull_w_0"]) for r in rows])
    p = p[np.isfinite(p)]
    return float(p.mean()), float(p.std() / np.sqrt(len(p)))


def main():
    ensure_output_dir("figures")    # create HERE on use, not as an import side effect
    l1_10, l1_10e = mean_pull("l1_auto_cross")
    cnn_10, cnn_10e = mean_pull("cnn_auto_cross")
    L1_20 = -0.37  # committed reference (Table C.2 / 20deg journey)

    fig, ax = plt.subplots(figsize=(WIDTH_SINGLE, 0.9 * WIDTH_SINGLE))
    xs = [20, 10]
    ax.axhline(0, color="k", lw=0.8, zorder=1)
    # l1 a+c: 20deg -> 10deg (the shrink)
    cl1 = ARMS["l1_product"]["color"]; ccn = ARMS["cnn"]["color"]
    ax.plot(xs, [L1_20, l1_10], "-o", color=cl1, lw=1.4, ms=6, label=r"$\ell_1$ auto+cross", zorder=3)
    ax.errorbar([10], [l1_10], yerr=[l1_10e], color=cl1, capsize=2, lw=1, zorder=3)
    # cnn a+c at 10deg
    ax.errorbar([10], [cnn_10], yerr=[cnn_10e], fmt="^", color=ccn, ms=6, capsize=2,
                label=r"CNN auto+cross", zorder=4)
    ax.annotate(r"$-0.37\sigma$", (20, L1_20), (18.3, L1_20 - 0.05), fontsize=8, color=cl1)
    ax.annotate(f"${l1_10:+.2f}\\sigma$", (10, l1_10), (10.4, l1_10 + 0.03), fontsize=8, color=cl1)
    ax.set_xlim(22, 8); ax.set_xticks([20, 10]); ax.set_xticklabels([r"$20\degree$", r"$10\degree$"])
    ax.set_xlabel("patch size"); ax.set_ylabel(r"fiducial $w_0$ pull  [bias$/\sigma$]")
    ax.set_ylim(-0.5, 0.2)
    ax.legend(loc="lower right", fontsize=7)
    out = HERE / "fig_w0_offset.pdf"
    fig.savefig(out); fig.savefig(HERE / "_preview" / "fig_w0.png", dpi=150)
    print(f"wrote {out} | l1 10deg {l1_10:+.3f}+-{l1_10e:.3f} | cnn 10deg {cnn_10:+.3f}")


if __name__ == "__main__":
    (HERE / "_preview").mkdir(parents=True, exist_ok=True)
    main()
