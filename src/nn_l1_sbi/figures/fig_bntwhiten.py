"""fig:bntwhiten -- whitening decomposition of the l1 BNT collapse.

no-BNT vs whitened (Q=(BB^T)^{-1/2}B) vs BNT FoM3 bars for the two l1 arms, + the deep-direction
ladder (1 / 2 deep channels). All numbers read from committed median_summary.json artifacts.
CPU. Single column.
"""
import matplotlib
matplotlib.use("Agg")
import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from nn_l1_sbi.figures.style import use_style, WIDTH_SINGLE
from nn_l1_sbi.figures._data import GEN_SWEEP, ROOT, ensure_output_dir, output_root, require
HERE = output_root() / "figures"    # $NN_L1_SBI_OUT/figures -- never the installed package
use_style()
FC = Path(ROOT)
WH = FC / "whiten_campaign"
# frame colours (no-BNT / whitened / BNT) -- distinct from the arm palette
C_NOBNT, C_WHITEN, C_BNT = "#0072B2", "#009E73", "#D55E00"
C_D1, C_D2 = "#CC79A7", "#E69F00"


def fom3(p):
    return json.load(open(require(Path(p) / "median_summary.json", GEN_SWEEP,
                                  what="a whitening/BNT campaign population sweep")))["fom3"]


def opt(p):
    p = FC / p
    return json.load(open(p))["fom3"] if p.exists() else None


def main():
    ensure_output_dir("figures")    # create HERE on use, not as an import side effect
    arms = []
    for op, label in (("none", r"$\ell_1$ auto"), ("product", r"$\ell_1$+product")):
        n = fom3(FC / f"population_sweep/flat_{op}")
        w = fom3(WH / f"population_sweep/l1_{op}")
        b = fom3(FC / f"bnt_campaign/population_sweep/l1_{op}")
        arms.append((label, n, w, b, (w - b) / (n - b)))
    d5 = opt("bntdeep_campaign/population_sweep/l1_none/median_summary.json")
    d6 = opt("bntdeep2_campaign/population_sweep/l1_none/median_summary.json")

    x = np.arange(len(arms)); width = 0.26
    fig, ax = plt.subplots(figsize=(WIDTH_SINGLE, 0.92 * WIDTH_SINGLE))
    for k, (vals, c, lab) in enumerate((
            ([a[1] for a in arms], C_NOBNT, "no BNT"),
            ([a[2] for a in arms], C_WHITEN, r"whitened $Q$"),
            ([a[3] for a in arms], C_BNT, "BNT"))):
        ax.bar(x + (k - 1) * width, vals, width, color=c, edgecolor="k", lw=0.4, label=lab)
    for xi, (label, n, w, b, rec) in enumerate(arms):
        ax.text(xi, w * 1.12, f"rec.\n{rec:.2f}", ha="center", fontsize=6.5, color=C_WHITEN)
        ax.text(xi + width, b * 1.25, f"{b/n:.2f}$\\times$", ha="center", fontsize=6.5, color=C_BNT)
    n0, b0 = arms[0][1], arms[0][3]
    for k, (val, c, lab) in enumerate(((d5, C_D1, "+1 deep"), (d6, C_D2, "+2 deep"))):
        if val is None:
            continue
        ax.bar((2 + k) * width, val, width, color=c, edgecolor="k", lw=0.4, label=lab)
        ax.text((2 + k) * width, val * 1.12, f"{(val-b0)/(n0-b0):.2f}", ha="center", fontsize=6.5, color=c)
    ax.set_yscale("log"); ax.set_ylim(top=max(a[1] for a in arms) * 4)
    ax.set_ylabel(r"$\mathrm{FoM}_3$ (9000-obs median)")
    ax.set_xticks(x); ax.set_xticklabels([a[0] for a in arms])
    ax.legend(frameon=False, ncol=3, loc="lower center", bbox_to_anchor=(0.5, 1.0), fontsize=6.5)
    out = HERE / "fig_bnt_whiten.pdf"
    fig.savefig(out); fig.savefig(HERE / "_preview" / "fig_bntwhiten.png", dpi=150)
    print("wrote", out, "| arms:", [(a[0], round(a[1]), round(a[2]), round(a[3]), round(a[4], 2)) for a in arms])


if __name__ == "__main__":
    (HERE / "_preview").mkdir(parents=True, exist_ok=True)
    main()
