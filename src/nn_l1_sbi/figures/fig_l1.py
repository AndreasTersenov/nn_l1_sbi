"""fig:l1 -- the starlet l1 statistics, now split into two figures.

(A) fig_l1_hist_vs_s8.pdf: the per-channel l1 histogram vs SNR for three representative channels
    (auto, conv, product) across the wavelet scales, coloured by sigma8 (cosmology sensitivity).
    Reads the committed l1_hist_vs_s8 cache.
(B) fig_l1_joint_cellmap.pdf: the JOINT l1 -- the 2-D (u_i,u_j) cell map for a representative bin
    pair and scale, whose 1-D projections ARE the per-channel l1. Reads the committed joint-l1
    fiducial datavector.
No GPU. (A) double column, (B) single column.
"""
import matplotlib
matplotlib.use("Agg")
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import Normalize
from matplotlib.gridspec import GridSpec
from pathlib import Path

from nn_l1_sbi.figures.style import use_style, ARMS, WIDTH_SINGLE, WIDTH_DOUBLE
from nn_l1_sbi.figures._data import DERIVED, ROOT, ensure_output_dir, output_root, require
HERE = output_root() / "figures"    # $NN_L1_SBI_OUT/figures -- never the installed package
use_style()
FC = Path(ROOT)
CACHE = DERIVED / "l1_hist_vs_s8_levels_cache.npz"
JOINT = FC / "analytical_nde_match/jointl1_nobnt_raw/fid.npz"
# the panel-(a) level cache has no surviving generator; it is a derived product of the raw
# l1 datavector caches (keys lvl_X = per-level l1 histograms, lvl_s8 = sigma8 per cosmology)
GEN_CACHE = ("no surviving generator (recon E section 9, tier D): rebuild the (lvl_X, lvl_s8) "
             "level cache from the raw l1 datavector caches -- see docs/FIGURE_MAP.md, fig:l1 (a)")
NS, NBINS, DROP = 5, 40, 1
CH = [("auto $\\kappa_4$", 3), ("conv $3\\times4$", 9), ("product $3\\times4$", 15)]
PAIR, SCALE, K = 5, 3, 10   # representative joint (u_i,u_j) cell map: real data with the clear
#                             two-peak diagonal (ell1-weighted) structure, like the design mockup


def plot_hist_grid():
    d = np.load(require(CACHE, GEN_CACHE, what="the fig:l1 panel-(a) level cache"))
    lvl_X, lvl_s8 = d["lvl_X"], d["lvl_s8"]
    feat = NS * NBINS
    lo_b, hi_b = DROP, NBINS - DROP
    xb = np.arange(lo_b, hi_b)
    norm = Normalize(vmin=lvl_s8.min(), vmax=lvl_s8.max())
    sm = cm.ScalarMappable(norm=norm, cmap="viridis"); sm.set_array([])

    plt.rcParams["figure.constrained_layout.use"] = False
    fig = plt.figure(figsize=(WIDTH_DOUBLE, 0.50 * WIDTH_DOUBLE), constrained_layout=False)
    gl = GridSpec(3, NS, figure=fig, left=0.075, right=0.985, top=0.92, bottom=0.20,
                  hspace=0.14, wspace=0.10)
    for r, (cname, ch) in enumerate(CH):
        for s in range(NS):
            ax = fig.add_subplot(gl[r, s])
            base = ch * feat + s * NBINS
            for b in range(len(lvl_s8)):
                ax.plot(xb, lvl_X[b, base + lo_b:base + hi_b], color=cm.viridis(norm(lvl_s8[b])), lw=0.7)
            ax.set_xticks([]); ax.set_yticks([])
            if r == 0:
                ax.set_title(f"scale {s+1}", fontsize=7.5)
            if s == 0:
                ax.set_ylabel(cname, fontsize=8)
    fig.text(0.53, 0.135, "SNR bin", ha="center", fontsize=8)
    cax = fig.add_axes([0.33, 0.075, 0.40, 0.03])
    cb = fig.colorbar(sm, cax=cax, orientation="horizontal")
    cb.set_label(r"$\sigma_8$", fontsize=8, labelpad=1); cb.ax.tick_params(labelsize=7)
    out = HERE / "fig_l1_hist_vs_s8.pdf"
    fig.savefig(out); fig.savefig(HERE / "_preview" / "fig_l1_hist.png", dpi=160)
    print("wrote", out)


def plot_joint_cellmap():
    plt.rcParams["figure.constrained_layout.use"] = False
    fig = plt.figure(figsize=(WIDTH_SINGLE, WIDTH_SINGLE), constrained_layout=False)
    gr = GridSpec(2, 2, figure=fig, width_ratios=[4, 1], height_ratios=[1, 4],
                  wspace=0.05, hspace=0.05, left=0.17, right=0.95, top=0.965, bottom=0.155)
    S = np.load(require(JOINT, "nn_l1_sbi.data.build_flatsky_joint_arm --out-fid "
                               "(scripts/slurm/build_joint_arm.slurm)",
                        what="the raw joint-l1 fiducial datavector"))["S"].mean(0).reshape(6, NS, K, K)
    H = S[PAIR, SCALE]
    axc = fig.add_subplot(gr[1, 0]); axt = fig.add_subplot(gr[0, 0], sharex=axc)
    axr = fig.add_subplot(gr[1, 1], sharey=axc)
    LIM = 5.0   # the actual (adaptive) SNR binning half-range; the old +-8 stretched the data flat
    ext = [-LIM, LIM, -LIM, LIM]; ticks = [-4, -2, 0, 2, 4]
    axc.imshow(H.T, origin="lower", extent=ext, cmap="viridis", aspect="equal", rasterized=True)
    axc.set_xlabel(r"$u_i$  (bin $i$ SNR)", fontsize=8.5)
    axc.set_ylabel(r"$u_j$  (bin $j$ SNR)", fontsize=8.5)
    axc.set_xticks(ticks); axc.set_yticks(ticks); axc.tick_params(labelsize=7.5)
    cen = np.linspace(-LIM, LIM, K)
    axt.plot(cen, H.sum(1), "-", color=ARMS["l1_product"]["color"], lw=1.4)
    axr.plot(H.sum(0), cen, "-", color=ARMS["l1_product"]["color"], lw=1.4)
    axt.tick_params(labelbottom=False); axr.tick_params(labelleft=False)
    axt.set_ylim(bottom=0); axr.set_xlim(left=0)
    out = HERE / "fig_l1_joint_cellmap.pdf"
    fig.savefig(out); fig.savefig(HERE / "_preview" / "fig_l1_joint.png", dpi=160)
    print("wrote", out)


def main():
    ensure_output_dir("figures")    # create HERE on use, not as an import side effect
    plot_hist_grid()
    plot_joint_cellmap()


if __name__ == "__main__":
    (HERE / "_preview").mkdir(parents=True, exist_ok=True)
    main()
