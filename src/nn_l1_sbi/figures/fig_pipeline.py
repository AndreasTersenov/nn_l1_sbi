"""fig:pipeline -- the estimator-controlled design schematic.

kappa maps -> {l1-based / CNN-VMIM} compressor -> 10-D summary -> shared RealNVP -> posterior,
every arm gated by TARP-DRP + SBC. A vector schematic; the only data are the four real auto-kappa
maps in the input stack (same patch/source as fig:data). Double column.
"""
import matplotlib
matplotlib.use("Agg")
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle
from pathlib import Path

from nn_l1_sbi.figures.style import use_style, ARMS, WIDTH_DOUBLE
from nn_l1_sbi.figures._data import MAPS, ensure_output_dir, output_root, require
HERE = output_root() / "figures"    # $NN_L1_SBI_OUT/figures -- never the installed package
use_style()
# real auto-kappa maps for the input stack (same fiducial patch as fig:data)
FIDP = MAPS / "cosmo_fiducial_perm0.npz"
PATCH = 90


def main():
    ensure_output_dir("figures")    # create HERE on use, not as an import side effect
    fig, ax = plt.subplots(figsize=(WIDTH_DOUBLE, 0.40 * WIDTH_DOUBLE))
    ax.set_xlim(0, 100); ax.set_ylim(0, 40); ax.axis("off")
    # stacked input: the four real auto-kappa maps (same size/position/stacking as the old blanks)
    autos = np.load(require(FIDP, "nn_l1_sbi.data.build_full_sphere_cross_cache",
                            what="the fiducial 10deg patch cache"))["patches"][PATCH, :, :, :4].astype(np.float32)
    autos = autos - autos.mean((0, 1), keepdims=True)
    v = np.percentile(np.abs(autos), 99) + 1e-30
    for k in range(4):
        x0, y0 = 4 + 1.4 * k, 17 - 1.4 * k
        ax.imshow(autos[:, :, k], extent=[x0, x0 + 9, y0, y0 + 9], cmap="RdBu_r",
                  vmin=-v, vmax=v, zorder=k, rasterized=True, aspect="auto")
        ax.add_patch(Rectangle((x0, y0), 9, 9, fill=False, ec="0.4", lw=0.6, zorder=k + 0.5))
    ax.text(9, 10.5, "Tomographic\n$\\kappa$ maps (4 bins)", ha="center", va="top", fontsize=8)

    def box(x, y, w, h, text, ec):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.3", fc="white", ec=ec, lw=1.4))
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=8)

    box(30, 25, 24, 8, r"$\ell_1$-based" "\n(analytical)", ARMS["l1_product"]["color"])
    box(30, 10, 24, 8, "CNN-VMIM\n(learned)", ARMS["cnn"]["color"])
    box(62, 17, 15, 8, "Flow NDE\n(RealNVP)", "0.3")
    ax.add_patch(FancyBboxPatch((84, 15), 12, 12, boxstyle="round,pad=0.1", fc="white", ec="0.4", lw=0.8))
    # banana-shaped posterior (a curved degeneracy), two NESTED filled credible levels: each level is
    # shorter AND thinner (own length Ls + width Ws) so the inner sits properly inside the outer rather
    # than being a thinner band that tapers to the same tips.
    def _banana(Ls, Ws):
        tt = np.linspace(-Ls, Ls, 160)
        sx = 90 + 3.9 * tt
        sy = 21 - 1.9 * tt + 1.7 * (tt ** 2 - 1 / 3)
        dx = np.gradient(sx); dy = np.gradient(sy); nl = np.hypot(dx, dy)
        nx, ny = -dy / nl, dx / nl
        w = Ws * np.sqrt(np.clip(1 - (tt / Ls) ** 2, 0.0, 1.0))
        return (np.concatenate([sx + nx * w, (sx - nx * w)[::-1]]),
                np.concatenate([sy + ny * w, (sy - ny * w)[::-1]]))
    for Ls, Ws, a in [(1.0, 1.35, 0.20), (0.66, 0.82, 0.45)]:
        bx, by = _banana(Ls, Ws)
        ax.fill(bx, by, color=ARMS["cnn"]["color"], alpha=a, lw=0)
    ax.text(90, 28.3, r"posterior $p(\theta\,|\,\mathrm{maps})$", ha="center", va="bottom", fontsize=8)

    arr = dict(arrowstyle="-|>", color="0.3", lw=1.1, mutation_scale=10)
    ax.add_patch(FancyArrowPatch((15, 16), (29, 29), **arr))
    ax.add_patch(FancyArrowPatch((15, 14), (29, 14), **arr))
    ax.add_patch(FancyArrowPatch((54, 29), (61.5, 22), **arr))
    ax.add_patch(FancyArrowPatch((54, 14), (61.5, 20), **arr))
    ax.add_patch(FancyArrowPatch((77, 21), (83.5, 21), **arr))
    gx = 57; gcx = gx + 21   # gate block under the right half; it validates the POSTERIOR calibration
    ax.add_patch(FancyBboxPatch((gx, 1), 42, 5, boxstyle="round,pad=0.2", fc="white",
                 ec=ARMS["jointl1"]["color"], lw=1.2))
    ax.text(gcx, 3.5, r"every arm gated: TARP-DRP + SBC $\checkmark$", ha="center", va="center",
            fontsize=8, color=ARMS["jointl1"]["color"])
    # TARP-DRP + SBC are posterior-calibration tests -> a vertical arrow drops from the middle of the
    # posterior box (centred at x=90)
    ax.add_patch(FancyArrowPatch((90, 14.6), (90, 6.4), arrowstyle="-|>",
                 color=ARMS["jointl1"]["color"], lw=1.0, mutation_scale=9))
    out = HERE / "fig_pipeline.pdf"
    fig.savefig(out); fig.savefig(HERE / "_preview" / "fig_pipeline.png", dpi=150)
    print("wrote", out)


if __name__ == "__main__":
    (HERE / "_preview").mkdir(parents=True, exist_ok=True)
    main()
