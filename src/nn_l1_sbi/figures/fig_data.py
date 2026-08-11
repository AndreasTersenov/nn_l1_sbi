"""fig:data -- the input channels for one representative 10deg patch.

4 tomographic auto-maps + the 6 convolution + 6 product patch-local cross-maps, all built from the
patch's own autos via flatsky_cross.build_channels_np. Reads the committed fiducial patch cache.
imshow panels rasterized inside the vector PDF. CPU. Double column.
"""
import os
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
import matplotlib
matplotlib.use("Agg")
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from nn_l1_sbi.figures.style import use_style, WIDTH_DOUBLE
from nn_l1_sbi.figures._data import MAPS, ensure_output_dir, output_root, require
from nn_l1_sbi.flatsky import flatsky_cross as fx
HERE = output_root() / "figures"    # $NN_L1_SBI_OUT/figures -- never the installed package
use_style()
FIDP = MAPS / "cosmo_fiducial_perm0.npz"
PAIRS = fx.cross_pairs(4)
PATCH = 90


def main():
    ensure_output_dir("figures")    # create HERE on use, not as an import side effect
    z = np.load(require(FIDP, "nn_l1_sbi.data.build_full_sphere_cross_cache",
                        what="the fiducial 10deg patch cache"))
    autos = z["patches"][PATCH, :, :, :4].astype(np.float32)
    autos = autos - autos.mean((0, 1), keepdims=True)
    chans = fx.build_channels_np(autos[None], "both")[0]   # (80,80,16): 4 auto, 6 conv, 6 product

    fig, axes = plt.subplots(3, 6, figsize=(WIDTH_DOUBLE, 0.52 * WIDTH_DOUBLE))
    rowlab = ["auto $\\kappa_i$", "convolution $\\kappa^{ij}_{\\rm conv}$", "product $\\kappa^{ij}_{\\rm prod}$"]
    # row 0: 4 autos (+2 blank); row 1: 6 conv; row 2: 6 product
    layout = [(0, b, f"$\\kappa_{{{b+1}}}$") for b in range(4)] + [(0, 4, None), (0, 5, None)]
    layout += [(1, 4 + k, f"{i+1}$\\times${j+1}") for k, (i, j) in enumerate(PAIRS)]
    layout += [(2, 10 + k, f"{i+1}$\\times${j+1}") for k, (i, j) in enumerate(PAIRS)]
    grid = {}
    for r in range(3):
        for c in range(6):
            grid[(r, c)] = axes[r, c]; axes[r, c].axis("off")
    col = {0: 0, 1: 0, 2: 0}
    for r, ch, lab in layout:
        c = col[r]; col[r] += 1
        ax = axes[r, c]
        if lab is None:
            continue
        ax.axis("on"); ax.set_xticks([]); ax.set_yticks([])
        m = chans[:, :, ch]
        v = np.percentile(np.abs(m), 99) + 1e-30
        ax.imshow(m, cmap="RdBu_r", vmin=-v, vmax=v, rasterized=True)
        ax.text(0.5, 1.02, lab, transform=ax.transAxes, ha="center", va="bottom", fontsize=7)
    for r in range(3):
        axes[r, 0].axis("on"); axes[r, 0].set_xticks([]); axes[r, 0].set_yticks([])
        axes[r, 0].set_ylabel(rowlab[r], fontsize=8)
    out = HERE / "fig_data_flatlocal.pdf"
    fig.savefig(out); fig.savefig(HERE / "_preview" / "fig_data.png", dpi=150)
    print("wrote", out)


if __name__ == "__main__":
    (HERE / "_preview").mkdir(parents=True, exist_ok=True)
    main()
