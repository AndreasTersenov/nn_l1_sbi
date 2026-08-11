"""fig:leak panel (b) -- the full-sphere cross-map channels are smooth and coherent.

Mollweide views of the four full-sphere auto-maps and the six full-sphere cross-maps (alm-products
on the whole sphere) from the committed snapshot: the cross-maps are visibly large-scale, so a 10deg
patch cut from them carries information from well beyond its footprint. Inherently raster (mollweide);
saved as a high-dpi PNG for the subfigure. CPU.
"""
import os
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
import matplotlib
matplotlib.use("Agg")
import numpy as np
import healpy as hp
import matplotlib.pyplot as plt
from pathlib import Path

from nn_l1_sbi.flatsky import flatsky_cross as fx
from nn_l1_sbi.figures.style import use_style, WIDTH_DOUBLE
from nn_l1_sbi.figures._data import MAPS, ensure_output_dir, output_root, require
HERE = output_root() / "figures"    # $NN_L1_SBI_OUT/figures -- never the installed package
use_style()
SNAP = MAPS / "fullsphere_nobnt_cosmo_fiducial_perm0.npz"
PAIRS = fx.cross_pairs(4)


def main():
    ensure_output_dir("figures")    # create HERE on use, not as an import side effect
    z = np.load(require(SNAP, "nn_l1_sbi.data.build_full_sphere_cross_cache",
                        what="the full-sphere snapshot"))
    FA = z["full_auto"]; FX = z["full_cross"]
    # shared symmetric clip per group (as in the original diagnostic): the cross-maps share one
    # scale so their large-scale coherence is visible; no smoothing.
    auto_clip = float(np.percentile(np.abs(FA), 99))
    cross_clip = float(np.percentile(np.abs(FX), 99))
    fig = plt.figure(figsize=(WIDTH_DOUBLE, 0.62 * WIDTH_DOUBLE))
    panels = [("auto $\\kappa_{%d}$" % (b + 1), FA[b], auto_clip) for b in range(4)]
    panels += [("cross $%d\\times%d$" % (i + 1, j + 1), FX[k], cross_clip) for k, (i, j) in enumerate(PAIRS)]
    for k, (title, m, clip) in enumerate(panels):
        hp.mollview(m.astype(np.float64), sub=(3, 4, k + 1), title=title, cbar=False, cmap="RdBu_r",
                    min=-clip, max=clip, margins=(0.002, 0.002, 0.002, 0.012), notext=True)
    out = HERE / "fig_fullsphere_maps.png"
    fig.savefig(out, dpi=300)
    fig.savefig(HERE / "_preview" / "fig_leak_b.png", dpi=120)
    print("wrote", out)


if __name__ == "__main__":
    (HERE / "_preview").mkdir(parents=True, exist_ok=True)
    main()
