"""fig:leak panel (a) -- the cross-maps are large-scale / non-local.

Full-sphere alm-product cross-maps carry a large fraction of their variance on super-patch scales
(ell < 18, larger than a 10deg patch), unlike the auto-maps. Reads the committed full-sphere
snapshot and computes the angular variance spectrum. CPU. ~single column (half of the double-col fig).
"""
import os
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
import matplotlib
matplotlib.use("Agg")
import numpy as np
import healpy as hp
import matplotlib.pyplot as plt
from pathlib import Path

from nn_l1_sbi.figures.style import use_style, ARMS, WIDTH_SINGLE
from nn_l1_sbi.figures._data import MAPS, ensure_output_dir, output_root, require
HERE = output_root() / "figures"    # $NN_L1_SBI_OUT/figures -- never the installed package
use_style()
SNAP = MAPS / "fullsphere_nobnt_cosmo_fiducial_perm0.npz"
L_PATCH = 18  # ~180/10deg


def main():
    ensure_output_dir("figures")    # create HERE on use, not as an import side effect
    z = np.load(require(SNAP, "nn_l1_sbi.data.build_full_sphere_cross_cache",
                        what="the full-sphere snapshot"))
    FA = z["full_auto"]; FX = z["full_cross"]; lmax = int(z["lmax"])
    ell = np.arange(lmax + 1); w = 2 * ell + 1

    def varspec(m):
        cl = hp.anafast(m.astype(np.float64), lmax=lmax)
        v = w * cl; v[0] = 0
        return v / v.sum()

    auto_v = np.mean([varspec(FA[b]) for b in range(FA.shape[0])], axis=0)
    cross_v = np.mean([varspec(FX[k]) for k in range(FX.shape[0])], axis=0)
    fa = np.cumsum(auto_v)[L_PATCH]; fc = np.cumsum(cross_v)[L_PATCH]

    c_auto = ARMS["l1_auto"]["color"]; c_cross = ARMS["cnn"]["color"]
    fig, ax = plt.subplots(figsize=(WIDTH_SINGLE, 0.92 * WIDTH_SINGLE))
    ax.axvspan(1, L_PATCH, color="0.88", zorder=0)
    ax.plot(ell[1:], auto_v[1:], color=c_auto, lw=1.4, label=r"auto $\kappa$ (4 bins)")
    ax.plot(ell[1:], cross_v[1:], color=c_cross, lw=1.4, label=r"cross $\kappa^{ij}$ (6 pairs)")
    ax.axvline(L_PATCH, color="k", ls="--", lw=1.0)
    ax.set_xscale("log")
    ax.set_xlabel(r"multipole $\ell$")
    ax.set_ylabel(r"variance fraction per $\ell$  [$(2\ell+1)C_\ell$, norm.]")
    ax.text(0.04, 0.93, r"$\ell<18$ (super-patch):" "\n"
            f"auto {fa*100:.1f}%  vs  cross {fc*100:.1f}%",
            transform=ax.transAxes, fontsize=7, va="top")
    ax.legend(loc="upper right", fontsize=7)
    out = HERE / "fig_cross_leakage_scales.pdf"
    fig.savefig(out); fig.savefig(HERE / "_preview" / "fig_leak_a.png", dpi=150)
    print(f"wrote {out} | super-patch auto {fa*100:.1f}% cross {fc*100:.1f}%")


if __name__ == "__main__":
    (HERE / "_preview").mkdir(parents=True, exist_ok=True)
    main()
