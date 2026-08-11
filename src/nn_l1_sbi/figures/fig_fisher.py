"""fig:fisher -- where the cosmology information lives in the wavelet (scale x SNR) plane.

For the per-bin starlet-l1 auto datavector (4 tomographic bins x 5 scales x 40 SNR bins), we regress
each bin's l1 value on the (standardised) cosmological parameters across the simulation grid, and
form the per-bin diagonal Fisher contribution F = (d<l1>/dtheta)^2 / var(l1) (the uncorrelated /
diagonal approximation -- a sensitivity map, not the full information). Summed over the 4 tomographic
bins and shown as a (scale x SNR) heatmap, one panel per parameter: it shows which wavelet scales and
which parts of the SNR (peak/void) tail carry the Omega_m / sigma8 / w0 response.

Reads the committed raw auto-l1 cache (flat_local_none, 800-D + theta). CPU. Double column (1x3).
"""
import matplotlib
matplotlib.use("Agg")
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from pathlib import Path

from nn_l1_sbi.figures.style import use_style, WIDTH_DOUBLE
from nn_l1_sbi.figures._data import ROOT, ensure_output_dir, output_root, require
HERE = output_root() / "figures"    # $NN_L1_SBI_OUT/figures -- never the installed package
use_style()

RAW = f"{ROOT}/l1_matrix/l1_none_cache/flat_local_none/l1_train.npz"
NBIN, NS, NSNR = 4, 5, 40       # tomographic bins, wavelet scales, SNR bins (800 = 4*5*40)
PARAMS = [(0, r"$\Omega_{\rm m}$"), (1, r"$\sigma_8$"), (2, r"$w_0$")]
SNR_HALF = 5.0                  # adaptive SNR half-range used for the l1 histograms (cf. fig_l1)


def main():
    ensure_output_dir("figures")    # create HERE on use, not as an import side effect
    z = np.load(require(RAW, "nn_l1_sbi.inference.npe_l1norm_cross_jaxili_nbody_tomo "
                             "(raw l1 datavector cache; scripts/slurm/train_l1both_build.slurm)",
                        what="the 800-D auto-l1 train cache"))
    X = z["x"].astype(np.float64)                       # (N, 800)
    th = z["theta"].astype(np.float64)[:, :3]           # Om, s8, w0
    ths = (th - th.mean(0)) / th.std(0)                 # standardise -> per-param Fisher comparable
    # multi-linear regression of every l1 bin on [1, Om, s8, w0]
    D = np.column_stack([np.ones(len(ths)), ths])       # (N, 4)
    beta, *_ = np.linalg.lstsq(D, X, rcond=None)        # (4, 800): rows = const, dOm, ds8, dw0
    resid = X - D @ beta
    var = resid.var(0) + 1e-30                          # per-bin scatter (unexplained by linear theta)
    fisher = (beta[1:] ** 2) / var                      # (3 params, 800) diagonal Fisher contribution
    F = fisher.reshape(3, NBIN, NS, NSNR).sum(1)        # sum over the 4 tomographic bins -> (3, NS, NSNR)

    plt.rcParams["figure.constrained_layout.use"] = False   # honour subplots_adjust + manual colorbar
    fig, axes = plt.subplots(1, 3, figsize=(WIDTH_DOUBLE, 0.34 * WIDTH_DOUBLE), sharey=True,
                             constrained_layout=False)
    ext = [-SNR_HALF, SNR_HALF, 0.5, NS + 0.5]          # SNR on x, scale on y
    for ax, (pi, plab) in zip(axes, PARAMS):
        m = F[pi] / F[pi].max()                         # relative within the panel
        im = ax.imshow(m, origin="lower", extent=ext, aspect="auto", cmap="magma",
                       vmin=0, vmax=1, rasterized=True)
        ax.set_title(plab, fontsize=9)
        ax.set_xlabel("SNR", fontsize=8.5)
        ax.set_xticks([-4, -2, 0, 2, 4]); ax.tick_params(labelsize=7.5)
        ax.axvline(0, color="w", lw=0.5, alpha=0.4)
    axes[0].set_ylabel("wavelet scale", fontsize=8.5); axes[0].set_yticks(range(1, NS + 1))
    fig.text(0.44, 0.925, r"auto starlet-$\ell_1$, summed over the 4 tomographic bins", ha="center",
             fontsize=8.5)
    fig.subplots_adjust(left=0.065, right=0.82, top=0.80, bottom=0.16, wspace=0.08)
    cax = fig.add_axes([0.84, 0.16, 0.014, 0.74])
    cb = fig.colorbar(cm.ScalarMappable(cmap="magma"), cax=cax)
    cb.set_label("rel. Fisher info (per panel)", fontsize=7.5, labelpad=2)
    cb.ax.tick_params(labelsize=7)
    out = HERE / "fig_fisher.pdf"
    fig.savefig(out); fig.savefig(HERE / "_preview" / "fig_fisher.png", dpi=160)
    print("wrote", out)


if __name__ == "__main__":
    (HERE / "_preview").mkdir(parents=True, exist_ok=True)
    main()
