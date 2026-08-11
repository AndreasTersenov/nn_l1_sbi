"""fig:sbc -- simulation-based calibration, 3 science params (Om, s8, w0), 4 arms.

Emitted for BOTH bases:
  no-BNT (main calibration):  fig_sbc_hist.pdf, fig_sbc_ecdf.pdf, fig_sbc_hist_overlay.pdf
  BNT counterpart:            fig_sbc_hist_bnt.pdf, fig_sbc_ecdf_bnt.pdf, fig_sbc_hist_overlay_bnt.pdf

Two presentations, both with the "expected under uniformity" band sized to the INDEPENDENT
test count (the 3 NDE seeds share identical test theta, so the unit is one seed ~600, not the
1800 pool; bars/curves = mean over the 3 seeds):
  (A) fig_sbc_ecdf*  -- ECDF-difference Fhat(r)-r + SIMULTANEOUS 99% band (Sailynoja+ 2022),
      4-line overlay (lines overlay cleanly). The shape-legible view (appendix).
  (B) fig_sbc_hist*  -- Talts+ 2018 rank histograms + 99% binomial band, as a FACETED GRID
      (arms=rows x params=cols; histograms do not overlay legibly for >3 arms). Main figure.
      fig_sbc_hist_overlay* keeps the old single-row overlay for comparison.

joint l1 = 3-compressor ensemble in both bases (the single-compressor BNT joint FAILS SBC;
the ensemble recovers it, as in no-BNT). Data: committed gate dumps via _data.load_gate_per_seed.
CPU. Double column.
"""
import matplotlib
matplotlib.use("Agg")
import numpy as np
from scipy.stats import binom
import matplotlib.pyplot as plt
from pathlib import Path

from nn_l1_sbi.figures.style import use_style, ARMS, WIDTH_DOUBLE
from nn_l1_sbi.figures import _data
HERE = _data.output_root() / "figures"   # $NN_L1_SBI_OUT/figures -- never the installed package
use_style()
ORDER = ["l1_auto", "l1_product", "jointl1", "cnn"]
PARAMS = [r"$\Omega_{\rm m}$", r"$\sigma_8$", r"$w_0$"]
NB = 20                       # histogram bins (Talts rec.: ~num_sbc_runs/30)
BASES = [("nobnt", ""), ("bnt", "_bnt")]


def _legend(ax, loc, fs=6.5):
    h = [plt.Line2D([], [], color=ARMS[a]["color"], lw=1.4, label=ARMS[a]["label"]) for a in ORDER]
    ax.legend(handles=h, loc=loc, fontsize=fs, handlelength=1.2, frameon=False)


def _sim_band(n, level=0.99, ngrid=400, ndraw=6000, seed=0):
    """Constant-width simultaneous band d: P(sup_r |Fhat(r) - r| > d) = 1 - level for n
    uniform draws (a Kolmogorov-type envelope -> a genuinely simultaneous test). MC, fixed
    seed for a deterministic figure."""
    rng = np.random.default_rng(seed)
    g = np.linspace(0, 1, ngrid)
    dmax = np.empty(ndraw)
    for i in range(ndraw):
        u = np.sort(rng.uniform(size=n))
        dmax[i] = np.abs(np.searchsorted(u, g, side="right") / n - g).max()
    return float(np.quantile(dmax, level))


def plot_ecdf(perseed, n_ind, suffix):
    g = np.linspace(0, 1, 400)
    dband = _sim_band(n_ind)
    fig, axes = plt.subplots(1, 3, figsize=(WIDTH_DOUBLE, 0.36 * WIDTH_DOUBLE), sharey=True)
    for d, (ax, p) in enumerate(zip(axes, PARAMS)):
        ax.axhspan(-dband, dband, color="0.88", zorder=0, lw=0)
        ax.axhline(0.0, color="k", ls="--", lw=0.8, alpha=0.7, zorder=1)
        for arm in ORDER:
            bs = perseed[arm]
            curves = [np.searchsorted(np.sort(bs[s][:, d]), g, side="right") / bs[s].shape[0] - g
                      for s in bs]
            ax.plot(g, np.mean(curves, 0), color=ARMS[arm]["color"], lw=1.3, zorder=2)
        ax.set_xlim(0, 1); ax.set_ylim(-0.095, 0.095); ax.set_xlabel(r"SBC rank $r$")
        ax.text(0.5, 0.078, p, ha="center", fontsize=9)
    axes[0].set_ylabel(r"$\widehat{F}(r) - r$")
    axes[0].text(0.04, -dband - 0.004, f"99% null (N={n_ind})", fontsize=5.6, color="0.4", va="top")
    _legend(axes[2], "lower left")
    out = HERE / f"fig_sbc_ecdf{suffix}.pdf"
    fig.savefig(out); fig.savefig(HERE / "_preview" / f"fig_sbc_ecdf{suffix}.png", dpi=150)
    plt.close(fig)
    print("wrote", out.name, f"(simultaneous band d99={dband:.3f}, N={n_ind})")


def plot_hist(perseed, n_ind, suffix):
    """Faceted grid (arms = rows, params = columns): the canonical SBC layout."""
    edges = np.linspace(0, 1, NB + 1); bw = 1.0 / NB
    lo = binom.ppf(0.005, n_ind, 1 / NB) / (n_ind * bw)
    hi = binom.ppf(0.995, n_ind, 1 / NB) / (n_ind * bw)
    nr = len(ORDER)
    fig, axes = plt.subplots(nr, 3, figsize=(WIDTH_DOUBLE, 0.205 * nr * WIDTH_DOUBLE),
                             sharex=True, sharey=True)
    for ai, arm in enumerate(ORDER):
        c = ARMS[arm]["color"]
        for d in range(3):
            ax = axes[ai, d]
            ax.axhspan(lo, hi, color="0.88", zorder=0, lw=0)
            ax.axhline(1.0, color="k", ls="--", lw=0.7, alpha=0.7, zorder=1)
            bs = perseed[arm]
            dens = np.mean([np.histogram(bs[s][:, d], bins=NB, range=(0, 1), density=True)[0]
                            for s in bs], axis=0)
            ax.stairs(dens, edges, color=c, lw=1.1, fill=True, alpha=0.55, zorder=2)
            ax.stairs(dens, edges, color=c, lw=1.0, zorder=3)
            ax.set_xlim(0, 1); ax.set_ylim(0, 1.85)
            if ai == 0:
                ax.set_title(PARAMS[d], fontsize=9)
        axes[ai, 0].set_ylabel(ARMS[arm]["label"], color=c, fontsize=8)
    for ax in axes[-1]:
        ax.set_xlabel("SBC rank")
    axes[0, 2].text(0.96, hi + 0.06, f"99% null (N={n_ind})", fontsize=5.6, color="0.4",
                    ha="right", va="bottom")
    fig.text(0.012, 0.5, "rank density", rotation=90, va="center", fontsize=8)
    out = HERE / f"fig_sbc_hist{suffix}.pdf"
    fig.savefig(out); fig.savefig(HERE / "_preview" / f"fig_sbc_hist{suffix}.png", dpi=150)
    plt.close(fig)
    print("wrote", out.name, f"(binomial band [{lo:.2f},{hi:.2f}], N={n_ind}, {nr}x3 grid)")


def plot_hist_combined():
    """Both bases OVERLAID in each cell: 4 arms (rows) x 3 params (cols); no-BNT filled+solid, BNT
    dashed outline, same arm colour (matches the fig:bnt solid=no-BNT / dashed=BNT convention). Same
    4x3 footprint as one grid but shows both bases. The two separate grids are kept.
    -> fig_sbc_hist_combined.pdf"""
    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D
    edges = np.linspace(0, 1, NB + 1); bw = 1.0 / NB
    PS = {b: {a: _data.load_gate_per_seed(a, b) for a in ORDER} for b in ("nobnt", "bnt")}
    n_ind = min(min(v.shape[0] for v in PS[b][a].values()) for b in PS for a in ORDER)
    lo = binom.ppf(0.005, n_ind, 1 / NB) / (n_ind * bw)
    hi = binom.ppf(0.995, n_ind, 1 / NB) / (n_ind * bw)

    def dens(bs, d):
        return np.mean([np.histogram(bs[s][:, d], bins=NB, range=(0, 1), density=True)[0]
                        for s in bs], axis=0)
    nr = len(ORDER)
    plt.rcParams["figure.constrained_layout.use"] = False
    fig = plt.figure(figsize=(WIDTH_DOUBLE, 0.205 * nr * WIDTH_DOUBLE), constrained_layout=False)
    gs = fig.add_gridspec(nr, 3, left=0.075, right=0.99, top=0.875, bottom=0.075,
                          wspace=0.07, hspace=0.16)
    ref = None
    for ai, arm in enumerate(ORDER):
        c = ARMS[arm]["color"]
        for d in range(3):
            ax = fig.add_subplot(gs[ai, d], sharex=ref, sharey=ref)
            ref = ref or ax
            ax.axhspan(lo, hi, color="0.88", zorder=0, lw=0)
            ax.axhline(1.0, color="k", ls="--", lw=0.6, alpha=0.6, zorder=1)
            ax.stairs(dens(PS["nobnt"][arm], d), edges, color=c, fill=True, alpha=0.20, lw=0, zorder=2)
            ax.stairs(dens(PS["nobnt"][arm], d), edges, color=c, lw=1.0, zorder=3)          # no-BNT solid
            ax.stairs(dens(PS["bnt"][arm], d), edges, color=c, lw=1.0, ls="--", zorder=4)    # BNT dashed
            ax.set_xlim(0, 1); ax.set_ylim(0, 1.95)
            ax.set_xticks([0, 0.5, 1]); ax.tick_params(labelsize=6, length=2)
            if ai == 0:
                ax.set_title(PARAMS[d], fontsize=8.5)
            if d == 0:
                ax.set_ylabel(ARMS[arm]["label"], color=c, fontsize=8)
            else:
                ax.tick_params(labelleft=False)
            if ai != nr - 1:
                ax.tick_params(labelbottom=False)
    h = [Patch(facecolor="0.55", alpha=0.5, edgecolor="0.3", label="no BNT"),
         Line2D([0], [0], color="0.3", lw=1.2, ls="--", label="BNT")]
    fig.legend(handles=h, loc="upper center", bbox_to_anchor=(0.53, 0.985), ncol=2,
               frameon=False, fontsize=8.5, handlelength=1.7, columnspacing=1.6)
    fig.text(0.53, 0.016, "SBC rank", ha="center", fontsize=8)
    fig.text(0.011, 0.47, "rank density", rotation=90, va="center", fontsize=8)
    out = HERE / "fig_sbc_hist_combined.pdf"
    fig.savefig(out); fig.savefig(HERE / "_preview" / "fig_sbc_hist_combined.png", dpi=170)
    plt.close(fig)
    print("wrote", out.name, f"(overlaid no-BNT/BNT, 4x3, band [{lo:.2f},{hi:.2f}], N={n_ind})")


def plot_hist_overlay(perseed, n_ind, suffix):
    """Old design kept for comparison: single row (params), all arms overlaid."""
    edges = np.linspace(0, 1, NB + 1); bw = 1.0 / NB
    lo = binom.ppf(0.005, n_ind, 1 / NB) / (n_ind * bw)
    hi = binom.ppf(0.995, n_ind, 1 / NB) / (n_ind * bw)
    fig, axes = plt.subplots(1, 3, figsize=(WIDTH_DOUBLE, 0.36 * WIDTH_DOUBLE), sharey=True)
    for d, (ax, p) in enumerate(zip(axes, PARAMS)):
        ax.axhspan(lo, hi, color="0.88", zorder=0, lw=0)
        ax.axhline(1.0, color="k", ls="--", lw=0.8, alpha=0.7, zorder=1)
        for arm in ORDER:
            bs = perseed[arm]
            dens = np.mean([np.histogram(bs[s][:, d], bins=NB, range=(0, 1), density=True)[0]
                            for s in bs], axis=0)
            ax.stairs(dens, edges, color=ARMS[arm]["color"], lw=1.2, zorder=2)
        ax.set_xlim(0, 1); ax.set_ylim(0, 1.85); ax.set_xlabel("SBC rank")
        ax.text(0.5, 1.7, p, ha="center", fontsize=9)
    axes[0].set_ylabel("rank density")
    axes[0].text(0.04, hi + 0.03, f"99% null (N={n_ind})", fontsize=5.6, color="0.4")
    _legend(axes[1], "lower center")
    out = HERE / f"fig_sbc_hist_overlay{suffix}.pdf"
    fig.savefig(out); fig.savefig(HERE / "_preview" / f"fig_sbc_hist_overlay{suffix}.png", dpi=150)
    plt.close(fig)
    print("wrote", out.name, f"(overlay, {len(ORDER)} arms, N={n_ind})")


def main():
    _data.ensure_output_dir("figures")   # create HERE on use, not as an import side effect
    for basis, suffix in BASES:
        perseed = {arm: _data.load_gate_per_seed(arm, basis) for arm in ORDER}
        n_ind = min(min(v.shape[0] for v in perseed[a].values()) for a in ORDER)
        print(f"--- basis={basis} (N_ind={n_ind}) ---")
        plot_ecdf(perseed, n_ind, suffix)
        plot_hist(perseed, n_ind, suffix)
        plot_hist_overlay(perseed, n_ind, suffix)
    plot_hist_combined()   # no-BNT + BNT in one grid (originals kept)


if __name__ == "__main__":
    (HERE / "_preview").mkdir(parents=True, exist_ok=True)
    main()
