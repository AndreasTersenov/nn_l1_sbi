"""fig:bnt -- the BNT frame, split into two panels (analytic | learned), talk style.

Family split (Andreas, 2026-06-26), per-arm no-BNT/BNT style (matches the talk M3 4-way):
each arm is shown twice -- no-BNT as a FILLED SOLID contour and BNT as an UNFILLED DASHED contour in
the SAME base colour (distinguished by being dashed, NOT by a darker shade, so the hue stays legible
and the rings are not mistaken for inner levels of another arm). Thin outlines and marginals. A
line-style key (solid = no BNT, dashed = BNT) plus the colour-per-arm key. In the analytic panels each
BNT arm is drawn as a SINGLE outer-95% dashed ring (outer_ring mode) to avoid a tangle of inner dashes.

  (a) ANALYTIC l1 under BNT  -> fig_bnt_analytic.pdf
      auto / l1+product / joint l1, each no-BNT (filled 2-level) + BNT (single dashed 95% ring). The
      per-channel auto and +product balloon under BNT; the complete joint l1 barely moves.
  (b) LEARNED CNN under BNT  -> fig_bnt_learned.pdf
      CNN no-BNT (filled) + BNT (dashed), near-coincident. Carries the FoM3 retention-ladder inset
      (all four configs: x0.16 / x0.24 / x0.72 / x0.96).

Reads the committed contour dumps; FoM3 numbers committed. BNT per-channel l1 (auto/+product) uses
the 3-compressor ENSEMBLE FoM3 (auto 388, product 718; the calibrated value -- see FOM note below).
CPU. Each panel is one column; placed side by side as a figure* in main.tex.
"""
import matplotlib
matplotlib.use("Agg")
import numpy as np
from pathlib import Path

from nn_l1_sbi.figures.style import use_style, ARMS, WIDTH_SINGLE, getdist_settings
from nn_l1_sbi.figures._data import DERIVED, ensure_output_dir, output_root, require
HERE = output_root() / "figures"    # $NN_L1_SBI_OUT/figures -- never the installed package
use_style()
NB = DERIVED / "contour_nobnt_samples.npz"
BN = DERIVED / "contour_bnt_samples.npz"
GEN_CONTOUR = "nn_l1_sbi.figures.dump_contour_samples (one-time GPU; --basis nobnt,bnt)"
# committed FoM3 (no-BNT, BNT, retention). no-BNT = single-compressor headline (joint l1 = ensemble);
# BNT per-channel l1 (auto, +product) = 3-compressor ENSEMBLE, the calibrated value (single BNT is
# over-confident): auto 388 (vs single 425), product 718 (vs 779); joint 2424; CNN 3186 single.
FOM = {"l1_auto": (2448, 388, 0.16), "l1_product": (3045, 718, 0.24),
       "jointl1": (3371, 2424, 0.72), "cnn": (3326, 3186, 0.96)}
LW = 0.9          # no-BNT outlines + marginals (talk style)
# BNT ring uses the SAME base hue as its no-BNT arm (distinguished by being dashed, not by a darker
# shade), so e.g. the joint-l1 ring clearly reads as green and is not mistaken for an inner (68%)
# level of the grey auto ring.
BNT_LW = 0.8


def _retention_bar(g, arms, rect):
    """FoM3 no-BNT (faint) -> BNT (solid) bars with x-retention labels, in figure rect."""
    ax = g.fig.add_axes(rect)
    x = np.arange(len(arms)); w = 0.38
    ax.bar(x - w / 2, [FOM[a][0] for a in arms], w,
           color=[ARMS[a]["color"] for a in arms], alpha=0.42)
    ax.bar(x + w / 2, [FOM[a][1] for a in arms], w,
           color=[ARMS[a]["color"] for a in arms], alpha=1.0)
    for i, a in enumerate(arms):
        ax.text(i, FOM[a][1] + 130, rf"$\times${FOM[a][2]:.2f}", ha="center",
                fontsize=7, color=ARMS[a]["color"], fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels([{"l1_auto": "auto", "l1_product": r"$\ell_1$+p",
                         "jointl1": "joint", "cnn": "CNN"}[a] for a in arms], fontsize=7)
    ax.set_ylabel(r"$\mathrm{FoM}_3$", fontsize=8, labelpad=2)
    ax.set_title(r"no BNT (faint) $\rightarrow$ BNT (solid)", fontsize=7.5)
    ax.xaxis.set_ticks_position("bottom"); ax.yaxis.set_ticks_position("left")
    ax.tick_params(which="both", top=False, right=False)
    ax.tick_params(axis="y", labelsize=6.5, length=2.5)
    ax.tick_params(axis="x", length=0, pad=2)
    ax.set_ylim(0, 3900)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


def _style_legend(g, anchor, fontsize=8):
    """The solid=no BNT / dashed=BNT line-style key (neutral grey)."""
    from matplotlib.lines import Line2D
    h = [Line2D([0], [0], color="0.3", lw=1.3, ls="-", label="no BNT"),
         Line2D([0], [0], color="0.3", lw=1.3, ls="--", label="BNT")]
    leg = g.fig.legend(handles=h, loc="center", bbox_to_anchor=anchor, frameon=False,
                       fontsize=fontsize, handlelength=2.0, labelspacing=0.4)
    g.fig.add_artist(leg)
    return leg


def main():
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from getdist import MCSamples, plots
    ensure_output_dir("figures")    # create HERE on use, not as an import side effect
    znb = np.load(require(NB, GEN_CONTOUR, what="the no-BNT mean-observation contour dump"))
    zbn = np.load(require(BN, GEN_CONTOUR, what="the BNT mean-observation contour dump"))
    truth = znb["truth"]; names = ["Om", "s8", "w0"]; labels = [r"\Omega_{\rm m}", r"\sigma_8", "w_0"]
    markers = {n: float(truth[i]) for i, n in enumerate(names)} if truth.size == 3 else None

    def mc(z, arm, suffix=""):
        return MCSamples(samples=z[arm], names=names, labels=labels, label=ARMS[arm]["label"] + suffix)

    def _limits(arms):
        """Generous per-param limits over no-BNT + BNT samples so the ballooning BNT does not clip."""
        lims = {}
        for k, n in enumerate(names):
            v = np.concatenate([znb[a][:, k] for a in arms] + [zbn[a][:, k] for a in arms])
            lo, hi = np.percentile(v, [0.3, 99.7]); pad = 0.05 * (hi - lo)
            lims[n] = (float(lo - pad), float(hi + pad))
        return lims

    def panel(arms, out, color_anchor=None, style_anchor=None, bar_rect=None, bar_arms=None,
              getdist_legend=False, leg_fs=8.5, axes_fs=None, lab_fs=None, outer_ring=False):
        g = plots.get_subplot_plotter(width_inch=WIDTH_SINGLE)
        getdist_settings(g, WIDTH_SINGLE)
        g.settings.linewidth = LW
        if axes_fs:
            g.settings.axes_fontsize = axes_fs
        if lab_fs:
            g.settings.lab_fontsize = lab_fs

        if outer_ring:
            # 2-pass: no-BNT filled (2 levels, size-ordered largest->back) via triangle_plot, then
            # each BNT arm overlaid as a SINGLE outer-95% dashed ring -- removes the inner dashes that
            # tangle the core. Limits are widened so the BNT balloon does not clip.
            nb_arms = sorted(arms, key=lambda a: FOM[a][0])     # largest no-BNT first (back)
            nb_roots = [mc(znb, a, " no BNT") for a in nb_arms]
            g.triangle_plot(nb_roots, filled=True, colors=[ARMS[a]["color"] for a in nb_arms],
                            contour_lws=[LW] * len(nb_arms),
                            line_args=[dict(color=ARMS[a]["color"], lw=LW, ls="-") for a in nb_arms],
                            markers=markers, legend_labels=[], param_limits=_limits(arms))
            par = g.get_param_array(nb_roots[0], names)
            # draw BNT rings largest-first so joint l1 (tightest) ends on top and stays visible
            for a in sorted(arms, key=lambda a: FOM[a][1]):
                broot = mc(zbn, a, " BNT")
                for i in range(len(names)):
                    for j in range(i):
                        dens = g.sample_analyser.get_density_grid(broot, par[j], par[i],
                                                                  conts=g.settings.num_plot_contours)
                        g.add_2d_contours(broot, density=dens, contour_levels=dens.getContourLevels([0.95]),
                                          ax=g.subplots[i][j], filled=False, color=ARMS[a]["color"], ls="--",
                                          lw=BNT_LW, add_legend_proxy=False)
                    g.add_1d(broot, par[i], ax=g.subplots[i][i], color=ARMS[a]["color"], ls="--", lw=BNT_LW)
        else:
            # single triangle_plot: all contours size-ordered with explicit per-root zorder so the
            # no-BNT FILLS sit above the dashed BNT lines (matplotlib draws fills under lines).
            items = [(a, "noBNT", FOM[a][0]) for a in arms] + [(a, "BNT", FOM[a][1]) for a in arms]
            items.sort(key=lambda t: t[2])
            jb = [t for t in items if t[0] == "jointl1" and t[1] == "BNT"]
            items = [t for t in items if t not in jb] + jb
            roots, colors, lss, filled, lws, line_args, contour_args = [], [], [], [], [], [], []
            for z, (a, frame, _) in enumerate(items):
                if frame == "noBNT":
                    roots.append(mc(znb, a, " no BNT")); colors.append(ARMS[a]["color"])
                    lss.append("-"); filled.append(True); lws.append(LW)
                    line_args.append(dict(color=ARMS[a]["color"], lw=LW, ls="-", zorder=z))
                else:
                    roots.append(mc(zbn, a, " BNT")); colors.append(ARMS[a]["color"])
                    lss.append("--"); filled.append(False); lws.append(BNT_LW)
                    line_args.append(dict(color=ARMS[a]["color"], lw=BNT_LW, ls="--", zorder=z))
                contour_args.append(dict(zorder=z))
            tp = dict(filled=filled, colors=colors, contour_ls=lss, contour_lws=lws,
                      line_args=line_args, contour_args=contour_args, markers=markers)
            if getdist_legend:
                g.settings.legend_fontsize = leg_fs
                g.triangle_plot(roots, **tp)
            else:
                g.triangle_plot(roots, legend_labels=[], **tp)
                try:
                    if g.legend is not None:
                        g.legend.remove()
                except Exception:
                    pass

        if not getdist_legend:
            # colour-per-arm key (base colour, solid) + solid/dashed frame key, no title
            ch = [Line2D([0], [0], color=ARMS[a]["color"], ls="-", lw=2.0, label=ARMS[a]["label"]) for a in arms]
            leg = g.fig.legend(handles=ch, loc="center", bbox_to_anchor=color_anchor, frameon=False,
                               fontsize=leg_fs, handlelength=1.6, labelspacing=0.45)
            leg._legend_box.align = "left"
            g.fig.add_artist(leg)
            _style_legend(g, style_anchor, fontsize=leg_fs)
        if bar_rect is not None:
            _retention_bar(g, bar_arms, bar_rect)
        g.fig.savefig(str(out)); g.fig.savefig(str(HERE / "_preview" / (out.stem + ".png")), dpi=160)
        print("wrote", out)

    # (a) analytic: auto / +product / joint l1 -- the retention spread lives here (x0.17..x0.72)
    # smaller legend, nudged right
    panel(["l1_auto", "l1_product", "jointl1"], HERE / "fig_bnt_analytic.pdf",
          color_anchor=(0.85, 0.585), style_anchor=(0.85, 0.45), leg_fs=6.3, outer_ring=True,
          bar_rect=[0.605, 0.735, 0.365, 0.205], bar_arms=["l1_auto", "l1_product", "jointl1"])
    # (a') less-busy variant: identical, but WITHOUT the l1+product CONTOURS (the inset still keeps
    # all three bars, so the +product retention is still shown). Kept alongside the full version.
    panel(["l1_auto", "jointl1"], HERE / "fig_bnt_analytic_noprod.pdf",
          color_anchor=(0.85, 0.585), style_anchor=(0.85, 0.45), leg_fs=6.3, outer_ring=True,
          bar_rect=[0.605, 0.735, 0.365, 0.205], bar_arms=["l1_auto", "l1_product", "jointl1"])
    # (b) learned: CNN only, no inset (retention ~100%); getdist default legend
    # with simple "CNN BNT" / "CNN no BNT" labels. Bigger fonts (panel is sparse).
    panel(["cnn"], HERE / "fig_bnt_learned.pdf", getdist_legend=True,
          leg_fs=11, axes_fs=10, lab_fs=12)


if __name__ == "__main__":
    (HERE / "_preview").mkdir(parents=True, exist_ok=True)
    main()
