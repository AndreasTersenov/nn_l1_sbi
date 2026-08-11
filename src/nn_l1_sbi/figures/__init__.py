"""Paper figure generators: one script per figure, each reading committed dumps.

Every generator emits a vector PDF in the shared house style. Modules marked *(retained)*
have a working generator whose figure is not cited in the current manuscript.

Helpers:

* ``style`` — shared A&A style: column widths, the fixed per-arm palette, ``use_style()``,
  ``getdist_settings()``.
* ``_data`` — single place where figure data provenance lives: the per-patch and gate-dump
  path tables and their loaders.
* ``dump_contour_samples`` — one-time GPU dump of the mean-observation posteriors to
  ``contour_{nobnt,bnt}_samples.npz``, so the contour figures stay pure-CPU replots.
* ``gen_auto_contour.sh`` — shell driver that reproduces the l1 auto-only arm and merges it
  into the no-BNT contour dump.

Figures:

* ``fig_data`` — the tomographic auto-kappa and cross-channel patch stack.
* ``fig_l1`` — starlet-l1 histograms versus SNR coloured by sigma8, and the joint-l1 cell map.
* ``fig_pipeline`` — the estimator-controlled design schematic.
* ``fig_corner`` — mean-observation posterior triangle for the four arms, with the FoM3 inset.
* ``fig_cp`` — per-patch distributions of the marginal sigmas and FoM3 over 9000 observations.
* ``fig_ablation`` — the operator completeness ladder in the matched RealNVP frame.
* ``fig_tarp`` — TARP-DRP expected coverage with the sightline-bootstrap band, both bases.
* ``fig_sbc`` — SBC rank histograms and ECDF differences against their uniformity bands.
* ``fig_bnt`` — analytic and learned arms under BNT nulling, with FoM3 retention insets.
* ``fig_fisher`` — per-bin diagonal Fisher of the 800-D auto l1 as a (scale x SNR) heatmap.
* ``fig_leak`` — super-patch variance fraction per multipole, auto versus cross *(retained)*.
* ``fig_leak_maps`` — full-sphere mollweide views of the auto and cross channels *(retained)*.
* ``fig_bntwhiten`` — no-BNT versus whitened versus BNT FoM3, plus the deep-direction ladder
  *(retained)*.
* ``fig_w0`` — the fiducial w0 pull versus patch size *(retained)*.
* ``fig_embed`` — t-SNE of the 10-D summaries for the three compressors; illustrative, not
  evidential *(retained)*.
"""
