"""Neural posterior estimation, population sweeps, and the calibration gates.

Modules (import explicitly; most pull in JAX, and the l1 driver also torch, TensorFlow and
``wl_stats_torch``, at module level — note that importing
``npe_l1norm_cross_jaxili_nbody_tomo`` sets the process thread-budget environment variables
as a side effect):

* ``npe_l1norm_cross_jaxili_nbody_tomo`` — the l1 arm end-to-end (wavelet-l1 auto and
  flat-sky cross datavectors, preprocessing, jaxili NPE); also the library of record for
  ``preprocess_summaries`` and ``filter_zero_variance_bins``.
* ``population_sweep`` — CLI driver for the sweep over the 9000 fiducial observations:
  per-arm NDE seeds pooled per observation, per-patch sigmas, FoM3 and posterior moments.
* ``ensemble_eval`` — CLI driver for the compressor-ensemble variant of that sweep, pooling
  the three compressor-seed arms per physical observation.
* ``tarp_stratified_val`` — CLI driver for varied-theta TARP-DRP on the held-out validation
  ensemble, stratified into FoM3 terciles, dumping posterior samples per (tercile, seed).
* ``run_tarp_coverage`` — CLI driver that turns those dumps into TARP joint-coverage curves
  and the per-arm and overlay figures.
* ``run_joint_gate_c`` — GATE C orchestrator for the joint-statistic arms: cache preflight,
  then the two TARP drivers above as subprocesses, then the coverage verdict report.
"""
