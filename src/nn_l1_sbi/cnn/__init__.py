"""The learned arm: ResNet-18 VMIM compressor, its NPE flow, and the shared metric helpers.

Modules (import explicitly; all pull in JAX, and the CNN driver also TensorFlow, at module
level — note that importing ``npe_cnn_nbody_tomo`` sets the process thread-budget
environment variables as a side effect):

* ``npe_cnn_nbody_tomo`` — the CNN arm end-to-end (compressor training, compression of
  obs/train/test, conditional RealNVP NPE, sampling and corner plots); also the library of
  record for ``build_flow`` / ``build_spline_flow`` / ``train_flow`` / ``setup_environment``
  and the compressor helpers reused elsewhere.
* ``train_jaxili_from_compressed`` — metric library: ``setup_env``, ``compute_fom3``,
  ``marginal_stats``, ``fom2d``, ``posterior_moments``. Running it as a script
  (``python -m nn_l1_sbi.cnn.train_jaxili_from_compressed``) runs the FoM3 self-check.
* ``build_fiducial_summaries_cnn`` — CLI driver that compresses the fiducial observations
  through a trained CNN arm, under the G1 consistency gate.
"""
