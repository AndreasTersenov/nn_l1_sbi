"""VMIM compression to the shared 10-D summary, and the NDE-family sweep on top of it.

Modules (import explicitly; all pull in JAX at module level):

* ``vmim_from_cache`` — CLI driver that VMIM-compresses a ``(theta, x)`` cache to a
  low-dimensional summary via an MLP compressor plus a normalising-flow companion.
* ``vmim_maf_companion`` — hand-rolled conditional MAF (Haiku), identity-initialised, a
  drop-in companion for VMIM compressor training.
* ``train_nde_from_compressed`` — CLI driver for the NDE-family sweep
  (sbilens_realnvp / jaxili_maf / jaxili_realnvp / jaxili_mdn) on frozen summaries, holding
  the compressor fixed so any figure-of-merit delta is attributable to the density estimator.
"""
