"""Patch-local (flat-sky) channel construction and wavelet-l1 statistics.

Modules (import explicitly; several pull in torch or TensorFlow at module level):

* ``bnt_utils`` — the ``tomo4_bnt_v1`` 4x4 BNT nulling matrix and its numpy / TF appliers.
* ``flatsky_cross`` — the patch-local cross-map operators (convolution, product) in
  numpy / torch / jax, bit-identical across backends, plus the BNT / whiten / deep
  channel-mix matrices.
* ``flatsky_cross_l1`` — starlet-l1 datavectors on the channel stack, with the frozen
  per-(channel, scale) noise sigma as the SNR denominator.
* ``flatsky_joint_stats`` — joint one-point statistics (cov / pair2d / full4d / jointl1)
  over per-pixel wavelet-coefficient vectors, per scale.
* ``freeze_flatsky_cross_noise`` — CLI driver that measures and freezes the noise sigma
  table used as that SNR denominator.
"""
