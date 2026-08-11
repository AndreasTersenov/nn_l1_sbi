"""nn_l1_sbi — wavelet-l1 versus neural compression for tomographic weak-lensing SBI.

Reproduction code for *"Do we need deep learning for tomographic weak lensing?
Wavelets versus neural compression"* (Tersenov, Starck & Kilbinger).

The package implements both arms of the matched comparison, from CosmoGridV1
light-cones to a calibrated figure of merit: patch-local flat-sky cross-map
operators and starlet-l1 datavectors (``flatsky``), simulation ingestion and
cache building (``data``), VMIM compression to a 10-D summary and the NDE-family
sweep (``compression``), the ResNet-18 CNN arm (``cnn``), the NPE drivers,
population sweeps and calibration gates (``inference``), and the paper figure
generators (``figures``).

The stage-by-stage method description — settings, gates and provenance — is in
``docs/PIPELINE.md``.

Importing this package is deliberately cheap: no submodule is imported here, so
``import nn_l1_sbi`` succeeds without TensorFlow, torch, JAX or healpy present.
The submodules pull those in at module level, so import the one you need
explicitly, e.g. ``from nn_l1_sbi.flatsky import flatsky_cross``.
"""

__version__ = "1.0.0"
__author__ = "Andreas Tersenov"
