"""Simulation ingestion: CosmoGridV1 light-cones to TFDS datasets and cached datavectors.

Modules (import explicitly; several pull in TensorFlow, healpy or torch at module level):

* ``tf_dataset_nbody_tomo`` — TFDS builder for the 4-bin tomographic CosmoGridV1
  convergence maps; importing it registers the dataset.
* ``tf_dataset_nbody_tomo_cross`` — TFDS builder for the 10-channel auto+cross harmonic
  maps, reserialized from the validated ``.npz`` cross cache.
* ``tfds_cross_tfdata_loader`` — ``tfds.load`` + ``tf.data`` loader for the cross dataset,
  yielding ``{"maps", "theta"}`` batches with example-disjoint permutation splits.
* ``build_full_sphere_cross_cache`` — CLI driver that builds the full-sphere harmonic-space
  cross-maps and cuts them into gnomonic patch tensors.
* ``build_flatsky_joint_arm`` — CLI driver that builds one joint-statistic arm
  (cov / pair2d / full4d / jointl1) in either basis.
* ``precompute_fiducial_both_datavectors`` — CLI driver that precomputes the 16-channel
  ``both`` flat-local l1 datavector for the fiducial observations, for column slicing per arm.
"""
