"""Table-1 error bars: the code that produced the committed ``data/table1`` bundle.

``docs/NOTE_FOM_ERROR_BARS.md`` is the authority on *what* the Table-1 ``±`` means.
This subpackage is the executable half of that authority: the three error terms it
defines (seed spread, block-bootstrap median SE, leave-one-out ensemble jackknife)
are computed here, and nowhere else in the package.

Modules (import explicitly; each is a small CLI driver over ``numpy`` only — no JAX,
no torch, no TensorFlow):

* ``final_bars`` — the term library and a CLI. Seed term (spread of the per-compressor
  medians), median term (block bootstrap over the 180 patches keeping each patch's 50
  noise reps intact), ensemble term (leave-one-out jackknife over the K=3 compressor
  members via exact two-component mixture moments), plus ``population_stats`` (CV and
  the intra-patch correlation ρ by one-way ANOVA). Produced the committed
  ``data/table1/final_bars_all_v2.json``.
* ``make_table`` — the row driver that emits the per-arm error bars in the presentation
  ``docs/NOTE_FOM_ERROR_BARS.md`` §6 specifies, keeping the three quantities apart:
  the central, the ``±`` over the three *pre-ensemble* singles, and the single→ensemble
  shift as a **bias**. Produced the committed ``data/table1/TABLE1_ERRORBARS_V2.json``.
* ``derive_arms`` — the path-relativisation that turns the shipped provenance record
  ``data/table1/rows_table1_v2.json`` (absolute paths to the 153 MB
  ``per_patch_metrics.npz`` files) into the shipped ``data/table1/arms.json``
  (repo-relative paths to the tiny ``median_summary.json`` digests). Byte-exact.

Both ``final_bars`` and ``make_table`` read ``per_patch_metrics.npz``, which is Tier B
of the Zenodo record and is *not* in the repository — they are here so the recipe is
auditable and re-runnable, not because Table 1 needs them: every number in Table 1 is
already reproducible from the committed digests alone (``data/table1/README.md`` §6).

Importing this package is cheap; no submodule is imported here.
"""
