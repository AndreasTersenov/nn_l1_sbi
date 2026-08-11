# Patch-centre drivers — the geometry the whole analysis sits on

The paper's 180 non-overlapping 10° patch centres are **pinned** in
`scripts/slurm/inputs/centers_PAPER_180.npy`. Every cache, every datavector and every
observation in Table 1 is cut at those centres, so the file is load-bearing.

These five drivers are the investigation that recovered that geometry and then proved it,
and they are shipped for one reason: **so the pinned file can be verified rather than
trusted**. They are not a regeneration path.

---

## Why the centres must not be regenerated

`_build_non_overlapping_centers` (in `nn_l1_sbi.data.tf_dataset_nbody_tomo`) orders its
candidate pixels with `np.argsort`. That is quicksort — **unstable** — and every HEALPix
ring is a block of exact latitude ties, so the order inside a ring depends on the numpy
build. The tiling is therefore *not* a function of the four parameters alone.

Measured while porting these drivers, with the published parameters
(`n_centers=180, min_separation_deg=14.2, center_nside=64, max_abs_lat=75`) under
**numpy 1.26.4**:

```
centres reproduced in the same order as the pinned file :   0 / 180
centres present anywhere in the regenerated tiling      :   2 / 180
```

which is the same 2/180 that `scripts/slurm/inputs/README.md` records between numpy 1.24.4
and 1.26.4. A regenerated tiling is a *different sky selection*, and mixing it with the
published caches would silently corrupt every downstream number.

**Always pass `--centers-npy .../inputs/centers_PAPER_180.npy`.** None of the drivers here
ever writes into `scripts/slurm/inputs/`; their outputs go to a separate work directory.

---

## The five steps

Run by `scripts/slurm/setup/{find,fit,measure,prove,confirm}_centers.slurm`, which set
`NN_L1_SBI_CENTERS_SRC` to this directory by default.

| # | driver | what it does | needs |
|---|---|---|---|
| 1 | `find_centers.py` | bisects `min_separation_deg` per `(center_nside, max_abs_lat)` for the largest value that still yields 180 centres; writes one `centers_<geomkey>.npy` per distinct geometry plus `centers_search.json` | CPU only |
| 2 | `fit_centers.py` | fits `(min_separation_deg, max_abs_lat)` to the 22 **measured** centres, requiring all 22 to 1e-4 deg | `measured_centers.json` |
| 3 | `measure_centers.py` | measures the true `centers[k]` for as many patch indices as one TFDS shard exposes, by matching each stored patch to the HEALPix pixel centre that reproduces it (8×8 pre-filter, then bit-exact 80×80) | cross TFDS + CosmoGridV1 |
| 4 | `prove_centers.py` | ranks all candidate geometries by bit-exact reprojection on a few probe patches; writes `centers_PAPER_180.{npy,json}` for the winner | cross TFDS + CosmoGridV1 |
| 5 | `confirm_centers.py` | the actual verification: takes the richest `(cosmo, perm)` group in one shard and requires **every** patch in it to reproject bit-exactly (`max|Δ| < 1e-5`). This is what "confirmed 22/22" means | cross TFDS + CosmoGridV1 |

The chain is deterministic only within one numpy build; steps 3–5 are the ones that carry
evidential weight, because they compare against the TFDS the paper actually trained on.

### To verify the shipped centres

Run **step 5 alone**:

```bash
cd scripts/slurm
sbatch setup/confirm_centers.slurm
```

`confirm_centers.py` looks for `centers_PAPER_180.npy/.json` in the work directory first
and falls back to the pinned copies in `scripts/slurm/inputs/`, so with an empty work
directory it verifies exactly the shipped file. The pass criterion is `mismatches>1e-5 : 0`
over every patch of the chosen group. The runner-up "control" geometry it also prints is a
contrast, not part of the criterion, and is skipped when no `find_centers.py` run is
present in the work directory.

---

## Configuration

| variable | meaning | default |
|---|---|---|
| `NN_L1_SBI_CENTERS_SRC` | where the SLURM jobs look for these drivers | `<repo>/scripts/centers` |
| `NN_L1_SBI_CENTERS_OUT` | work directory for candidates, measurements and the confirmation JSON | `$NN_L1_SBI_DATA/centers` |
| `NN_L1_SBI_TFDS` | cross TFDS root (`nbody_cosmogrid_dataset_tomo_cross/grid_10deg_80px_nonoverlap180/0.0.1` is appended) | `$TFDS_DATA_DIR`, else `~/tensorflow_datasets` |
| `NN_L1_SBI_COSMOGRID` | CosmoGridV1 root; `CosmoGridV1_metainfo.h5` and the raw `projected_probes_maps_nobaryons512.h5` are read through it | none — steps 3–5 stop if unset |
| `CG_PARENT` | overrides the directory that *contains* the CosmoGridV1 tree (`path_par` in the metainfo is relative to it) | parent of `$NN_L1_SBI_COSMOGRID` |
| `N_GROUPS` | how many `(cosmo, perm)` groups `measure_centers.py` walks | 12 |

`PYTHONPATH` must contain `<repo>/src`; `scripts/slurm/env.sh` does that.

Steps 1 and 2 deliberately **AST-extract** `_angular_distance_deg` and
`_build_non_overlapping_centers` out of `nn_l1_sbi/data/tf_dataset_nbody_tomo.py` and
`exec` them, instead of importing the module: the extraction runs the shipped
implementation verbatim while keeping TensorFlow, TFDS and JAX out of a CPU-only job.

---

## What changed relative to the archived originals

These are the campaign's one-off investigative tools, ported — not rewritten. The numerics
(tolerances, the noise seed formula `12345 + 100·cosmo_idx + perm`, the projection recipe,
the pass criteria) are untouched. The changes are:

* absolute paths to one machine replaced by `nn_l1_sbi.paths` (`tfds_dir`,
  `cosmogrid_root`, `cosmogrid_metainfo`, `data_root`) and by the work directory above;
* the AST source is located through `importlib.util.find_spec` on the installed package
  instead of a rescued copy of the module;
* `confirm_centers.py` treats the control geometry and the `geometry` key as optional, and
  always writes its updated JSON to the work directory — so verifying the shipped centres
  neither requires a full search nor can modify a pinned input;
* `prove_centers.py` treats `centers_search.json` as optional (it only annotates the
  output with the generating parameters) and stops with a clear message when the work
  directory holds no candidate geometry.
