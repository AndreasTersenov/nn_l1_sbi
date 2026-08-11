# SLURM jobs — the Table-1 campaign

These are the batch jobs that produce every row of Table 1: the FoM3 central values and
the reproducibility error bars for six statistics in two bases.

The campaign is one three-stage pipeline, repeated per **(operator arm × frame)**:

```
raw datavector  ->  3 VMIM compressor seeds (41/42/43)  ->  population sweep
                    (10-D summary)                          (3 pooled RealNVP flow
                                                             seeds, 9000 mock obs)
```

A **single** row = one compressor, its three flows pooled. An **ensemble** row = the same
three compressors pooled (9 flows). The quoted ± is the spread across the three
compressor seeds. `docs/NOTE_FOM_ERROR_BARS.md` in this repo is the authority on which
estimator each row quotes; `CAMPAIGN_REFERENCE.md` next to this file records the verified
constants and the measured per-row numbers.

Six arms × two frames (`nobnt`, `bnt`) = the 12 rows.

---

## 1. Quick start

```bash
# 1. tell the jobs where the code, the data and the site inputs live
export NN_L1_SBI_JOBS=$PWD                      # this directory (scripts/slurm)
export NN_L1_SBI_DATA=/path/to/writable/root    # caches, arms, sweeps  (default: $SCRATCH/nn_l1_sbi)
export NN_L1_SBI_TFDS=/path/to/tensorflow_datasets
export NN_L1_SBI_COSMOGRID=/path/to/CosmoGridV1
export NN_L1_SBI_SIGMA_BNT=/path/to/flatsky_cross_noise_sigma_bnt.npz
export NN_L1_SBI_PYTHON=/path/to/env/bin/python

# 2. the #SBATCH --output lines are relative to the directory you submit from
mkdir -p logs

# 3. submit
sbatch setup/build_obs_cache.slurm
```

`NN_L1_SBI_SRC` is derived from this directory's location and normally needs no setting.
Anything you export before `sbatch` is inherited by the job.

For persistent settings, write them into a **site file** instead:

```bash
cp site.jean-zay.sh.example site.local.sh   # then edit the paths
```

`env.sh` sources `site.local.sh` (repo root or here) before it applies any default, so
everything it sets wins; the older name `local_paths.sh` is still read too, and both are
gitignored. `site.jean-zay.sh.example` is the site file the published numbers were produced
with — the IDRIS roots, module loads and proxy — kept as a worked example and a provenance
record, not as a default: every default in `env.sh` is derived from `$SCRATCH` / `$WORK` /
`$HOME` and names no machine. Off Jean-Zay, take the example as a template.

**Every job starts by sourcing `env.sh`.** Read its header before running anywhere other
than Jean-Zay: it documents each site-specific `#SBATCH` directive (account, partition,
QoS, constraint, walltime) and the module loads, and says what to change. Those directives
are deliberately left in the job files rather than abstracted away — they are the actual
recipe the published numbers were produced under.

Requirements that are not optional:

* the pinned python environment — `inputs/env_freeze.txt` and `inputs/constraints.txt`
  (`setup/build_env.slurm` builds it online, `setup/install_env.slurm` offline);
* the cross TFDS `nbody_cosmogrid_dataset_tomo_cross/grid_10deg_80px_nonoverlap180`
  (10°/80 px, ~290 GB) — this is the training set and is *not* rebuilt by these jobs;
* a read-only copy of CosmoGridV1 for the fiducial observation cache.

---

## 2. Submit order

```
setup/            environment, observation cache, noise sigma, raw-cache slicing
  |
  +-- raw datavectors      train_l1both_build.slurm, train_l1both_bnt.slurm,
  |                        build_joint_arm.slurm, cnn_*.slurm
        |
        +-- compressors    vmim_*.slurm, cnn_fidsum.slurm
              |
              +-- sweeps   sw_*.slurm, conv_arms/conv_arm.slurm
```

One exception to that order: `vmim_both.slurm` writes `fid_raw_both.npz`, which
`setup/slice_raw.slurm` consumes — see §6.

`setup/` in dependency order:

| job | what it does | cost |
|---|---|---|
| `build_env.slurm` / `install_env.slurm` | build the pinned python env, online / offline | ~1 h |
| `build_obs_cache.slurm` | the 10° fiducial observation cache: 200 perms × 180 patches, `--regime nobnt`, centres **pinned** from `inputs/centers_PAPER_180.npy`. Everything downstream reads it. | 10 h, 40 CPU |
| `freeze_sigma.slurm` | the no-BNT frozen per-(channel,scale) noise sigma table — the SNR denominator of the flat-local ℓ1 binning. Cross-checked field by field against the BNT twin. | ~2 h |
| `precompute_obs.slurm` | 16-channel no-BNT observation datavectors in one pass over the fiducial cache (not over the TFDS) | ~20 min |
| `bnt_obs_and_slice.slurm` | the same for BNT, then slices the BNT raw cache + obs into `product` (10 ch) and `none` (4 ch) | ~30 min |
| `slice_raw.slurm` | slices the no-BNT 16-ch `both` cache into `product` and `none` | CPU, ~5 min |
| `{find,fit,measure,prove,confirm}_centers.slurm` | the five-step investigation that reconstructed and then confirmed (22/22, by bit-exact reprojection) the 180 patch centres. **Its result is pinned in `inputs/centers_PAPER_180.npy` and must never be regenerated** — see `inputs/README.md`. The five python drivers ship in **`../centers/`** (`NN_L1_SBI_CENTERS_SRC` defaults there); they are provided so the pinned file can be *verified* — `setup/confirm_centers.slurm` alone does that — not so it can be rebuilt. See `../centers/README.md`. | verify: ~2 h CPU |

---

## 3. Which job produces which Table-1 row

| Table-1 arm | dim | raw datavector | compressor | sweep |
|---|---|---|---|---|
| ℓ1 auto | 4 ch / 800-D | `train_l1both_build` → `setup/slice_raw` (no-BNT) · `setup/bnt_obs_and_slice` (BNT) | `vmim_none_nobnt` · `vmim_none_bnt` | `sw_auto_{nobnt,bnt}_{s41,s42,s43,ensemble}` |
| ℓ1 + conv | 10 ch / 2000-D | `conv_arms/slice_conv` (both frames, one CPU job) | `conv_arms/conv_arm <frame>` | same job (chained) |
| ℓ1 + product | 10 ch / 2000-D | `setup/slice_raw` · `setup/bnt_obs_and_slice` | `vmim_product_nobnt` · `vmim_product_bnt` | `sw_product_{nobnt,bnt}_{s41,s42,s43,ensemble}` |
| ℓ1 + conv + product | 16 ch / 3200-D | `train_l1both_build` (no-BNT) · `train_l1both_bnt` (BNT) — both `--cross-op both` | `vmim_both` + `vmim_both_s4243` · `vmim_both_bnt` | `sw_both_{nobnt,bnt}_{s41,s42,s43,ensemble}` |
| joint ℓ1 | 3000-D | `build_joint_arm <frame>` | `vmim_joint <frame>` | `sw_joint_chain <frame>` (3 singles + ensemble, chained) |
| CNN (ResNet-18) | 10-D | — reads the 4 auto maps directly | `cnn_nobnt` · `cnn_bnt`, then `cnn_fidsum` | `sw_cnn <label> zscore 5.0 <arm…>` |

Notes that change the answer if you get them wrong:

* **`sw_cnn.slurm` takes `zscore 5.0`, not `none 0`.** The CNN driver standardizes its
  compressed summaries by default (`--standardize-summary`, `--summary-clip-value 5.0`)
  and that is what produced the published 3326; the ℓ1 driver has no such flag, so ℓ1
  arms feed the flow raw. Measured sensitivity: 3539 vs 3428 on `cnn_nobnt_s41` (3.2 %).
* The terse `sw_*.slurm` files omit `--preproc-transform/--clip-value/--min-feature-variance`
  because `population_sweep.py`'s defaults are exactly `none / 0.0 / 1e-12` — identical to
  what `sw_joint_chain.slurm` and `conv_arm.slurm` pass explicitly. They are not
  under-specified.
* Never put a flow on a raw ℓ1 vector; the compression stage is not optional. The RealNVP
  craters on the 2000-D vector (observed as all-NaN training).
* Patch centres always come from `inputs/centers_PAPER_180.npy` via `--centers-npy`.

Extra jobs, not Table-1 rows:

| job | purpose |
|---|---|
| `train_l1product_s41.slurm` | **the campaign gate** — retrains ℓ1+product no-BNT s41 and compares the population median FoM3 against the published 3045. If it lands within ~10–15 %, the whole train → eval → FoM → bootstrap chain is validated. |
| `extra_seeds.slurm <jointl1_bnt\|auto_bnt>` | compressor seeds 44/45/46 for the two double-digit-spread rows. n=3 → n=6 drops the relative uncertainty on σ from ~46 % to ~30 %. |
| `conv_arms/nde_variance.slurm`, `conv_arms/nde_variance_rows.slurm` | ± convention diagnostic: re-sweeps a fixed compressor with a different pooled flow-seed set (51/52/53). Finding: for the CNN the flow-set term is 2–3× the compressor term. See `conv_arms/README.md`. |

---

## 4. Reproducibility gates built into the jobs — keep them

Each of these stops the pipeline rather than producing a plausible wrong number.

1. `build_joint_arm.slurm` — feature dim `6 pairs × 5 scales × 10×10 = 3000`, finite,
   non-negative (the cells are sums of `|u|`).
2. `setup/build_obs_cache.slurm` — patch shape `(180,80,80,10)`, finiteness, polar leak
   `max|lat| < 75`, auto/cross channel scales, **and** the cache's own centres against
   `inputs/centers_PAPER_180.npy` to `< 1e-4`.
3. `setup/freeze_sigma.slurm` — nine metadata fields against the BNT twin (BNT only
   mixes channels, so they must agree exactly), plus `bnt == False` on the new table.
4. `conv_arms/slice_conv.slurm` — the sliced auto block must be byte-identical to the
   `none` arm, in both frames; otherwise the channel-ordering assumption is wrong.
5. `cnn_fidsum.slurm` — per-arm G1 gate `max|Δ| ≤ 9e-4` against the driver's own persisted
   `cache/cnn_obs.npz`, plus a SHA-256 re-hash of the compressor checkpoint so an arm can
   never be evaluated with another seed's weights.
6. `train_l1product_s41.slurm` — the campaign gate against the published 3045.

---

## 5. Environment notes that cost real time

* `build_flatsky_joint_arm.py` computes in torch but loads through `tf.data` and never
  calls `tf.config.set_memory_growth`. Without **`TF_FORCE_GPU_ALLOW_GROWTH=true`**
  TensorFlow takes the whole A100 and torch OOMs with under 1 GB of its own allocations.
  Set in `build_joint_arm`, `vmim_joint`, `sw_joint_chain`, `conv_arm`, `extra_seeds` and
  `nde_variance*`.
* `--fiducial-obs-cache-dir` (ℓ1 driver) wants the cache **ROOT**; the loader appends
  `<regime>/obs` itself. The CNN driver's equivalent flag is `--fiducial-obs-cache`
  (no `-dir`).
* The ℓ1 driver's `--fiducial-summaries-out` is not flat-local-aware — it bins with scalar
  SNR ranges while the observations use per-channel ones, and its own G1 gate rejects it.
  Use `setup/precompute_obs.slurm` instead.
* `population_sweep.py` auto-detects the cache prefix (`l1_*` from `vmim_from_cache.py`,
  `cnn_*` from the CNN driver); pass the arm directory either way.
* `sw_cnn.slurm` writes `logs/sw_cnn_<jobid>.out` whatever `--job-name` says — grep by job
  id, not by name.
* `harmonic_regime` is `nobnt` **even for the BNT arms**: the flat-local route reads raw
  autos from the no-BNT cache and applies BNT on-device (noise → demean → BNT →
  cross-build → whiten). CNN and ℓ1 rows read the same observation cache.
* Three package pins are load-bearing; see `inputs/README.md`.

---

## 6. `build_fiducial_summaries_cnn_flatlocal.py` and the ℓ1+conv+product arm

**`build_fiducial_summaries_cnn_flatlocal.py`** (next to this README) is a flat-local-only
variant of `nn_l1_sbi.cnn.build_fiducial_summaries_cnn`, kept because it is what actually
produced the CNN summaries behind the published numbers. Differences: it reads the frozen
per-channel RMS from `cache/cnn_cache_meta.npz['info_channel_scale']` instead of
recomputing it, requires `max|Δ| ≤ 9e-4` rather than `allclose(rtol=1e-3, atol=1e-3)`, and
verifies the compressor checkpoint SHA-256 by default. Cross-checked against the packaged
version: functionally equivalent on the flat-local route, slightly stricter.

**ℓ1+conv+product** is the `--cross-op both` arm — 16 channels = 4 autos + 6 conv +
6 product. Its full chain, both frames:

| stage | no-BNT | BNT |
|---|---|---|
| raw 16-ch datavector | `train_l1both_build.slurm` | `train_l1both_bnt.slurm` (same driver + `--flatsky-channel-mix bnt`, `$NN_L1_SBI_SIGMA_BNT`, `--cache-dir …/l1_datavector_cache_bnt`) |
| observation datavectors | `setup/precompute_obs.slurm` → `fiducial_both_datavectors_nobnt.npz` | `setup/bnt_obs_and_slice.slurm` → `fid_raw_both_bnt.npz` |
| compressors (3 seeds) | `vmim_both.slurm` (s41) then `vmim_both_s4243.slurm` (s42, s43) | `vmim_both_bnt.slurm` (s41–43 in one job) |
| sweeps | `sw_both_nobnt_{s41,s42,s43,ensemble}.slurm` | `sw_both_bnt_{s41,s42,s43,ensemble}.slurm` |

Two things about that table are not cosmetic:

* **`vmim_both.slurm` also writes `fid_raw_both.npz`.** `precompute_obs.slurm` stores the
  no-BNT observation datavectors under key `x`; `vmim_from_cache` and
  `setup/slice_raw.slurm` want key `S`. The one-cell adapter at the top of
  `vmim_both.slurm` is the only producer of `fid_raw_both.npz` in this tree, so in the
  no-BNT frame it must run **before** `setup/slice_raw.slurm`. The BNT frame has no such
  ordering constraint: `bnt_obs_and_slice.slurm` writes `fid_raw_both_bnt.npz` itself.
* The no-BNT compressors are split across two jobs because that is how they ran (s41 with
  a 100-minute budget, s42/43 afterwards at 45). The flags are otherwise identical.

The sweeps are the ordinary compressed-arm ones — `population_sweep` on
`vmim/both_<frame>_s4x` with the default `none / 0 / 1e-12` — and their `median_summary.json`
files are what `data/table1/arms.json` reads for this row. No-BNT is quoted as the published
**single, 3255** (the three compressor singles measured here are 3564.9 / 3154.9 / 3414.8,
whose spread is the ±200); BNT is quoted as the **ensemble, 703.9**, and has no published
counterpart (singles 761.7 / 879.8 / 715.1 → ±76).

`sweep_both.slurm` in this directory is **superseded and does not run**: it is the first
sweep attempted on this arm, against the full-driver arm directory, and its
`--fiducial-datavectors` / `--op` flags belong to `population_sweep_flatsky.py`, which was
destroyed in the storage failure and is not part of this release. It is kept as a record of
that attempt and stops with a message pointing at the `sw_both_*` jobs.

This arm's sweep output directories are named `sweeps/joint_l1_nobnt_*`, `sweeps/joint_bnt_*`
and `vmim/both_*`: those labels date from an early mislabelling of this arm as "joint ℓ1".
The statistic was always correct — `--cross-op both` really is autos + conv + product —
only the name was wrong. The genuine joint-ℓ1 arm is the 2-D pairwise ℓ1-weighted
histogram in `flatsky_joint_stats.py` (`--stat jointl1`), built by `build_joint_arm.slurm`
and written to `joint/jointl1_*`. Do not confuse the two directory families.

The `--arm-label` and `--out` names are therefore left exactly as the campaign wrote them —
renaming them would orphan `data/table1/arms.json`, which indexes this row by those
directory names — while the **job files** were renamed `sw_both_*` so the arm is legible
from the file listing. The mapping to the archived job names, for anyone tracing a log:

| job here | archived name | `--arm-label` / `sweeps/` directory |
|---|---|---|
| `sw_both_nobnt_{s41,s42,s43}` | `sweep_joint_{s41,s42,s43}` | `joint_l1_nobnt_{s41,s42,s43}` |
| `sw_both_nobnt_ensemble` | `sweep_joint_ens` | `joint_l1_nobnt_ensemble` |
| `sw_both_bnt_{s41,s42,s43,ensemble}` | `sw_joint_bnt_{s41,s42,s43,ensemble}` | `joint_bnt_{s41,s42,s43,ensemble}` |
