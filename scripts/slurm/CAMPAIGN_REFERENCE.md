# Campaign reference — architecture, verified constants, measured numbers

Companion to `README.md`. That file says which job to run; this one records **what the
jobs are configured to be**: the readout architecture, the constants that must not be
re-derived, the definition of the quoted ±, and the numbers the campaign measured. Read
it before changing any flag in a job script — most of the values in there are pinned
against an archived artefact of the original run, not chosen.

---

## 1. The architecture

Every arm's readout is the same:

```
VMIM(compressor seed)  ->  sbi_lens ConditionalRealNVP 4x128, POOLED over 3 flow seeds
```

* **ℓ1 rows** — raw ℓ1 datavector → **MLP compressor (hidden 256,256) trained with VMIM**
  → 10-D → RealNVP. Do **not** put a flow on the raw ℓ1 vector: the same RealNVP craters
  on the 2000-D vector (observed as all-NaN training). Compression is not optional.
* **CNN rows** — maps → **ResNet-18 VMIM** → 10-D → the same RealNVP. Reads the **4 auto
  maps only** (`--cross-op none`, `channel_slice=slice(0,4)`); no cross channels.
* **Single-quoted row** = one compressor, its flows pooled. Its bar is the spread across
  the three compressor seeds.
* **Ensemble-quoted row** = the *same* three compressors pooled (9 flows).
* So: **3 compressors per (operator, frame)**. Their spread is the singles band; their
  pooling is the ensemble.

---

## 2. What the error bar is

Authority: `docs/NOTE_FOM_ERROR_BARS.md` in this repository.

```
+/-   = mean +/- std over the 3 PRE-ENSEMBLE compressor singles, for ALL rows
        -- including rows whose central value is an ensemble.
        Also quote the min-max band. A 3-draw sigma carries ~50% relative
        uncertainty; present it as INDICATIVE reproducibility, not a
        statistical error.
bias  = the single->ensemble shift. NOT an error bar. Reported separately.
        The measured de-inflations 779->718 and 425->388 ARE the bias term,
        and they are why the ensemble is the quoted estimator in those rows.
median= block bootstrap over the 180 patches, keeping all 50 reps, 10^4
        replicates, 68% PERCENTILE interval, run once per arm at seed 41.
        Caption term only.
```

The leave-one-out ensemble jackknife is a diagnostic and must never be quoted as the bar.

**Which estimator each published row quotes** — the table mixes them deliberately: the
ensemble is quoted only where the single failed the calibration battery.

| arm | no-BNT | quoted as | BNT | quoted as |
|---|--:|---|--:|---|
| ℓ1 auto | **2448** (ens 2429) | SINGLE | 388 | ENSEMBLE |
| ℓ1 +product | **3045** (ens 3009) | SINGLE | 718 | ENSEMBLE |
| joint ℓ1 | 3371 | ENSEMBLE | 2424 | ENSEMBLE |
| CNN | 3326 | SINGLE | 3186 | SINGLE (already calibrated) |

**Never compare a retrained single against a published ensemble, or vice versa.** Across
every row the ensemble sits below the singles — pooling 9 flows widens the posterior:
auto no-BNT 2511 vs singles ~2777; +product no-BNT 3054 vs ~3232; +product BNT 758 vs
~912. A single running ~11 % above its ensemble is the expected pattern, not an anomaly.

---

## 3. Verified constants — do not re-derive

```
geometry   n_centers=180  center_nside=64  min_separation_deg=14.2  max_abs_lat=75
           -> inputs/centers_PAPER_180.npy  (confirmed 22/22 by bit-exact reprojection
              against the cross TFDS; NEVER regenerate -- see inputs/README.md)
field      10 deg / 80 px / reso 7.5'   nside 512   lmax 1024
noise      sigma_e 0.26   galaxy_density 10.0   noise_seed_base 12345
           seed formula: base + 100*cosmo_idx + perm   (verified: reprojection at 1e-8)
sigma tbl  n_real 48   n_scales 5   seed_base 987654321   op both (16 channels)
obs        9000 = 180 patches x perms 0..49   (eval uses perm<50; the cache holds 200)
TFDS       nbody_cosmogrid_dataset_tomo_cross/grid_10deg_80px_nonoverlap180
FoM3       1/sqrt(det(cov(samples[:, :3])))   param order Omega_m, sigma_8, w_0
           verified to 1.7e-16 against 8 archived posterior.fom.json records
preproc    compression stage: log1p-zscore / clip 5   / min-var 1e-5
           downstream sweep:  none         / clip 0   / min-var 1e-12
           CNN sweep:         zscore       / clip 5.0 / min-var 1e-12   (see README sec 3)
compressor summary_dim 10, hidden 256,256, nf_layers 4, nf_hidden 128, steps 30000,
           seeds 41/42/43 (44/45/46 for the six-seed check)
CNN        resnet18, dim 10, 80k steps, batch 128, lr 5e-4, checkpoint policy best_val
           measured ~70 steps/s => ~19 min train + ~2 min compress per seed
joint l1   stat=jointl1  k=10  adaptive_ranges=True  dequantize=False  snr_range=5.0
           6 pairs x 5 scales x 10x10 = 3000-D
```

The joint-ℓ1 statistic is the **2-D pairwise ℓ1-weighted histogram**: per bin pair (i,j)
and scale s, a K×K grid over the joint (u_i,u_j) plane whose cells hold `sum ½(|u_i|+|u_j|)`.
It is *not* `--cross-op both`, and it is *not* `pair2d_features(weighted=False)` — the
unweighted `pair2d` variant gives 4922/5156/4513 at N=1000, a different statistic.
`--rotated-binning` is also not the published lineage: that shear-aware experiment failed
its BNT gate (3221, SBC 0.338).

---

## 4. Measured numbers

Retrained rows against their published values, all at N=9000:

| Row | Published | Retrained (ensemble) | Δ | Median term | Singles band |
|---|--:|--:|--:|---|---|
| ℓ1 auto, no-BNT | 2429 (ens) / 2448 | 2511.1 | +3.4 % | ±4.14 (0.16 %) | ±30 (1.09 %) |
| ℓ1 auto, BNT | 388 (ens) | **390.7** | **+0.7 %** | ±1.00 (0.26 %) | ±46 (9.88 %) |
| ℓ1 +product, no-BNT | 3009 (ens) / 3045 (single) | 3053.9 | +1.5 % | ±5.45 (0.18 %) | ±185 (5.72 %) |
| ℓ1 +product, BNT | 718 | 758.3 | +5.6 % | ±2.36 (0.31 %) | ±37 (4.02 %) |
| joint ℓ1, BNT | 2424 (ens) | — | — | ±6.74 (0.28 %) | — |
| CNN, no-BNT | 3326 | 3503.5 | +5.3 % | ±8.20 (0.23 %) | ±34 (0.96 %) |
| CNN, BNT | 3186 | 3265.7 | +2.5 % | ±8.06 (0.25 %) | ±18 (0.53 %) |

All retrained rows land **+0.7 % to +5.6 %** of their published values, inside the
~10–15 % agreement the exercise was scoped to.

CNN singles are far tighter than ℓ1 singles (0.53–0.96 % vs 1.1–9.9 %): a ResNet-18 VMIM
is a more reproducible compressor than an MLP on ℓ1.

Independent checks that the published numbers are exactly what the archived per-observation
data says (all N=9000):

| archived run | median FoM3 | published |
|---|--:|--:|
| `auto_nobnt_ensemble` | 2428.6 | 2429 |
| `auto_bnt_ensemble` | 388.4 | 388 |
| `jointl1_bnt_ensemble` | 2424.3 | 2424 |
| `l1product_rnvp_s41_n9000` | 3044.9 | 3045 |

Validation targets for the joint-ℓ1 sweeps (`sw_joint_chain.slurm`):
no-BNT singles 3754 / 3761 / 4034, ensemble 3371 (published);
BNT singles 3232 / 3235 / 2708, ensemble 2424.3.
Validation target for ℓ1+conv no-BNT seed 41: 2671 (reproduced to +1.8 %, the closest
agreement of any ℓ1 arm — an independent check that the column-slicing is sound).

**BNT rows for `conv` and `conv+product` are new.** The published operator ladder is
no-BNT only, so those two centrals have no published counterpart; flag them as new in the
paper caption.

---

## 5. Data-product layout under `$NN_L1_SBI_DATA`

```
tmp/                                              TMPDIR (node /tmp is too small for torch sdists)
logs/                                             $NN_L1_SBI_LOGS
full_sphere_cache_fiducial_10deg/                 observation cache ROOT
  nobnt/obs/cosmo_fiducial_perm<N>.npz              (the loader appends regime/obs itself)
sigma_tables/flatsky_cross_noise_sigma.npz        no-BNT frozen sigma
                                                  (its BNT twin is $NN_L1_SBI_SIGMA_BNT,
                                                   which defaults to the _bnt.npz beside it)
l1_datavector_cache/flat_local_{both,product,none,conv}
l1_datavector_cache_bnt/flat_local_{both,product,none,conv}_bnt
fiducial_both_datavectors_{nobnt,bnt}.npz         16-ch observation datavectors
fid_raw_{both,product,none,conv}[_bnt].npz        sliced per operator (key "S")
joint/jointl1_{nobnt,bnt}_raw/{cache,fid.npz}     raw 3000-D joint l1
joint/jointl1_{nobnt,bnt}_s4{1,2,3}               its VMIM arms (+ _s4{4,5,6} extra seeds)
vmim/{none,product,conv,both}_{nobnt,bnt}_s4{1,2,3}
cnn/{nobnt,bnt}_s4{1,2,3}/cache/{cnn_obs.npz,cnn_cache_meta.npz}
cnn/{nobnt,bnt}_s4{1,2,3}/fiducial_summaries.npz
arms/l1{both,product}_{nobnt,bnt}_s41             full-driver arms (the gate)
sweeps/<label>/{per_patch_metrics.npz, median_summary.json}
wheelhouse/                                       offline install only, ~12 GB, disposable
```

`sweeps/joint_l1_nobnt_*`, `sweeps/joint_bnt_*` and `vmim/both_*` belong to the
**ℓ1+conv+product** arm, not to joint ℓ1 — see `README.md` §6.

---

## 6. Gotchas that cost real time

* **numpy tie-order.** `_build_non_overlapping_centers` uses `np.argsort` (quicksort,
  unstable) and every HEALPix ring is a block of exact ties, so numpy 1.24.4 and 1.26.4
  give tilings sharing only 2 of 180 centres. Always pass
  `--centers-npy inputs/centers_PAPER_180.npy`; never regenerate.
* **`wl_stats_torch` PyPI v0.1.0 is stale** — it lacks `subtract_coarse_mean` and the
  wavelet transform silently takes a different code path. Install from git at the pinned
  commit (`inputs/env_freeze.txt`).
* **`sbi_lens` needs a two-line patch**: `tfp.experimental.substrates.jax` no longer
  exists; use `from tensorflow_probability.substrates import jax as tfp`.
* **`tensorflow-metadata` must be 1.16.1 with protobuf 5.29.3** — newer versions ship
  protobuf gencode 6.31.1 and break tfds under TF 2.18.
* **`TF_FORCE_GPU_ALLOW_GROWTH=true`** wherever torch and tf.data share a GPU — otherwise
  TF takes the whole card and torch OOMs at under 1 GB. See `README.md` §5.
* **Environment inode pressure.** Conda environments are hundreds of thousands of small
  files; put them on a filesystem with inode headroom, and bulk caches on scratch.
* **Never load the 8–13 GB caches on a login node** — they get OOM-killed.
* **`qos_gpu_a100-dev` allows at most 10 submitted jobs per user** (2 h wall). That is why
  `sw_joint_chain.slurm` and `conv_arms/conv_arm.slurm` chain their sweeps into one job
  instead of fanning out. Do not "fix" this by padding the walltime onto a longer QoS —
  every job here finishes well inside 2 h (longest measured: 29 min).
