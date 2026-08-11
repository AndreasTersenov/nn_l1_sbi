# PIPELINE — from CosmoGrid light-cones to a calibrated figure of merit

End-to-end description of the analysis in *"Do we need deep learning for tomographic weak
lensing? Wavelets versus neural compression"* (Tersenov, Starck & Kilbinger), mapped stage
by stage onto the code in this repository.

Each section states **what the stage computes and why the choice is load-bearing**, then
**which module runs it** and **the exact command the paper's numbers were produced with**.
Every command below is transcribed from a job file in `scripts/slurm/` — it is the recipe,
not a reconstruction. Where a flag is omitted the driver's default applies and the omission
is deliberate; the two places where that matters are flagged in the text.

The scientific question the pipeline is built to answer: **does a learned convolutional
compressor extract more cosmological information from tomographic convergence maps than a
fixed, hand-crafted multiscale statistic?** Answering it fairly requires that the two
summaries differ *only* as summaries — same simulations, same noise, same patches, same
compression dimension, same density estimator, same calibration gate. That constraint is
what fixes almost every design decision below.

**Companion documents.** `scripts/slurm/README.md` — which job to submit, in what order.
`scripts/slurm/CAMPAIGN_REFERENCE.md` — the verified constants and the measured per-row numbers.
`data/table1/README.md` — the shipped Table-1 digest and the exact table recipe.
`docs/NOTE_FOM_ERROR_BARS.md` — the authority on what the quoted ± is.

---

## 0. The pipeline at a glance

```
CosmoGridV1 light-cones (PkdGrav3)
   └─ 4 tomographic kappa shells per realisation, HEALPix nside=512
        └─ + Gaussian shape noise on the sphere (per bin, independent)
             └─ [optional] BNT nulling, applied in a_lm space
                  └─ SHT -> real space -> 180 non-overlapping gnomonic patches
                       (10 deg x 10 deg, 80 x 80 px, 7.5 arcmin/px, |b| < 75 deg)
                       └─ per-patch, per-channel mean subtraction (mass-sheet)
                            └─ TFDS dataset (train / test splits by cosmology index)
                                 |
        ┌────────────────────────┴──────────────────────────┐
        │  (a) ANALYTICAL ARM                                │  (b) LEARNED ARM
        │  build channels patch-locally:                     │  4 auto maps only
        │    auto | +conv | +product | +conv+product         │       │
        │    or joint-l1 (pairwise, autos only)              │       │
        │        │                                           │       │
        │  starlet transform, 5 scales                       │  ResNet-18 CNN
        │  SNR bins under FROZEN per-(channel,scale) sigma   │  trained with VMIM
        │  l1 data vector (800 - 3200 dims) or joint grid    │       │
        │        │                                           │       │
        │  VMIM MLP compression -> 10-D                      │  -> 10-D summary
        └────────────────────────┬──────────────────────────┘
                                 │
                 SAME conditional RealNVP (4 coupling blocks x 128)
                 3 NDE seeds, pooled per observation
                 [+ 3 compressor seeds pooled = "deep ensemble" where needed]
                                 │
              posterior at each of 9000 mock observations (180 patches x 50 realisations)
                                 │
        per-obs sigma(Om), sigma(s8), sigma(w0), FoM3 = 1/sqrt(det Cov3)
                                 │
             median over observations  ->  headline number
                                 │
        gated by TARP expected coverage + SBC rank statistics (N = 600)
```

Two rules govern the whole design:

1. **Matched comparison.** Every summary is compressed to the *same* ten dimensions and
   passed through the *same* normalising flow, whose configuration was independently
   optimised for each summary and found to be the same optimum. The posteriors therefore
   differ only through the summaries.
2. **Calibration is a precondition, not a diagnostic.** No constraining-power number is
   quoted for an arm that has not passed the coverage battery. A tighter posterior counts
   as more information only once it is known not to be over-confident.

### 0.1 Stage → code

| # | stage | module | job |
|---|---|---|---|
| 1 | full-sphere maps, shape noise, BNT in $a_{\ell m}$, gnomonic patch extraction (fiducial observation cache) | `src/nn_l1_sbi/data/build_full_sphere_cross_cache.py` | `setup/build_obs_cache.slurm` |
| 2 | TFDS training set `grid_10deg_80px_nonoverlap180` | `src/nn_l1_sbi/data/tf_dataset_nbody_tomo{,_cross}.py`, loader `tfds_cross_tfdata_loader.py` | — (built once, not rebuilt by these jobs) |
| 3 | frozen per-(channel, scale) noise $\sigma$ table | `src/nn_l1_sbi/flatsky/freeze_flatsky_cross_noise.py` | `setup/freeze_sigma.slurm` |
| 4 | patch-local channel operators (conv / product / BNT) | `src/nn_l1_sbi/flatsky/flatsky_cross.py`, `flatsky/bnt_utils.py` | (library; used by 5–7) |
| 5 | starlet $\ell_1$ data vector under frozen $\sigma$ | `src/nn_l1_sbi/flatsky/flatsky_cross_l1.py`, driven by `src/nn_l1_sbi/inference/npe_l1norm_cross_jaxili_nbody_tomo.py` | `train_l1both_build.slurm` |
| 5b | observation $\ell_1$ data vectors; column slicing into arms | `src/nn_l1_sbi/data/precompute_fiducial_both_datavectors.py` | `setup/precompute_obs.slurm`, `setup/bnt_obs_and_slice.slurm`, `setup/slice_raw.slurm`, `conv_arms/slice_conv.slurm` |
| 6 | joint (pairwise) $\ell_1$ data vector | `src/nn_l1_sbi/flatsky/flatsky_joint_stats.py`, driven by `src/nn_l1_sbi/data/build_flatsky_joint_arm.py` | `build_joint_arm.slurm` |
| 7 | CNN-VMIM compressor on maps | `src/nn_l1_sbi/cnn/npe_cnn_nbody_tomo.py` | `cnn_nobnt.slurm`, `cnn_bnt.slurm` |
| 7b | CNN summaries of the 9000 observations (+ G1 gate) | `src/nn_l1_sbi/cnn/build_fiducial_summaries_cnn.py` | `cnn_fidsum.slurm` |
| 8 | VMIM compression of $\ell_1$ vectors → 10-D | `src/nn_l1_sbi/compression/vmim_from_cache.py` (companion `vmim_maf_companion.py`) | `vmim_*.slurm`, `conv_arms/conv_arm.slurm` |
| 9 | RealNVP NPE on 10-D summaries | `src/nn_l1_sbi/compression/train_nde_from_compressed.py`, helpers in `src/nn_l1_sbi/cnn/train_jaxili_from_compressed.py` | (library; used by 10) |
| 10 | 9000-observation population sweep, per-obs metrics, median FoM₃ | `src/nn_l1_sbi/inference/population_sweep.py` | `sw_*.slurm`, `sw_joint_chain.slurm`, `sw_cnn.slurm`, `conv_arms/conv_arm.slurm` |
| 10b | compressor deep-ensemble evaluation | `src/nn_l1_sbi/inference/ensemble_eval.py` | (ensemble rows are produced by `population_sweep` with three `--arm-dir`) |
| 11 | TARP coverage + SBC ranks | `src/nn_l1_sbi/inference/tarp_stratified_val.py`, `run_tarp_coverage.py`, orchestrated by `run_joint_gate_c.py` | — (see §14) |
| 12 | figures | `src/nn_l1_sbi/figures/fig_*.py` (`fig_pipeline.py` draws this diagram) | — |

---

## 1. Simulations, cosmology, and the parameter vector

**Suite.** `CosmoGridV1` (Kacprzak et al. 2023; Fluri et al. 2022), full-sky weak-lensing
convergence maps from the `PkdGrav3` N-body code, delivered as HEALPix maps at
`nside = 512`, band-limited at `lmax = 1024` (multipoles $\ell \le 1024$).

**Parameter space.** A flat $w$CDM model in six parameters, sampled with a Sobol sequence
over the CosmoGridV1 grid:

$$\boldsymbol{\theta} = (\Omega_{\rm m},\ \sigma_8,\ w_0,\ H_0,\ n_{\rm s},\ \Omega_{\rm b}).$$

The parameter order is fixed everywhere in the code, and `H0` is rescaled to
`h0 = H0/100` in preprocessing (`theta[3] /= 100`) before it ever reaches a network.
The fiducial cosmology used for the mock observations is

```
theta_fid = [Omega_m 0.26, sigma_8 0.84, w0 -1.0, h 0.6736, n_s 0.9649, Omega_b 0.0493]
```

Constraints are reported in the **lensing-constrained subspace**
$(\Omega_{\rm m}, \sigma_8, w_0)$; the remaining three parameters are inferred but only
weakly constrained, and are marginalised over.

**Cosmology splits.** Grid cosmology indices `1–899` form the training split; indices
`900–1299` form the validation/test split. The two are never mixed or swapped: the
held-out cosmologies are the only unbiased estimate of generalisation available.
The learned stages (compressors and density estimators) see $3.2\times10^5$ patch
examples drawn from those **899** cosmologies — exactly the same simulations from which
the $\ell_1$ data vectors are built, so the comparison is not confounded by training data.

**Tomography.** Four tomographic redshift bins, read from the `kg/stage3_lensing{1..4}`
maps of each realisation's HDF5 file, with mean redshifts $z \approx 0.3, 0.5, 0.7, 0.9$.
The analysis works directly on the convergence $\kappa$: **both summaries are computed
from $\kappa$ itself, with no mass-mapping reconstruction step from shear.**

**In code.** The four-bin selection is `--nbins 4 --tomo-bin-indices 1,2,3,4` on every
driver; CosmoGridV1 is located by `--cosmogrid-meta $NN_L1_SBI_COSMOGRID/CosmoGridV1_metainfo.h5`
plus the `CG_PARENT` environment variable that `build_full_sphere_cross_cache._remap_path_par`
honours. The HDF5 reader lives in `data/build_full_sphere_cross_cache.py`; the TFDS builder
`data/tf_dataset_nbody_tomo.py` performs the same read for the training set.

---

## 2. Noise model

Shape noise is added **on the full sphere, per tomographic bin, before any transform**,
as independent per-pixel Gaussian noise:

$$\sigma_{\rm pix} = \frac{\sigma_\epsilon}{\sqrt{n_{\rm gal}\, A_{\rm pix}}},$$

with

| quantity | value |
|---|---|
| intrinsic ellipticity dispersion (per component) | $\sigma_\epsilon = 0.26$ |
| effective source density | $n_{\rm gal} = 10\ {\rm arcmin}^{-2}$ **per tomographic bin**, i.e. $40\ {\rm arcmin}^{-2}$ summed over the four bins |
| pixel area | `hp.nside2pixarea(512, degrees=True) * 3600` = $47.210\ {\rm arcmin}^2$ |
| resulting per-pixel noise | $\sigma_{\rm pix} = 0.26/\sqrt{10\times47.210} = 0.011966$ |

**The density is 10 per bin, not 7.5 — read it off the builder.** The maps every arm is
trained and evaluated on were written by `data/build_full_sphere_cross_cache.py`, whose
`--galaxy-density` default is `10.0` (arcmin⁻², applied identically to each of the four
bins: one `noise_std`, one `rng.normal` draw over the `(4, npix)` noiseless stack).
`setup/build_obs_cache.slurm` passes neither `--sigma-e` nor `--galaxy-density`, so both
defaults stand, and the TFDS training set is a reserialisation of caches from the same
builder — the noise is baked in there, not re-drawn at training time. The two NPE drivers
(`inference/npe_l1norm_cross_jaxili_nbody_tomo.py`, `cnn/npe_cnn_nbody_tomo.py`) do carry
a `--galaxy-density` default of `30/4 = 7.5`, but that flag feeds only the legacy
on-the-fly noise paths (`--cross-maps-route flat`, `load_observed_map`), which the
production `flat_local` route does not use. **A referee recomputing the shape noise must
use 10 arcmin⁻² per bin**; the resulting $\sigma_{\rm pix}$ is $0.011966$, not the
$0.013817$ that $7.5\ {\rm arcmin}^{-2}$ would give. (Earlier drafts of this document, and
the manuscript's "*Euclid*-like 30 arcmin⁻² total" wording, describe a configuration the
released code does not implement; the code is the authority and the manuscript sentence
needs correcting, not the code.)

The noise draw is **deterministic in `(cosmo_idx, perm)`**
(`seed = noise_seed_base + 100*cosmo_idx + perm`, base `12345`), so any stage of the
pipeline can be rebuilt bit-for-bit. Noise is independent between bins — a fact that
becomes central under the BNT transform (§5).

**Ordering invariant (do not reorder):** noise is injected *before* the BNT transform and
before the patch extraction. Applying BNT to noiseless maps and adding noise afterwards
would produce a different (and wrong) noise covariance in the nulled frame, and is one of
the classic ways this pipeline breaks.

**In code.** `data/build_full_sphere_cross_cache.py` implements this in
`_per_pixel_noise_std` and `compute_noisy_alms` (seed from `_noise_seed`, draw added to the
noiseless bin stack, then the SHT); the patches are cut afterwards by
`_patch_one_realization`. Its defaults `sigma_e 0.26`, `galaxy_density 10.0`,
`noise_seed_base 12345` are the pinned values and must not be passed differently. The same
constants are re-read out of the fiducial cache by the $\sigma$-freezing stage below, so
they cannot drift between the two.

### 2.1 Noise *freezing* — the SNR denominator

The $\ell_1$-norm bins wavelet coefficients by their signal-to-noise
$\nu = w_j/\sigma_j$. The denominator $\sigma_j$ is **not** re-measured per map. It is a
frozen table of per-(channel, scale) noise levels, measured once and reused by every arm:

- Estimated at the **fiducial cosmology** from $R = 48$ noise realisations, by subtracting
  the across-realisation per-pixel mean (which removes the fixed signal term) and taking
  $\sigma(c,s) = \sqrt{\mathrm{mean}_{\rm patch,pix} \mathrm{Var}_r[\text{coeff}]}$, with
  the $R/(R-1)$ Bessel correction.
- **Per (channel, scale), not a single white number.** Cross-map noise is *coloured*: the
  product / convolution of two noisy maps contains $S\!\times\!N$ and $N\!\times\!N$ terms,
  so propagating one white auto-$\sigma$ per scale is simply wrong. The empirical
  per-channel table is the fix.
- **Frozen, uniform, and cosmology-independent by design.** Because the frozen $\sigma$ is
  a fixed deterministic normalisation — a change of variables applied identically to
  simulations and to the mock data — it cannot bias the inference. It must *not* be made
  per-patch or cosmology-dependent.
- A **separate table is frozen in the BNT basis** (`..._sigma_bnt.npz`), since the nulling
  mixes the per-bin noise.

Why this matters: the cross-map channels are smaller in amplitude than the auto-maps by
orders of magnitude. Under a single shared SNR threshold their wavelet SNR collapses to
nearly zero, the cross-channel histogram bins empty out, and the inter-bin information
those channels carry is silently discarded. The per-channel frozen noise level is what
restores them.

**Runs as** — `scripts/slurm/setup/freeze_sigma.slurm`, after the observation cache:

```bash
python -u -m nn_l1_sbi.flatsky.freeze_flatsky_cross_noise \
    --op both --n-real 48 --n-scales 5 --seed-base 987654321 \
    --workers 32 --out-dir "$NN_L1_SBI_DATA/sigma_tables"
```

with `FLATSKY_FID_CACHE` pointing at `.../full_sphere_cache_fiducial_10deg/nobnt/obs`.
The job then cross-checks nine metadata fields against the intact BNT twin
(`$NN_L1_SBI_SIGMA_BNT`) and asserts `bnt == False` on the new table — BNT only mixes
channels, so those fields must agree exactly. Only the four no-BNT rows depend on the
regenerated file; the BNT rows read the twin directly.

---

## 3. Flat-sky patch extraction

The maps are analysed as flat-sky patches — the natural input for a convolutional
compressor, and the geometry a patch-level survey actually observes.

| property | value |
|---|---|
| projection | gnomonic (tangent-plane) |
| patch size | $10^\circ \times 10^\circ$ |
| pixels | $80 \times 80$ |
| pixel scale | $7\farcm5$ |
| patch centres | 180 per realisation, non-overlapping, laid on an `nside = 64` grid with a minimum angular separation of $14.2^\circ$ |
| latitude cut | $\lvert b \rvert < 75^\circ$ (avoids the polar-cap projection distortion) |

**Why $10^\circ$ and not larger.** Both the flat-sky approximation and the patch-local
cross-map construction (§4) are more nearly valid on smaller fields: the fractional
distortion of the gnomonic projection at the patch corners falls from $6.3\%$ on a
$20^\circ$ patch to $1.5\%$ at $10^\circ$. One $10^\circ$ patch $= 100\ {\rm deg}^2$ is
one "observation"; per-observation constraints are correspondingly weaker than for a
$20^\circ$ patch, which is the intended trade.

**Mean subtraction (mandatory).** Each patch is demeaned **per channel** before
compression. The mean convergence over a finite field is not an observable, owing to the
mass-sheet degeneracy. Retaining it lets a learned compressor read the simulated
per-channel mean as a clean, shot-noise-free function of cosmology — a feature no real
survey supplies — and it reports constraints far tighter than the observable data could
support. The $\ell_1$-norm is insensitive to the offset (the wavelet filters are
compensated), but the subtraction is applied to *all* arms for uniformity. A compressor
checkpoint trained with and one trained without demeaning are physically incompatible and
must never be mixed. In code this is the `--zero-mean-maps` flag, present on every
training driver.

**The 180 centres are a pinned input, not a computed one.**
`tf_dataset_nbody_tomo._build_non_overlapping_centers` orders candidate centres with
`np.argsort(|lat|)`, whose default quicksort is not stable, and every HEALPix ring is a
block of exact ties — so identical arguments give a different tiling under a different
numpy. Measured: numpy 1.24.4 and 1.26.4 share only **2 of 180** centres, and only the
1.24.4 tiling matches the paper (confirmed 22/22 by bit-exact reprojection against the
TFDS). The verified file ships as `scripts/slurm/inputs/centers_PAPER_180.npy` and is
always passed via `--centers-npy`. **Never regenerate it.**

**Runs as** — `scripts/slurm/setup/build_obs_cache.slurm`:

```bash
GEO="--field-size 10 --field-npix 80 --n-centers 180 --center-nside 64 \
     --min-separation-deg 14.2 --max-abs-lat 75"
FIDPERMS=$(seq -s, 0 199)          # "0,1,...,199" — the eval uses perm < 50

python -u -m nn_l1_sbi.data.build_full_sphere_cross_cache \
    --cosmo-meta   "$NN_L1_SBI_COSMOGRID/CosmoGridV1_metainfo.h5" \
    --cosmogrid-root "$NN_L1_SBI_COSMOGRID" \
    --cosmo-subset fiducial --cosmo-id cosmo_fiducial \
    --realizations "$FIDPERMS" \
    $GEO \
    --centers-npy  "$NN_L1_SBI_INPUTS/centers_PAPER_180.npy" \
    --regime nobnt --num-workers 40 \
    --out-dir "$NN_L1_SBI_DATA/full_sphere_cache_fiducial_10deg"
```

The cache holds all 200 perms even though the evaluation uses only `perm < 50`, because
that is what the paper built. The job's inline
`verify_cache` gate checks the patch tensor shape `(180, 80, 80, 10)`, finiteness, the
polar leak `max|lat| < 75`, the per-channel scales, and the cache's own centres against
the pinned file to `< 1e-4`.

**`--regime nobnt` even for the BNT rows.** The cache stores raw noisy autos; BNT is
applied later, on device, as a channel mix (`--flatsky-channel-mix bnt` / `--flatsky-bnt`),
in the order noise → demean → BNT → cross-build → whiten. CNN and $\ell_1$ rows, both
frames, read the same observation cache.

### 3.1 The dataset

Training patches are materialised into a TFDS dataset,
`nbody_cosmogrid_dataset_tomo_cross/grid_10deg_80px_nonoverlap180`, built by
`data/tf_dataset_nbody_tomo_cross.py` and read through `data/tfds_cross_tfdata_loader.py`
(importing the builder module is what registers the dataset — keep that import):

- feature `map_nbody` of shape `(80, 80, 10)` float32 — **4 auto channels + 6 stored
  harmonic-cross channels** — plus `theta` (6), `cosmo_idx`, `perm`, `patch`;
- 1,636,740 examples: train = 1,132,740 (cosmologies 1–899), test = 504,000
  (cosmologies 900–1299).

The TFDS is ~290 GB and is **not rebuilt** by the campaign jobs; it is an input
(`$NN_L1_SBI_TFDS`, exported as `TFDS_DATA_DIR`).

**The production analysis reads only channels 0–3, the auto-maps.** The stored harmonic
cross-channels are the *full-sphere* construction and are deliberately not used for the
headline result (§4.1); the cross-maps that enter the science are rebuilt patch-locally
from the patch's own auto-maps.

Training subsets actually consumed:

| set | selection | size |
|---|---|---|
| train | cosmologies 1–899, realisation permutations 5–6, 180 patches | $899\times2\times180 = 323{,}640 \approx 3.2\times10^5$ |
| validation | cosmologies 900–1299, permutations 0–1 | $400\times2\times180 = 144{,}000$ |

These loader parameters are baked into the drivers (train: split `train`, perms 5–6, flip
seed 1001; val: split `test`, perms 0–1, no flip, seed 2001) and match every campaign arm.

**Compressor / NDE separation.** Where a *trainable* compression stage exists, the
compressor and the density estimator are trained on disjoint `(cosmology, patch)` example
sets (a 70/30 partition, asserted at load time). Training both on the same examples leaks
the compressor's per-example output into the NDE and narrows posteriors on held-out data.

### 3.2 The mock observations

The evaluation population is built from the cached fiducial-cosmology realisations
(200 realisation indices $\times$ 180 patches). The headline population takes the first
**50** realisation indices:

$$180\ \text{patches} \times 50\ \text{realisations} = 9000\ \text{mock observations per arm}.$$

Every arm is evaluated on the **same 9000 mocks** — a paired design, so population scatter
and training-set noise cancel in *differences* between summaries (§10.2). In code this is
`--n-obs 9000 --max-perm 50` on every `population_sweep` invocation.

---

## 4. Channel construction: tomographic cross-maps

The four auto-maps carry the bulk of the convergence signal, but the tomographic lensing
kernels overlap while the shape noise is independent, so the bins carry correlated
information: the cross-correlations trace the growth of structure and the distance ratios.
A **per-channel** statistic such as the $\ell_1$-norm has no direct access to this
inter-bin information — it reads one map at a time. The CNN, which ingests the four maps
jointly from its first convolutional layer, does. Supplying the analytical statistic with
explicit inter-bin channels is how the gap is closed.

The standard survey remedy — merge the galaxy catalogues of two bins and measure the
statistic on the combined map — supplies only *linear* cross-bin channels (the combined
map is, pixel by pixel, a count-weighted average of the per-bin maps, noise included).
Those determine the second-order cross-correlations but, for any fixed set of
combinations, only part of the higher joint structure. This pipeline instead supplies
**quadratic** channels, one pair of cross-maps per bin pair $i<j$ (6 pairs for 4 bins).

**Convolution cross-map**

$$\kappa^{ij}_{\rm conv} = \mathcal{F}^{-1}\!\big[\mathcal{F}(\kappa_i W)\ \mathcal{F}(\kappa_j W)\big],$$

with $\mathcal{F}$ the FFT and $W$ a cosine apodisation window (roll fraction 0.10). This
is the flat-sky analogue of the harmonic product of spherical-harmonic coefficients. Each
pixel sums over all pixel pairs of the patch; that patch-wide sum concentrates the map's
variance in a few tens of effective large-scale modes, so its one-point statistics carry
mostly two-point cross information — which the auto-maps already largely constrain.

**Pointwise-product cross-map**

$$\kappa^{ij}_{\rm prod} = \kappa_i\,\kappa_j.$$

The local complement of the convolution: it multiplies the two bins pixel by pixel, so
spatially coincident structure enters coherently, and it preserves the small scales that
the patch-wide convolution sum averages away. Its spatial mean is the zero-lag
cross-correlation $\langle\kappa_i\kappa_j\rangle$; what its one-point statistics add
beyond that is the higher joint structure of the bin pair.

**The four analytical input configurations** (channel counts for 4 tomographic bins):

| arm | channels | composition | `--cross-op` |
|---|---|---|---|
| `auto` (none) | 4 | the four auto-maps | `none` |
| `+conv` | 10 | 4 auto + 6 convolution cross-maps | `conv` |
| `+product` | 10 | 4 auto + 6 product cross-maps | `product` |
| `+conv+product` (`both`) | 16 | 4 auto + 6 conv + 6 product | `both` |

The `both` arm is the superset: `none`, `conv` and `product` are exact **column slices** of
the `both` data vector, so all four arms can be built from a single expensive loader pass —
which also guarantees they see bit-identical maps.

The CNN reads the **four auto-maps alone** (`--cross-op none`, `channel_slice=slice(0,4)`).
The cross-maps exist for the per-channel $\ell_1$-norm, which lacks the network's native
joint access; supplying them to the CNN yields no systematic gain (§13.2).

**In code.** The operators are `flatsky/flatsky_cross.py` (`build_channels_*`, applied
patch-locally on device). The slicing is done by three cheap jobs that never touch the
TFDS: `setup/slice_raw.slurm` (no-BNT `both` → `product`, `none`),
`setup/bnt_obs_and_slice.slurm` (the BNT equivalents), and
`conv_arms/slice_conv.slurm` (`conv`, both frames). All three take the column selection
from `flatsky_cross_l1.op_feature_columns(op, 4, 200)` so the column order cannot drift
from what `build_channels_*` emits, and `slice_conv` asserts the sliced auto block is
**byte-identical** to the independently built `none` arm.

### 4.1 Why the cross-maps are built *patch-locally*

The cross-maps are constructed from the two auto-maps **of the patch alone**, so they are
by construction available to a survey that observes only that patch. The alternative — the
idealised full-sphere construction that forms $a^i_{\ell m}a^j_{\ell m}$ on the whole
sphere and only then cuts out patches — produces strongly non-local channels: they carry
$12$–$20\%$ of their variance on super-patch scales ($\ell < 18$), against $0.4$–$1\%$ for
the auto-maps, with the cross power piling up at low multipole (median multipole
$\approx 60$–$90$, against an effective multipole $\approx 600$ for the autos). Most of
the full-sphere cross-map "gain" is that super-patch information, which a survey observing
the patch alone cannot access. Using it would credit the patch with information it does
not have and overstate the measured constraining power.

This is the meaning of `--cross-maps-route flat_local` / `--cnn-map-route flat_local`,
which every production job passes. The full-sphere channels remain in the TFDS (channels
4–9) and are the subject of the leakage figure (`figures/fig_leak.py`), not of the results.

---

## 5. The BNT transform (optional frame)

The **Bernardeau–Nishimichi–Taruya (BNT)** transform is a fixed, linear, **invertible**
recombination of the tomographic bins, $\tilde{\kappa} = B\kappa$ with $B$ a $4\times4$
matrix, that **localises the lensing kernel in redshift**. In the original frame every
tomographic bin has a broad kernel integrating structure over much of the line of sight;
the nulling replaces those four overlapping deep kernels with one shallow map and three
thin redshift slices.

**Why one would want it.** With compact kernels, a systematic confined to a redshift range
— for instance the small-scale, low-redshift structure affected by baryonic feedback —
affects only the corresponding nulled maps and scales, and can be isolated or cut there
without leaking into the rest of the data vector. That makes it attractive as a step
before a non-Gaussian analysis. Applied to higher-order statistics, however, it has been
found to inflate the cosmological contours dramatically, and this pipeline is built to
determine whether that reflects lost information or only the analysis frame.

**Implementation invariants:**

- BNT requires the **full 4-bin tomography** (`--nbins 4 --tomo-bin-indices 1,2,3,4`); the
  configuration validator refuses anything else.
- **Shape noise is injected before the transform, always.** In the full-sphere route the
  matrix is applied in $a_{\ell m}$ space (linear combinations of bins commute with the
  SHT); in the patch-local route it is applied to the noisy patch auto-maps before any
  channel is built.
- Because $B$ is linear and row-sum-zero in practice, a pre-BNT demeaned map is still
  demeaned post-BNT. Do not demean twice.
- The per-(channel, scale) noise table is **re-frozen in the BNT basis** (§2.1).
- Architecture, training, flow and gates are otherwise **identical** to the no-BNT case, so
  the only difference between the two frames is the frame.

**What it does to the noise.** Because the shape noise is independent between bins, the
subtractions that null the *signal* still **add the noise variances**: each nulled map
carries, on top of its own bin's noise, that of the bins mixed into it. Each nulled channel
integrates only a thin slice of structure, so it is signal-poor, and the noise field is now
correlated across channels. A summary that reads channels independently can no longer model
it. Consistently, with **noiseless** maps the BNT and no-BNT contours essentially coincide
for every summary considered.

**In code.** The matrix and its version tag are `flatsky/bnt_utils.py`
(`BNT_MATRIX`, `BNT_MATRIX_VERSION`). The frame is selected by
`--flatsky-channel-mix bnt` (ℓ1 driver), `--flatsky-bnt` (CNN driver) or
`--basis bnt` (joint-ℓ1 builder); the BNT rows read `$NN_L1_SBI_SIGMA_BNT` instead of
the regenerated no-BNT table, and write to the parallel `l1_datavector_cache_bnt/` tree.

---

## 6. Summary statistic (a): the starlet $\ell_1$-norm

The starlet $\ell_1$-norm provides a multiscale characterisation of the full probability
density of the convergence field, summing the absolute values of wavelet coefficients
within signal-to-noise bins (Ajani et al. 2021; Tersenov et al. 2026).

**Construction, per channel:**

1. Decompose the map with the **isotropic undecimated ("starlet") wavelet transform** into
   **5 scales** (coarse-scale mean subtracted). The four finest dyadic bands probe
   characteristic angular sizes of roughly $10'$, $20'$, $40'$ and $80'$.
2. Convert each coefficient to signal-to-noise units, $\nu = w_j/\sigma_j$, with
   $\sigma_j$ the **frozen** per-(channel, scale) noise level of §2.1.
3. Bin $\nu$ into **40 linearly spaced bins** and **sum $|w_j|$** in each bin (not count
   the pixels — the absolute-value weighting is what makes it an $\ell_1$-norm).
4. Concatenate over bins, scales and channels.

**Data-vector dimension:** $C \times 5\ \text{scales} \times 40\ \text{bins}
= 200\,C$, hence

| arm | channels $C$ | dimension |
|---|---|---|
| auto | 4 | 800 |
| +conv | 10 | 2000 |
| +product | 10 | 2000 |
| +conv+product | 16 | 3200 |

**SNR binning ranges are set per channel.** The auto channels use a range of
$[-13, 13]$; the cross channels are rescaled by their own frozen noise level before the
transform, and their histogram extent is calibrated from robust percentiles
(with a margin) over a pilot pass of training maps. The per-channel extent is not a
band-aid: the *denominator* is the correct frozen per-scale $\sigma$, and the percentiles
only set the histogram range. It is necessary because the pointwise product is
intrinsically heavy-tailed (peaky $\kappa_i\kappa_j$) while conv and auto are not — a
shared min/max range would spend nearly all bins on the product's rare tail.
The percentile settings are `--flatsky-snr-percentiles 0.5,99.5` and
`--cross-snr-percentile 1.0`; the calibrated ranges are written once to
`flat_local_ranges.npy` and reused by every arm, and `--l1-clamp-overflow` sends
out-of-range coefficients to the edge bins.

**Data-vector preprocessing** (fitted on the training split, then applied unchanged to
validation and to the observations):

```
log(1 + x)  ->  z-score standardisation  ->  clip at +/- 5 sigma
            ->  drop features with train variance <= 1e-5
```

i.e. `--summary-transform log1p-zscore --clip-value 5.0 --min-feature-variance 1e-05`.

**Never PCA the $\ell_1$ data vector.** PCA on this vector has repeatedly destroyed
constraining power in this pipeline; the compression path is VMIM (§8). Hence
`--pca-components 0`.

**Where the information lives.** A per-cell diagonal Fisher analysis of the auto data
vector (regress the $\ell_1$ value in each (bin, scale, SNR) cell on the parameters, divide
by the residual variance, sum over the four tomographic bins) shows the information
concentrating at **intermediate scales** (scales 2–3, tens of arcminutes) and **moderate
signal-to-noise** ($|\nu| \approx 1$–$2$) — the regime of mildly non-linear structure,
not the rare extreme peaks and not the noise-dominated bins. The $\Omega_{\rm m}$ and
$\sigma_8$ maps share the same structure, consistent with their strong degeneracy; $w_0$
is more diffuse. (`figures/fig_fisher.py`.)

**Runs as** — `scripts/slurm/train_l1both_build.slurm`, the **only expensive TFDS pass**
of the whole campaign. It builds the 16-channel `both` cache and `flat_local_ranges.npy`;
every other $\ell_1$ arm is a column slice of it.

```bash
python -u -m nn_l1_sbi.inference.npe_l1norm_cross_jaxili_nbody_tomo \
    --cross-maps-route flat_local --cross-op both \
    --flatsky-cross-sigma "$NN_L1_SBI_DATA/sigma_tables/flatsky_cross_noise_sigma.npz" \
    --flatsky-snr-percentiles 0.5,99.5 \
    --l1-clamp-overflow \
    --fiducial-obs-cache-dir "$NN_L1_SBI_DATA/full_sphere_cache_fiducial_10deg" \
    --cosmogrid-meta "$NN_L1_SBI_COSMOGRID/CosmoGridV1_metainfo.h5" \
    --harmonic-obs-cosmo-id cosmo_fiducial \
    --cross-tfds-name nbody_cosmogrid_dataset_tomo_cross/grid_10deg_80px_nonoverlap180 \
    --cross-tfds-data-dir "$NN_L1_SBI_TFDS" \
    --field-size 10 --field-npix 80 \
    --nbins 4 --tomo-bin-indices 1,2,3,4 \
    --n-scales 5 --l1-nbins 40 \
    --map-kind nbody --zero-mean-maps \
    --seed 41 \
    --pca-components 0 \
    --summary-transform log1p-zscore --clip-value 5.0 \
    --min-feature-variance 1e-05 \
    --cross-snr-percentile 1.0 \
    --nde-train-split train \
    --epochs 5000 --batch-size 128 --learning-rate 0.0001 \
    --npe-warmup-steps 128 --npe-decay-steps 10000 \
    --npe-samples 100000 \
    --cache-dir "$NN_L1_SBI_DATA/l1_datavector_cache" \
    --save-dir  "$NN_L1_SBI_DATA/arms/l1both_nobnt_s41" \
    --cuda-visible-devices 0 \
    --no-wandb
```

Three things about this invocation are load-bearing:

* `--fiducial-obs-cache-dir` wants the cache **ROOT**; the loader appends `<regime>/obs`
  itself. Passing `.../nobnt/obs` breaks it. (The CNN driver's equivalent flag is
  `--fiducial-obs-cache`, without `-dir`.)
* `--fiducial-summaries-out` is deliberately **absent**. On the flat-local route that dump
  bins with representative *scalar* SNR ranges while the observations use the per-channel
  `flat_local_ranges`, and the driver's own G1 gate correctly rejects it. Observation
  data vectors come from `setup/precompute_obs.slurm` instead.
* The datavector cache is shared across seeds on purpose: the training data vector comes
  from the fixed TFDS (noise baked in at TFDS build time), so it does not depend on
  `--seed`. Seeds 42/43 reuse it and skip the loader pass.

The BNT 16-channel cache is the same invocation plus `--flatsky-channel-mix bnt`, the
`_bnt` sigma table, and `--cache-dir $NN_L1_SBI_DATA/l1_datavector_cache_bnt`.

**Observation data vectors** — `setup/precompute_obs.slurm`, one pass over the small
fiducial cache (explicitly *not* over the 290 GB TFDS):

```bash
python -u -m nn_l1_sbi.data.precompute_fiducial_both_datavectors \
  --both-cache "$NN_L1_SBI_DATA/l1_datavector_cache/flat_local_both" \
  --sigma      "$NN_L1_SBI_DATA/sigma_tables/flatsky_cross_noise_sigma.npz" \
  --perms 50 \
  --out        "$NN_L1_SBI_DATA/fiducial_both_datavectors_nobnt.npz"
```

`setup/bnt_obs_and_slice.slurm` is the same with `--mode bnt`, `--sigma
"$NN_L1_SBI_SIGMA_BNT"`, and an inline slice into `product` (10 ch) and `none` (4 ch).

---

## 7. Summary statistic (a'): the joint $\ell_1$-norm

The per-channel $\ell_1$-norm reads each tomographic bin separately: at a given scale and
pixel, the four bins' coefficients form a joint 4-dimensional distribution, and the
per-channel statistic captures only its **one-dimensional marginals**. The joint
$\ell_1$-norm is the **pairwise generalisation of the statistic itself**, and it needs no
explicit cross-map channel — it is built from the four auto-maps alone.

**Definition.** At each scale $s$ let

$$u_i(\boldsymbol{p}) = \frac{(W_s\kappa_i)(\boldsymbol{p})}{\sigma_{i,s}}$$

be the coefficient of bin $i$ at pixel $\boldsymbol{p}$ in signal-to-noise units (same
frozen $\sigma$). For each bin pair $(i,j)$ and each scale, lay a fixed $K\times K$ grid
over the joint $(u_i, u_j)$ plane and sum the $\ell_1$ weight of the pixels in each cell:

$$L^{ij}_{ab} = \sum_{\boldsymbol{p}\,\in\,\mathrm{cell}(a,b)} \tfrac{1}{2}\big(|u_i(\boldsymbol{p})| + |u_j(\boldsymbol{p})|\big).$$

Concatenate the $K^2$ cell values over the $\binom{4}{2}=6$ bin pairs and the wavelet
scales. Production settings: $K = 10$, 5 scales, out-of-range coefficients **clamped to the
edge cells**, grid ranges either fixed symmetric $[-5, 5]$ in SNR or signal-adapted
percentile ranges. Feature ordering is `(pair, scale, cell_row*K + cell_col)`.

**Dimension:** $6\ \text{pairs} \times 5\ \text{scales} \times 100\ \text{cells} = 3000$.

**The figure.** The 2-D cell map, with the 1-D projections that *are* the per-channel
$\ell_1$, is drawn by `figures/fig_l1.py::plot_joint_cellmap` into
`fig_l1_joint_cellmap.pdf` (from `jointl1_nobnt_raw/fid.npz`, one representative pair and
scale, SNR half-range $\pm5$ = the adaptive `snr_range`). In the manuscript it carries its
own label, **`fig:l1joint`** — it is no longer panel (b) of `fig:l1`, which is now the
per-channel $\ell_1$ histogram figure (`plot_hist_grid`) alone.

**Why $\ell_1$ weight and not counts.** The count version of the same grid (`pair2d`) is an
estimate of the joint PDF; it over-fits badly (it reached a far higher raw figure of merit
and then **failed** the coverage gate). Summing absolute coefficient values instead makes
the statistic the genuine two-dimensional analogue of the per-channel wavelet
$\ell_1$-norm, emphasises the tails and peaks the $\ell_1$ lives on, and gives a robust,
low-variance summary of each pairwise distribution — one that stays calibratable.

**What it captures.** It encodes the one-point information of the per-channel
$\ell_1$-norm (its axis projections), the bin-pair correlation (its second cross-moments,
i.e. what the product cross-map supplies), **and** every higher pairwise joint moment —
cross-skewness, cross-kurtosis, the full non-Gaussian 2-D shape of each pair. What it does
*not* reach is the genuinely three- and four-bin connected structure.

**Why pairwise and not full 4-D.** Ideally the statistic would read all four bins at once,
as an $\ell_1$-weighted histogram over the full four-dimensional coefficient space. With
$K^4$ cells against the $80\times80 = 6400$ pixels of a patch, such a histogram is almost
everywhere empty and its occupied cells are dominated by sampling noise. The
two-dimensional pairwise grid is the highest joint dimension the patches populate densely;
the three- and four-bin structure remains accessible only to the learned compressor.

**Runs as** — `scripts/slurm/build_joint_arm.slurm <nobnt|bnt>`:

```bash
python -u -m nn_l1_sbi.data.build_flatsky_joint_arm \
    --stat jointl1 --basis "$BASIS" --k 10 --adaptive-ranges \
    --out-cache "$NN_L1_SBI_DATA/joint/jointl1_${BASIS}_raw/cache" \
    --out-fid   "$NN_L1_SBI_DATA/joint/jointl1_${BASIS}_raw/fid.npz"
```

The remaining inputs arrive through the environment: `JOINT_TFDS_DIR`, `JOINT_FID_OBS`
(`.../full_sphere_cache_fiducial_10deg/nobnt/obs`), `JOINT_SIGMA_NOBNT`, `JOINT_SIGMA_BNT`.
The job then asserts the feature dimension is exactly `6*5*10*10 == 3000`, finite, and
non-negative (the cells are sums of $|u|$).

**Two distinctions the directory names do not make for you.**
`--stat jointl1` is *not* `--cross-op both` (§4), and it is *not*
`pair2d_features(weighted=False)`, the unweighted count variant. `--rotated-binning` — a
rotated/whitened per-pair binning tested as an alternative fix for BNT over-confidence — is
also not the published lineage: it failed its BNT gate. See §11 for the sweep-directory
naming trap that follows from this.

---

## 8. Summary statistic (b): the CNN-VMIM compressor, and VMIM in general

### 8.1 What VMIM is

**VMIM = Variational Mutual Information Maximisation.** A compressor $f_\eta$ is trained
to make its output as informative about the parameters as possible, by maximising a
variational lower bound on the mutual information $I(\boldsymbol{\theta}; f_\eta(x))$:

$$\eta^\star = \arg\max_\eta\ \mathbb{E}_{p(\boldsymbol{\theta},x)}\Big[\log q_\psi\big(\boldsymbol{\theta} \,\big|\, f_\eta(x)\big)\Big],$$

where $q_\psi$ is an **auxiliary conditional density** — here a normalising flow — trained
jointly with the compressor. Maximising this bound drives the summary to retain the
parameter-relevant information in the input. **The auxiliary flow is discarded after
training** and plays no role in the inference: the summary, not the companion, is the
product. (The companion implementation is `compression/vmim_maf_companion.py`.)

VMIM is used twice in this pipeline: to train the CNN compressor on maps (§8.2), and to
compress the high-dimensional $\ell_1$ data vectors to ten dimensions (§8.3).

### 8.2 The CNN arm

- **Input:** the stack of four tomographic auto-maps, `(80, 80, 4)`, noisy and demeaned
  (BNT-transformed in the BNT frame). The network mixes all four channels from its first
  convolutional layer, so inter-bin information is available to it by construction.
- **Architecture:** a residual convolutional network of the **ResNet-18** family, mapping
  the stacked channels through global average pooling and a dense head to a $d = 10$
  summary $\boldsymbol{s} = f_\eta(\boldsymbol{\kappa})$.
- **Normalisation:** standard **batch normalisation is avoided** on the many-channel input.
  Its statistics, computed over batches that mix cosmologies, average away the cosmological
  signal and suppress the constraining power (group-normalised variants are used instead).
- **Training:** VMIM objective with a conditional-RealNVP companion, $5\times10^4$ steps,
  checkpointing on the **validation** loss (best-val, never last-step).
- **Output handling:** the 10-D summary is standardised to zero mean and unit variance
  before inference.

**Architecture search (so the learned side is not straw-manned).** The adopted
configuration is the best-performing point of a systematic sweep over plain, residual and
attention-augmented compressor networks (up to a ResNet-50 with group normalisation and a
cross-channel attention block), over the summary dimension, and over the optimiser
schedule and training budget (including a dedicated `optuna` search). The constraining
power is stable across these choices to within the seed-to-seed scatter, and alternatives
reach within a few per cent of the adopted one through the same inference flow. A deeper
ResNet-50 trained longer *reduces* the figure of merit and overfits the 899 cosmologies —
its validation loss bottoms out at about a third of training and drifts upward thereafter
— which identifies the limiting resource as the **number of distinct cosmologies**, shared
by both summaries, not network capacity.

**Runs as** — `scripts/slurm/cnn_nobnt.slurm` (and `cnn_bnt.slurm`, identical plus
`--flatsky-bnt`), looping seeds 41/42/43:

```bash
python -u -m nn_l1_sbi.cnn.npe_cnn_nbody_tomo \
    --train-compressor --exit-after-compress \
    --cnn-map-route flat_local --cross-op none \
    --compressor-arch resnet18 --compressor-dim 10 \
    --compressor-steps 80000 --compressor-batch-size 128 --compressor-lr 5e-4 \
    --compressor-checkpoint-policy best_val --compressor-save-every 5000 \
    --cross-tfds-name nbody_cosmogrid_dataset_tomo_cross/grid_10deg_80px_nonoverlap180 \
    --cross-tfds-data-dir "$NN_L1_SBI_TFDS" \
    --fiducial-obs-cache "$NN_L1_SBI_DATA/full_sphere_cache_fiducial_10deg" \
    --cosmogrid-meta "$NN_L1_SBI_COSMOGRID/CosmoGridV1_metainfo.h5" \
    --field-size 10 --field-npix 80 --nbins 4 --tomo-bin-indices 1,2,3,4 \
    --map-kind nbody --zero-mean-maps --seed $SEED \
    --cache-dir "$NN_L1_SBI_DATA/cnn/nobnt_s${SEED}/cache" \
    --save-dir  "$NN_L1_SBI_DATA/cnn/nobnt_s${SEED}" --total-steps 2 \
    --cuda-visible-devices 0 --no-wandb
```

`--exit-after-compress` (with the placeholder `--total-steps 2`) stops the driver after the
compressor stage: the NDE for the Table-1 rows is trained later, by `population_sweep`,
under exactly the same configuration as every $\ell_1$ arm. Note `--compressor-steps 80000`
here against the $5\times10^4$ steps of the method description above — the job scripts are
the operative recipe.

**Observation summaries** — `scripts/slurm/cnn_fidsum.slurm` compresses the 9000 fiducial
observations through each finished CNN arm:

```bash
python -u -m nn_l1_sbi.cnn.build_fiducial_summaries_cnn \
    --arm-dir "$NN_L1_SBI_DATA/cnn/$ARM" \
    --obs-cache "$NN_L1_SBI_DATA/full_sphere_cache_fiducial_10deg" \
    --max-perm 50 --g1-tol 9e-4 --cuda-visible-devices 0
```

Its **G1 gate** recomputes the driver's own persisted observed vector
(`cache/cnn_obs.npz`, perm 0 / patch 0) and requires `max|Δ| ≤ 9e-4`, and it re-hashes the
compressor checkpoint (SHA-256) so an arm can never be evaluated with another seed's
weights. A failure is fatal for that arm — the summaries would be wrong and the FoM₃
meaningless. `scripts/slurm/build_fiducial_summaries_cnn_flatlocal.py` is the
flat-local-only variant that actually produced the published summaries; it is functionally
equivalent on this route and slightly stricter.

### 8.3 VMIM compression of the analytical data vectors

The learned compressor already returns a ten-dimensional summary. The $\ell_1$ data
vectors (dimension of order $10^3$: 800 to 3200, and 3000 for the joint $\ell_1$) are
compressed to ten dimensions as well, so that the two summaries enter the inference
**identically**:

- **Compressor:** a small fully-connected network, **two hidden layers of 256 units**,
  output dimension **10**.
- **Objective:** the same VMIM objective, with a conditional-RealNVP companion
  (4 coupling blocks $\times$ 128 hidden units).
- **Optimisation:** ~$3\times10^4$ steps, batch size 512, learning rate $5\times10^{-4}$,
  best-validation checkpointing.
- **Seeds:** the default single compressor seed is 41; seeds 41/42/43 are trained where a
  three-compressor ensemble is required (§9.3) or where the error bar is being measured
  (§10).
- Preprocessing (log1p / z-score / clip / min-variance mask) is fitted on the training
  split *before* compression and stored, so that a later `--no-train` evaluation cannot
  silently apply a different transform than the checkpoint was trained under.

This compression is **effectively lossless**: modelling the raw $\ell_1$ data vector
directly, with a flow adapted to its dimension, gives the same constraints as the
compressed pipeline.

**Runs as** — the same five flags for every $\ell_1$ arm; only the input cache changes.
From `scripts/slurm/vmim_none_nobnt.slurm` (and its `product`/`bnt`/`joint`/`conv` twins):

```bash
for SEED in 41 42 43; do
  python -u -m nn_l1_sbi.compression.vmim_from_cache \
    --cache-dir "$NN_L1_SBI_DATA/l1_datavector_cache/flat_local_none" \
    --fid-npz   "$NN_L1_SBI_DATA/fid_raw_none.npz" \
    --out-cache "$NN_L1_SBI_DATA/vmim/none_nobnt_s${SEED}" \
    --out-fid   "$NN_L1_SBI_DATA/vmim/none_nobnt_s${SEED}/fiducial_summaries.npz" \
    --summary-dim 10 --hidden 256,256 --nf-layers 4 --nf-hidden 128 \
    --steps 30000 --seed $SEED \
    --preproc-transform log1p-zscore --clip-value 5 --min-feature-variance 1e-5 \
    --max-minutes 35 --cuda-visible-devices 0
done
```

| arm | `--cache-dir` | `--fid-npz` | `--out-cache` |
|---|---|---|---|
| ℓ1 auto, no-BNT | `l1_datavector_cache/flat_local_none` | `fid_raw_none.npz` | `vmim/none_nobnt_s4{1,2,3}` |
| ℓ1 auto, BNT | `l1_datavector_cache_bnt/flat_local_none_bnt` | `fid_raw_none_bnt.npz` | `vmim/none_bnt_s4{1,2,3}` |
| ℓ1 +product, no-BNT | `l1_datavector_cache/flat_local_product` | `fid_raw_product.npz` | `vmim/product_nobnt_s4{1,2,3}` |
| ℓ1 +product, BNT | `l1_datavector_cache_bnt/flat_local_product_bnt` | `fid_raw_product_bnt.npz` | `vmim/product_bnt_s4{1,2,3}` |
| ℓ1 +conv, both frames | `l1_datavector_cache*/flat_local_conv*` | `fid_raw_conv*.npz` | `vmim/conv_{nobnt,bnt}_s4{1,2,3}` |
| ℓ1 +conv+product, both frames | `l1_datavector_cache*/flat_local_both*` | `fid_raw_both*.npz` | `vmim/both_{nobnt,bnt}_s4{1,2,3}` |
| joint ℓ1, both frames | `joint/jointl1_<basis>_raw/cache` | `joint/jointl1_<basis>_raw/fid.npz` | `joint/jointl1_<basis>_s4{1,2,3}` |

### 8.4 Why lossy compression does not invalidate the inference

A learned compressor, even a lossy one, does not break NPE **provided the same compressor
is applied to the simulations used for training and to the observed data**. NPE then
targets the exact posterior *given the summary*: if the summary discards information, the
posterior widens towards the prior, but it remains calibrated, so the enlarged
uncertainties honestly report the loss. For the $\ell_1$ summaries the compression is in
any case effectively lossless; for the learned compressor, any information loss widens its
posterior rather than making it over-confident.

---

## 9. Inference: neural posterior estimation

### 9.1 The density estimator

Given a ten-dimensional summary $\boldsymbol{s}$, the posterior
$p(\boldsymbol{\theta}\,|\,\boldsymbol{s})$ is estimated with a **conditional normalising
flow** trained by maximising the expected log-probability of the true parameters over the
joint distribution of training pairs (neural posterior estimation, NPE).

| component | setting |
|---|---|
| flow family | coupling-based **RealNVP** (`sbi_lens` `ConditionalRealNVP`) |
| capacity | **4 coupling blocks, width 128** |
| optimiser | Adam, warmup-then-cosine learning-rate schedule |
| stopping | early stopping on a held-out validation split |
| posterior samples | $\mathcal{O}(10^3)$ per (observation, seed) in the population sweep; $10^5$ for published corner plots |

The same flow configuration serves **every** summary. It was optimised for each summary
independently — over the flow family (RealNVP vs. a masked autoregressive flow of five
autoregressive transforms with two hidden layers of fifty units; vs. a neural spline flow)
and over capacity (deeper and wider RealNVP variants) — and the same configuration came out
best for all of them. Added capacity or flexibility *widened* the posteriors rather than
sharpening them, identifying the production configuration as the optimum of the family.
Compressing every summary to the same ten dimensions is what makes this matched-and-
optimised comparison possible.

**In code.** The training entry point is `train_sbilens_realnvp` in
`compression/train_nde_from_compressed.py`; the flow hyper-parameters are a module-level
frozen block (`FLOW = SimpleNamespace(...)`) in `inference/population_sweep.py`. Sampling,
FoM₃ and posterior-moment helpers live in `cnn/train_jaxili_from_compressed.py`. All of
these are imported *inside functions* so that importing the package does not drag in
TensorFlow/JAX/torch — keep them there.

**Never put a flow on a raw $\ell_1$ vector.** The same RealNVP craters on the 2000-D
vector (observed as all-NaN training). The compression stage is not optional.

### 9.2 Flow seeds are pooled

Each arm trains **three NDE seeds**, and their posteriors are **pooled per observation**
before any metric is computed: the sampler is run once per seed at the same observation,
the sample sets are concatenated, non-finite rows are dropped, and the metrics are computed
on the pooled cloud. (An observation retaining fewer than 100 finite pooled samples is
skipped and reported.) Pooling the flow seeds removes NDE-training stochasticity from the
per-observation numbers; the residual effect on the medians is small — 0.3–1.5 % as
measured (§10.2), against the several-per-cent compressor-seed term. It is a *different*
quantity from the compressor-ensembling shift of §9.3, which is much larger.
In code: `--seeds 41,42,43`, identical for all 54 sweep directories.

### 9.3 Compressor deep ensembles (a calibration tool, not a tuning knob)

Where the single-compressor posterior fails the coverage battery, the quoted estimator is a
**deep ensemble over compressor seeds**: the posteriors of three independently trained VMIM
compressors (seeds 41/42/43), each with its three pooled NDE seeds, are pooled per physical
observation. Diversifying the 10-D summary washes out the mild over-confidence that arises
because the VMIM compressor and the flow are trained on the same maps. In code this is
simply three `--arm-dir` flags on one `population_sweep` call (9 pooled flows).

Where it is used, and what it does:

| arm | single-compressor gate | ensemble gate |
|---|---|---|
| joint $\ell_1$, no-BNT | caveat (SBC rank std ≈ 0.31) | **clean PASS** (0.299/0.298/0.298) |
| joint $\ell_1$, BNT | **FAIL** (TARP worst +0.110, SBC ≈ 0.33) | **clean PASS** (0.304/0.304/0.298) |
| $\ell_1$ auto / +product, BNT | mildly over-confident | calibrated as ensembles |
| $\ell_1$ arms, no-BNT | pass | quoted as singles (the ensemble is *not* the quoted read; the shift is large — see below) |
| CNN | pass (mildly *conservative*) | **do not ensemble** — it moves the CNN further towards conservative, the wrong direction |

**How large is the single → ensemble shift for the no-BNT $\ell_1$ arms?** In the
retrained data shipped with this repository, −9.6 % (auto), −9.5 % (+conv), −5.5 %
(+product) and −12.2 % (+conv+product) — *not* the "≤1.2 %" that circulated in the
pre-recovery published run (where it was quoted as 2448 → 2429 and 3045 → 3009). Those
published-run numbers are **superseded**; the authority is
`data/table1/TABLE1_ERRORBARS_V2.json` (`bias_pct`), tabulated in full in §13.4. The
sub-per-cent-to-1.5 % figure quoted elsewhere in this document is a different quantity — the
**NDE / flow seed** term of §9.2 and §10.2 — and must not be read as an ensembling shift.

Two things the ensemble is **not**: it is not a way to make a number bigger (it generally
*reduces* the figure of merit, by removing over-confidence — in the shipped retrained data,
joint $\ell_1$ no-BNT single 3762 $\to$ ensemble 3380, i.e. −10.2 %; the published central
for that row, 3371, is likewise an ensemble read), and its effect is a **bias correction**,
never an error bar (§10.2). A rotated/whitened per-pair binning was tested as an alternative fix for the same
over-confidence and did **not** work — a pairwise grid cannot follow BNT's full 4-D
cross-channel rotation.

---

## 10. From posteriors to the headline number

### 10.1 The figure of merit

For each of the 9000 mock observations, from the pooled posterior samples restricted to the
three science parameters:

- marginal uncertainties $\sigma(\Omega_{\rm m})$, $\sigma(\sigma_8)$, $\sigma(w_0)$;
- the three pairwise 2-D figures of merit;
- the **three-parameter figure of merit**

$$\boxed{\ \mathrm{FoM}_3 \;=\; \frac{1}{\sqrt{\det \mathbf{C}_{\Omega_{\rm m},\sigma_8,w_0}}}\ }$$

where $\mathbf{C}$ is the $3\times3$ sample covariance of the posterior samples over
$(\Omega_{\rm m}, \sigma_8, w_0)$ — in code, `1/sqrt(det(np.cov(samples[:, :3].T)))`, which
is the same quantity as $\exp(-\tfrac12 \log\det \mathbf{C}_3)$. **Larger is better.**

The arm is then characterised by the **median over observations** of each quantity (robust
to single-realisation scatter). The marginals measure the constraint on each parameter
individually; the figure of merit measures the three-parameter space jointly.

**FoM₃ fragility — read it with the marginals.** Because the posterior is strongly
correlated ($\rho(\Omega_{\rm m},\sigma_8) \approx -0.93$), a few-per-cent change in the
marginals moves $\mathrm{FoM}_3$ by tens of per cent. This is intrinsic to the definition,
not a bug. The mitigation is discipline: hold **one** definition fixed across all arms, and
always quote it alongside the marginals.

**Runs as** — `scripts/slurm/sw_auto_nobnt_s41.slurm` is the minimal form:

```bash
python -u -m nn_l1_sbi.inference.population_sweep \
  --arm-dir   "$NN_L1_SBI_DATA/vmim/none_nobnt_s41" \
  --arm-label auto_nobnt_s41 \
  --out       "$NN_L1_SBI_DATA/sweeps/auto_nobnt_s41" \
  --seeds 41,42,43 --n-obs 9000 --max-perm 50 --m-samples 2000
```

and the ensemble form repeats `--arm-dir` three times (`…_s41 …_s42 …_s43`) with
`--arm-label auto_nobnt_ensemble`. Each run writes `median_summary.json` (the digest that
ships in `data/table1/`), `per_patch_metrics.npz` and `per_obs_samples.npy`.

**The one flag that differs between arm families.** `population_sweep`'s defaults are
`--preproc-transform none --clip-value 0.0 --min-feature-variance 1e-12`; the terse `sw_*`
scripts omit them and are therefore *not* under-specified — they are identical to what
`sw_joint_chain.slurm` and `conv_arms/conv_arm.slurm` pass explicitly. Only the **CNN**
rows deviate: they are swept with `zscore 5.0`, because the CNN driver standardises its
compressed summaries by default (`--standardize-summary`, `--summary-clip-value 5.0`) and
that is what produced the published 3326. Measured sensitivity on `cnn_nobnt_s41`:
3539 vs 3428, i.e. 3.2 %. `population_sweep`'s `preprocess_summaries` fits mean/std on the
arm's training `x` and clips — identical to the driver's
`fit_and_apply_summary_standardization`.

### 10.2 Error bars, and the thing that is not an error bar

Each quoted entry is a two-stage estimate: $\mathrm{FoM}_3(x)$ is a deterministic
functional of the *learned* posterior at mock observation $x$, and the quoted value is the
median of that functional over $n = 9000$ mocks. The estimate departs from its target
(the FoM the exact posterior would deliver under the same recipe) through **bias** —
the learned posterior is mis-calibrated — and **variance** — re-running the pipeline gives
a different number. The two are reported separately:

> **± = the standard deviation of the median $\mathrm{FoM}_3$ over three independently
> trained, *pre-ensemble* VMIM compressors (seeds 41/42/43).** It is quoted for *every*
> row, including rows whose central value is an ensemble estimate, so that the column is
> comparable across summaries and interpretable as **training stochasticity**.
>
> **The single → ensemble shift is the BIAS term.** It is reported separately and is
> **never summed with, or presented as, the ±.**

Explicitly, the ± is **not**:

- the population scatter across the 9000 observations (that is a different question,
  answered by the per-patch violin/distribution figure);
- a posterior width (the FoM already summarises one);
- the effect of NDE seeds or flow family (that is 0.3–1.5% on the medians, measured below,
  and family choice is a recipe decision, not noise);
- the single → ensemble shift over compressor seeds (that is the bias term, 0.4–22% across
  the rows of §13.4 — large, and reported separately).

**Error budget** (variance sources, largest first):

| source | independent across arms? | measured size |
|---|---|---|
| compressor training seed | **yes** (different networks) | **~0.4–11% depending on arm; the dominant term** |
| NDE training seed / flow family | yes | **0.3–1.5%** on medians (single → 3-seed pooled) |
| median sampling error over finite mocks | no (same mocks for all arms) | **0.20–0.42%** as measured (block bootstrap; `median_se` in `TABLE1_ERRORBARS_V2.json`) |
| training-set realisation (899 cosmologies) | no (shared) | not quantifiable without new simulations; largely cancels in comparisons |

**Median sampling error, done properly.** The 9000 mocks are $180 \times 50$, not 9000
independent draws: values sharing a patch are correlated through that patch's signal
realisation. The estimate is therefore a **block bootstrap**: resample the 180 patches with
replacement, keep all 50 noise realisations of each sampled patch, recompute the robust
median, $10^4$ replicates, report the 68% percentile interval. The measured intra-patch
correlation is $\rho \le 0.008$ on every row, so this term lands at the bottom of its
predicted range: $\pm1.1$ to $\pm8.4$ in $\mathrm{FoM}_3$ units, 0.20–0.42 % of the
central, sub-per-cent everywhere. Residual caveat,
stated rather than fixed: the 180 patches tile the *same* fiducial sky realisations, so
even patch-level resampling is mildly optimistic.

**The paired design.** All arms are evaluated on the same 9000 mocks, so population scatter
and training-set noise cancel in differences; a difference between two arms inherits their
independent per-arm seed noise, $\approx\sqrt{2}\times(5\text{–}7\%) \approx 7\text{–}10\%$.
Consequences: the 1.4% joint-$\ell_1$-vs-CNN difference is deep inside the noise — the error
bars formally back the word "tie"; the +24% product gain is $\gtrsim 3\sigma$ of difference
noise and is robust; the BNT retention ladder (0.16 → 0.96) is far outside any band.

**A measured caveat on the ± convention itself.** `conv_arms/nde_variance*.slurm` re-sweep
a *fixed* compressor with a different pooled flow-seed set (51/52/53) so the paired
difference isolates the NDE term. For the CNN that term is 0.74 % (no-BNT) and 1.42 %
(BNT) — 2–3× the compressor term — so the quoted CNN bar under-represents reproducibility.
For the $\ell_1$ arms the flow-set term is 0.3–1.5 % and never dominates. These jobs are a
diagnostic, not a Table-1 row.

---

## 11. Reproducing Table 1

Table 1 is 6 summaries × 2 frames = **12 arms**. Each arm needs four sweeps (three
compressor singles + one three-compressor ensemble), giving 48 directories; six more come
from the `extra_seeds` check, for the **54 `sweeps/*/median_summary.json`** that ship in
`data/table1/sweeps/`.

### 11.1 Submit order

```
setup/            environment, observation cache, noise sigma, obs datavectors, slicing
  |
  +-- raw datavectors     train_l1both_build · build_joint_arm · cnn_nobnt/cnn_bnt
        |
        +-- compressors   vmim_* · conv_arms/conv_arm · cnn_fidsum
              |
              +-- sweeps  sw_* · sw_joint_chain · sw_cnn · conv_arms/conv_arm (chained)
```

**Stage 0 — setup**, in dependency order:

| job | produces |
|---|---|
| `setup/build_env.slurm` (online) or `setup/install_env.slurm` (offline) | the pinned python environment |
| `setup/build_obs_cache.slurm` | `full_sphere_cache_fiducial_10deg/nobnt/obs` — 200 perms × 180 patches. Everything downstream reads it. |
| `setup/freeze_sigma.slurm` | `sigma_tables/flatsky_cross_noise_sigma.npz` (no-BNT). The BNT twin is an input (`$NN_L1_SBI_SIGMA_BNT`). |

**Stage 1 — raw data vectors:**

| job | produces |
|---|---|
| `train_l1both_build.slurm` | `l1_datavector_cache/flat_local_both` (16 ch) + `flat_local_ranges.npy`. The only expensive TFDS pass. |
| the same driver with `--flatsky-channel-mix bnt`, `_bnt` sigma, `--cache-dir …_bnt` | `l1_datavector_cache_bnt/flat_local_both_bnt` |
| `setup/precompute_obs.slurm` | `fiducial_both_datavectors_nobnt.npz` (16-ch observations) |
| `setup/bnt_obs_and_slice.slurm` | `fiducial_both_datavectors_bnt.npz`, then slices the BNT cache + obs → `product`, `none` |
| `setup/slice_raw.slurm` | no-BNT slices → `flat_local_{product,none}` + `fid_raw_{product,none}.npz` |
| `conv_arms/slice_conv.slurm` | `flat_local_conv*` + `fid_raw_conv*.npz`, **both frames**, in one CPU job |
| `build_joint_arm.slurm nobnt` / `bnt` | `joint/jointl1_{nobnt,bnt}_raw/{cache,fid.npz}` (3000-D) |
| `cnn_nobnt.slurm`, `cnn_bnt.slurm` | `cnn/{nobnt,bnt}_s4{1,2,3}` compressors (no separate raw datavector — the CNN reads maps) |
| `cnn_fidsum.slurm` | `cnn/*/fiducial_summaries.npz` (+ G1 gate) |

**Stage 2 — compressors:** `vmim_none_{nobnt,bnt}.slurm`, `vmim_product_{nobnt,bnt}.slurm`,
`vmim_joint.slurm <frame>`, and `conv_arms/conv_arm.slurm <frame>` (which chains its own
compressors and sweeps). Each trains seeds 41/42/43.

**Stage 3 — sweeps, and the 48 directories they write:**

| job (invocation) | sweep dirs written | Table-1 arm |
|---|---|---|
| `sw_auto_nobnt_{s41,s42,s43,ensemble}.slurm` | `auto_nobnt_{s41,s42,s43,ensemble}` | ℓ1 auto, no-BNT |
| `sw_auto_bnt_{s41,s42,s43,ensemble}.slurm` | `auto_bnt_{…}` | ℓ1 auto, BNT |
| `sw_product_nobnt_{s41,s42,s43,ensemble}.slurm` | `product_nobnt_{…}` | ℓ1 +product, no-BNT |
| `sw_product_bnt_{s41,s42,s43,ensemble}.slurm` | `product_bnt_{…}` | ℓ1 +product, BNT |
| `conv_arms/conv_arm.slurm nobnt` | `conv_nobnt_{s41,s42,s43,ensemble}` | ℓ1 +conv, no-BNT |
| `conv_arms/conv_arm.slurm bnt` | `conv_bnt_{…}` | ℓ1 +conv, BNT |
| `sw_joint_chain.slurm nobnt` | `jointl1_nobnt_{s41,s42,s43,ensemble}` | joint ℓ1, no-BNT |
| `sw_joint_chain.slurm bnt` | `jointl1_bnt_{…}` | joint ℓ1, BNT |
| `sw_cnn.slurm cnn_nobnt_s41_pnone zscore 5.0 nobnt_s41` (and s42, s43, and `cnn_nobnt_ensemble_pnone … nobnt_s41 nobnt_s42 nobnt_s43`) | `cnn_nobnt_{s41,s42,s43,ensemble}_pnone` | CNN, no-BNT |
| the same four invocations with `bnt_s4x` | `cnn_bnt_{…}_pnone` | CNN, BNT |
| **not in this tree** (see below) | `joint_l1_nobnt_{s41,s42,s43,ensemble}` | ℓ1 +conv+product, no-BNT |
| **not in this tree** | `joint_bnt_{…}` | ℓ1 +conv+product, BNT |

That is 12 × 4 = **48**. `sw_joint_chain` and `conv_arm` run their four sweeps sequentially
in a single job — `qos_gpu_a100-dev` allows at most 10 submitted jobs per user, and each
sweep is only a few minutes, so chaining is cheaper than fanning out. Both are idempotent:
a stage whose `median_summary.json` already exists is skipped.

**Stage 4 — the six extra-seed directories.**
`extra_seeds.slurm jointl1_bnt` and `extra_seeds.slurm auto_bnt` train compressor seeds
44/45/46 for the two rows with double-digit spread and sweep them, writing
`jointl1_bnt_s4{4,5,6}` and `auto_bnt_s4{4,5,6}` — 48 + 6 = **54**. Going from $n=3$ to
$n=6$ drops the relative uncertainty on the ± from ~46 % to ~30 %. These are *not*
Table-1 centrals; they feed only the `n6_*` fields of `TABLE1_PAPER.json`.

**Three naming traps, all of which produce a plausible wrong number:**

1. `joint_l1_*` / `joint_bnt_*` (with underscore) are the **ℓ1 +conv+product** arm;
   `jointl1_*` (no underscore) is the **joint ℓ1** arm. They are different statistics with
   different numbers. The underscore names are a historical mislabelling from an early
   recovery — `--cross-op both` really is autos + conv + product, only the label was wrong.
   **Always resolve a directory through `data/table1/arms.json`; never string-munge a label.**
2. The ℓ1 +conv+product compressor and sweep jobs are not in `scripts/slurm/`
   (`scripts/slurm/README.md` §6). The recipe is identical to the other ℓ1 arms: VMIM on
   `l1_datavector_cache*/flat_local_both*` with `fid_raw_both*.npz`, then `population_sweep`
   with `none / 0 / 1e-12`.
3. The `_pnone` suffix on the CNN sweep directories is part of the label string passed as
   `sw_cnn.slurm`'s first argument. It does **not** describe the preprocessing those rows
   were swept with, which is `zscore 5.0` (§10.1, `CAMPAIGN_REFERENCE.md` §3). The label must
   match `arms.json` exactly.

### 11.2 From the 54 directories to the printed table

Everything needed is committed under `data/table1/` (the 54 sweep summaries plus seven
digest files and a README — under 100 kB in total): `arms.json` maps
each of the 12 arm labels to its three singles, its ensemble, its `published` central and
which estimator the paper `quoted`; the 54 `median_summary.json` carry the measurements.
No download, no GPU, under a second on a laptop. This is section A of
`notebooks/reproduce_paper.ipynb`.

The recipe, verified end to end against every cross-check file in that directory:

```python
import json, os
import numpy as np

ROOT = "data/table1"
load = lambda rel: json.load(open(os.path.join(ROOT, rel)))
arms = load("arms.json")

for label, a in arms.items():
    singles = [load(p) for p in a["singles"].values()]        # order: s41, s42, s43
    ens     =  load(a["ensemble"])
    src     = singles[0] if a["quoted"] == "single" else ens  # s41 is index 0
    per_seed = [s["fom3"] for s in singles]
    rel     = np.std(per_seed, ddof=1) / np.mean(per_seed)    # THE error bar, relative
    central = a["published"] if a["published"] is not None else src["fom3"]
    pm      = central * rel
    sig     = (src["sigma_Om"], src["sigma_s8"], src["sigma_w0"])
    sig_rel = [np.std([s[k] for s in singles], ddof=1) / np.mean([s[k] for s in singles])
               for k in ("sigma_Om", "sigma_s8", "sigma_w0")]
    print(f"{label:26s} {central:7.0f} +- {pm:5.0f}")
```

Four rules make this the right recipe rather than one of several plausible ones
(`docs/NOTE_FOM_ERROR_BARS.md` §5–6):

1. The ± is **always** the std over the three **pre-ensemble** compressor singles, for
   every row — including rows whose central is an ensemble. It is carried as a *relative*
   quantity `std/mean` and multiplied by whichever central the row quotes, so it rescales
   correctly onto the published central.
2. The single → ensemble shift is a **bias**, reported separately, never summed into ±.
   (`bias_pct` in `TABLE1_ERRORBARS_V2.json` uses `per_seed.s41` as its baseline, not the
   mean of the three.)
3. `median_se` — block bootstrap over the 180 patches, all 50 reps kept, $10^4$ replicates,
   68 % percentile — is a **caption-only** subdominant term. It is precomputed in
   `TABLE1_ERRORBARS_V2.json`; recomputing it needs `per_patch_metrics.npz`, which is not
   in the repo.
4. The leave-one-out ensemble jackknife in `final_bars_all_v2.json` (`ensemble_term`) is a
   diagnostic and is **never** the bar.

Two rows (`l1 +conv, BNT` and `l1 +conv+product, BNT`) have `published == null`: they are
new results with no previously published central, and the retrained ensemble is quoted.
The retrained centrals for all 12 rows are tabulated in `data/table1/README.md` §7 — note
that they differ from the published values by up to +13 %, and that **the published
centrals are what the table prints**; the retrained run supplies only the *relative*
spread (§16).

**What is deliberately not in the repo.** `per_patch_metrics.npz` (153 MB across the 54
dirs) and `per_obs_samples.npy` (216 MB) live on Zenodo. They are needed only to recompute
the block bootstrap or to draw the per-patch FoM₃ violins; Table 1 is complete without
them. Verified: `fom3` in `median_summary.json` equals `np.median(per_patch_metrics['fom3'])`
to full float precision, and likewise for the three `sigma_*`.

---

## 12. What you need to run each stage

Walltimes are the `#SBATCH --time` requests from `scripts/slurm/` with the measured
durations beside them where they are known. The hardware column is the Jean-Zay
configuration the published numbers were produced under: one A100 (`gpu_p5`, quarter node,
32 cores) or one `cpu_p1` node. Nothing here needs more than a single GPU.

| stage | job | hardware | walltime (requested / measured) | main input |
|---|---|---|---|---|
| environment | `setup/build_env.slurm` | `prepost` (needs internet), 8 cores | 3 h | — |
| environment, offline | `setup/install_env.slurm` | **CPU** (`cpu_p1`), 10 cores | 2 h | ~12 GB wheelhouse (disposable) |
| observation cache | `setup/build_obs_cache.slurm` | **CPU**, 40 cores | **10 h** | CosmoGridV1, read-only; writes 200 perm files of ~34 MB |
| frozen σ table | `setup/freeze_sigma.slurm` | **CPU**, 32 cores | 4 h / ~2 h | the observation cache |
| ℓ1 `both` datavector | `train_l1both_build.slurm` | A100, 8 cores | 2 h | **the ~290 GB TFDS** — the only job that streams it fully |
| observation datavectors | `setup/precompute_obs.slurm` | A100, 16 cores | 2 h / ~20 min | fiducial cache only, *not* the TFDS |
| BNT obs + slice | `setup/bnt_obs_and_slice.slurm` | A100, 32 cores | 2 h / ~30 min | BNT `both` cache |
| raw slicing | `setup/slice_raw.slurm`, `conv_arms/slice_conv.slurm` | **CPU**, 8 cores | 1 h / ~5 min | the 16-ch caches (the 800-D `flat_local_none` train file, `l1_train.npz`, is 1.04 GB measured) |
| joint-ℓ1 datavector | `build_joint_arm.slurm <frame>` | A100, 32 cores | 2 h | TFDS + fiducial cache; writes a 3000-D cache and a `fid.npz` of ~430 MB |
| CNN compressors ×3 | `cnn_{nobnt,bnt}.slurm` | A100, 32 cores | 2 h / ~1 h (measured ≈70 steps/s ⇒ ~19 min train + ~2 min compress **per seed**) | TFDS |
| CNN obs summaries | `cnn_fidsum.slurm` | A100, 16 cores | 1 h | fiducial cache + CNN checkpoints |
| VMIM compressors ×3 | `vmim_*.slurm` | A100, 32 cores | 2 h (`--max-minutes 35` per seed) | one ℓ1 cache (1.04 GB `l1_train.npz` for the 800-D arm) |
| conv arm, end to end | `conv_arms/conv_arm.slurm <frame>` | A100, 32 cores | 2 h / ~70 min | 3 compressors + 4 sweeps, chained |
| population sweep, single | `sw_*_s4x.slurm` | A100, 32 cores | 2 h / **~4 min** (227–281 s measured) | one 10-D arm dir (~53 MB) |
| population sweep, ensemble | `sw_*_ensemble.slurm` | A100, 32 cores | 2 h / **~11 min** (656–699 s measured) | three 10-D arm dirs |
| joint sweeps, chained | `sw_joint_chain.slurm <frame>` | A100, 32 cores | 2 h | four sweeps in one job |
| extra seeds 44/45/46 | `extra_seeds.slurm <row>` | A100, 32 cores | 2 h | one raw cache |
| **print Table 1** | `notebooks/reproduce_paper.ipynb` §A | **laptop CPU** | **< 1 s** | `data/table1/`, under 100 kB, in the repo |

Three environment facts that cost real time if missed:

* **`TF_FORCE_GPU_ALLOW_GROWTH=true`** wherever torch and `tf.data` share a GPU.
  `build_flatsky_joint_arm.py` computes in torch but loads through `tf.data` and never
  calls `tf.config.set_memory_growth`; without the variable TensorFlow takes the whole
  A100 and torch OOMs with under 1 GB of its own allocations. Set in `build_joint_arm`,
  `vmim_joint`, `sw_joint_chain`, `conv_arm`, `extra_seeds`, `nde_variance*`.
* **`TMPDIR`** must point at bulk storage; a compute node's `/tmp` is too small for the
  torch source distributions during an environment build.
* Three package pins are load-bearing: `wl_stats_torch` must come from git at commit
  `77eafc84` (the PyPI 0.1.0 carries the same version string but different code — it lacks
  `subtract_coarse_mean`, the wavelet transform silently takes another path and the ℓ1 data
  vectors change); `sbi_lens` needs the two-line tfp-substrate patch in `patches/`;
  `tensorflow-metadata` must be 1.16.1 with protobuf 5.29.3 or tfds breaks under TF 2.18.
  See `pyproject.toml`, `constraints.txt` and `scripts/slurm/inputs/README.md`.

---

## 13. Results the pipeline produces

All entries are medians over the 9000 mock observations, through the matched RealNVP
pipeline. Tighter is better (smaller $\sigma$, larger $\mathrm{FoM}_3$).

### 13.1 Headline comparison (no-BNT frame)

| Summary | $\sigma(\Omega_{\rm m})$ | $\sigma(\sigma_8)$ | $\sigma(w_0)$ | $\mathrm{FoM}_3$ |
|---|---|---|---|---|
| $\ell_1$ auto-only | 0.053 | 0.085 | 0.245 | 2448 |
| $\ell_1$ +product | 0.048 | 0.077 | 0.229 | 3045 |
| joint $\ell_1$ | 0.044 | 0.072 | 0.223 | 3371 |
| CNN | 0.045 | 0.072 | 0.231 | 3326 |

> **These are the *published* numbers, and they are not what the notebook prints.**
> `notebooks/reproduce_paper.ipynb` §A reads the $\sigma$ triplet straight out of the
> committed **retrained** sweep (`data/table1/sweeps/*/median_summary.json`), while keeping
> the published FoM₃ as the central (§11.2). The retrained triplets for these four rows are
> $\ell_1$ auto 0.0503 / 0.0799 / 0.2340, $\ell_1$ +product 0.0462 / 0.0746 / 0.2271,
> joint $\ell_1$ 0.0440 / 0.0716 / 0.2234, CNN 0.0444 / 0.0702 / 0.2288. The joint-$\ell_1$
> row agrees to the three decimals printed above; the CNN row differs by ≤2.5 %, and the
> auto and +product rows by up to 6 % (their retrained posteriors are tighter, consistent
> with their higher retrained FoM₃, §11.2). Quote one set or the other, never a mixture.

### 13.2 The completeness ladder — inter-bin information added in stages

| arm | $\mathrm{FoM}_3$ | note |
|---|---|---|
| $\ell_1$ auto-only | 2448 | sees no inter-bin structure |
| $\ell_1$ +conv | 2671 | +9%; convolution carries mostly two-point cross information |
| $\ell_1$ +product | 3045 | **+24%**; the bulk of the gain |
| $\ell_1$ +conv+product | 3255 | both operators |
| **joint $\ell_1$** | **3371** | complete pairwise statistic, no explicit cross-map |
| CNN | 3326 | the learned reference |

The learned compressor gains nothing systematic from the explicit cross operators, because
it already reads the four bins jointly — the same property that makes it insensitive to
BNT.

### 13.3 BNT retention

| Summary | no BNT | BNT | retention |
|---|---|---|---|
| $\ell_1$ auto-only | 2448 | 388 | **0.16** |
| $\ell_1$ +product | 3045 | 718 | **0.24** |
| joint $\ell_1$ | 3371 | 2424 | **0.72** |
| CNN | 3326 | 3186 | **0.96** |

Reading: **what a summary retains under the nulling tracks how jointly it reads the bins.**
A per-channel statistic sees only 1-D marginals of the across-channel distribution and
keeps 0.16; the products (a single derived field per pair) lift this to 0.24; reading the
pairwise distributions in full keeps 0.72; the learned compressor, which mixes all four
channels before any non-linearity, is near-lossless at 0.96. The residual
$0.72 \to 0.96$ is the genuinely three- and four-bin structure a pairwise statistic cannot
reach.

Underlying it is a closure property: a fixed summary is BNT-invariant exactly when the
transform's action on it can be inverted. The full auto- and cross-second-moment vector
qualifies (BNT acts on the pixel covariance as the invertible congruence
$\hat{\mathbf{C}} \mapsto B\hat{\mathbf{C}}B^{\rm T}$); the per-channel $\ell_1$-norm does
not, because the transform turns it into a mix of channels followed by a single-channel
marginal, and **marginalising after mixing cannot be undone**.

### 13.4 Error bars and the bias term

$\pm$ = std over three independently trained, pre-ensemble compressors (§10.2). These were
computed in a dedicated retraining sweep and are **not yet in the manuscript table** (see
§16).

| Summary | frame | $\mathrm{FoM}_3$ | relative ± | bias (single→ensemble) |
|---|---|---|---|---|
| $\ell_1$ auto | no-BNT | 2448 ± 27 | 1.1% | −9.6% |
| $\ell_1$ auto | BNT | 388 ± 43 | 11.1% | +0.4% |
| $\ell_1$ +conv | no-BNT | 2671 ± 133 | 5.0% | −9.5% |
| $\ell_1$ +conv | BNT | 458 ± 15 | 3.3% | −7.8% |
| $\ell_1$ +product | no-BNT | 3045 ± 183 | 6.0% | −5.5% |
| $\ell_1$ +product | BNT | 718 ± 29 | 4.1% | −16.9% |
| $\ell_1$ +conv+product | no-BNT | 3255 ± 200 | 6.1% | −12.2% |
| $\ell_1$ +conv+product | BNT | 704 ± 76 | 10.8% | −7.6% |
| joint $\ell_1$ | no-BNT | 3371 ± 96 | 2.8% | −10.2% |
| joint $\ell_1$ | BNT | 2424 ± 208 | 8.6% | −21.8% |
| CNN | no-BNT | 3326 ± 14 | 0.4% | −3.0% |
| CNN | BNT | 3186 ± 19 | 0.6% | −1.2% |

Block-bootstrap standard error of the median (subdominant, caption-level): between $\pm1.1$
and $\pm8.4$ in $\mathrm{FoM}_3$ units across the twelve rows — 0.20 % to 0.42 % of the
central. (`median_se` in `data/table1/TABLE1_ERRORBARS_V2.json`; the smallest bars belong to
the low-FoM₃ BNT rows, the largest to the CNN.)

The **bias** column above is `bias_pct` from the same file: the shipped retrained ensemble
against that row's seed-41 single. It is the measurement that supersedes the pre-recovery
claim of a ≤1.2 % ensembling shift for the no-BNT $\ell_1$ arms (§9.3).

**Caveat measured with extra seeds.** Six compressor seeds were run for the two BNT rows
with the widest $n=3$ spread. $\ell_1$ auto BNT *tightened* (11.1% → 8.8%), but joint
$\ell_1$ BNT **widened with every added seed** (8.6% → 10.9% → 13.4%; range 2551–3712,
best/worst = 1.46×). For that row the $n=3$ bar understates the true
compressor-to-compressor variability by roughly a third. The $n=3$ values are retained in
the table for cross-row comparability.

---

## 14. Validation gates: TARP and SBC

Because a learned pipeline can produce uncalibrated posteriors, **every** posterior — in
both frames, for every arm — is subjected to the same coverage tests before its constraints
are quoted. Passing is a precondition for reporting constraining power.

**TARP — tests of accuracy with random points** (Lemos et al. 2023). Measures, at each
credibility level, the fraction of true parameters falling inside the corresponding
credible region, over parameters drawn from the prior. A calibrated posterior lies on the
identity line. The reported scalar is the **integrated net coverage**: positive =
conservative (safe), negative = over-confident (the failure mode).

**SBC — simulation-based calibration** (Talts et al. 2020). The rank of the true parameter
among the posterior samples is uniform under calibration. A $\cup$-shaped rank histogram
signals over-confidence, an $\cap$ shape under-confidence, a slope a biased mean. The
scalar is the **rank standard deviation**, whose uniform value is $0.289$.

**Protocol.** Both tests use $N = 600$ validation points drawn from the prior, which
resolve coverage deviations down to a binomial band of $\sim 0.02$
($\sqrt{0.25/N} \approx 0.02$); all calibration statements hold at that resolution. The
uncertainty band on the TARP curve is a bootstrap over the 600 validation sightlines (not
the much smaller scatter of the random reference points). For SBC, the three retraining
seeds share identical test parameters, so the independent unit is **one seed, not the
pooled 1800**: bars are the mean over the three seeds and the band is the central 99% of
the binomial scatter for $N = 600$.

**Gate thresholds used in production:**

| verdict | condition |
|---|---|
| **PASS** | worst-tercile $\lvert \mathrm{ECP} - \alpha\rvert \le 0.05$ **and** SBC rank std $\in [0.275, 0.305]$ |
| **PASS with caveat** | worst $\lvert\text{dev}\rvert \le 0.10$ (and rank-std band held) — reportable, flagged |
| **FAIL** | worst $\lvert\text{dev}\rvert > 0.10$ **or** rank std outside the band — the tightness is over-confidence; quote the gated (ensemble) read instead |

**Measured outcomes.** Integrated net coverage in the standard frame: $+0.018$, $+0.004$,
$+0.003$, $+0.032$ for $\ell_1$ auto-only, $\ell_1$+product, joint $\ell_1$ and the CNN;
in the BNT frame $+0.001$, $+0.008$, $+0.005$, $+0.034$. The analytical arms are consistent
with ideal coverage to better than $0.5\sigma$ in both frames; the CNN is mildly
**conservative** ($\sim 1.4\sigma$, not significant, and on the safe side) — which is the
expected explanation for the small residual joint-$\ell_1$-vs-CNN difference. SBC rank
standard deviations in the standard frame are $0.305$, $0.302$, $0.298$, $0.290$ against
the uniform $0.289$: a mild over-confidence at the few-per-cent level, within the band in
all three parameters and both frames.

*(A local classifier two-sample test, L-C2ST, is not used: on the high-dimensional $\ell_1$
data vector it loses statistical power and its own null calibration fails, so it would not
be applicable uniformly across arms.)*

### 14.1 The gate in code, and its current status

Three modules implement the battery:

| module | role |
|---|---|
| `src/nn_l1_sbi/inference/tarp_stratified_val.py` | For one arm: retrain the NDE (3 seeds), sample $N=600$ held-out validation points drawn from the prior, compute the per-point pooled FoM₃, split into LOW/MID/HIGH FoM₃ **terciles**, and dump `posterior_samples.npz` (`samples`, `theta`) per (tercile, seed). The stratification is the point: it answers whether the *tight* posteriors specifically are calibrated, rather than averaging their behaviour away. |
| `src/nn_l1_sbi/inference/run_tarp_coverage.py` | Reads the dumps tree and produces DRP coverage curves per (arm, seed, tercile) in the 3-D science subspace and in the full 6-D space, using the `tarp` package's own bootstrap (`--dims 3 6`, `--num-bootstrap 200`). |
| `src/nn_l1_sbi/inference/run_joint_gate_c.py` | Orchestrator: preflight on each arm's cache metadata (asserted without decompressing the big arrays), then the two phases above, then a report whose PASS / PASS-with-caveat / FAIL verdicts are **derived** from the registered bands (`DEV_PASS=0.05`, `DEV_CAVEAT=0.10`, `STD_LO/HI=0.275/0.305`). SBC is computed inside it, by `sbc_from_dumps`: the rank of the truth among the posterior samples, pooled over terciles and seeds, reported as per-parameter mean, std, and a minimum KS p-value against uniformity. |

The NDE retrains inside the gate mirror the full-rigor sweeps: `log1p-zscore` / clip 5 /
min-var 1e-5 / seeds 41,42,43 / 50 000 epochs / batch 256 / lr 1e-4, 600 validation points,
$m=2000$ samples each.

**Status in this release — read before running the gate.**

* `tarp_stratified_val.py` and `run_tarp_coverage.py` are fully CLI-driven and portable.
  `tarp_stratified_val.py` takes `--train-cache-dir`, `--cache-prefix {cnn,l1}`,
  `--arm-label` and `--dumps-root` (all required), plus the preprocessing and training
  flags; `run_tarp_coverage.py` takes `--dumps-root`, `--outdir`, `--dims` and
  `--num-bootstrap`. Both can be run directly against any arm directory produced by §8.
  `run_tarp_coverage.py` needs the `tarp` package, which is in the `notebook` extra of
  `pyproject.toml` and was **not** in the paper environment.
* `run_joint_gate_c.py` **resolves its paths like every other module in the release**: the
  arm caches it reads are `data_root()/analysis/flatsky_cross_2026_06/overnight_menu/<arm>/cache`
  and everything it writes (dumps, TARP curves, logs, `GATE_C_JOINT.md`) goes under
  `output_root()/flatsky_cross_2026_06/overnight_menu/gate_c` — so `$NN_L1_SBI_DATA` and
  `$NN_L1_SBI_OUT` move it wherever you need (`src/nn_l1_sbi/paths.py`). The child phases
  run as `python -m nn_l1_sbi.inference.{tarp_stratified_val,run_tarp_coverage}` under
  `sys.executable`, i.e. the interpreter that launched the orchestrator; no path or
  interpreter is baked in.
  **The one real caveat is its arm list.** `ARMS` is a module-level constant naming the four
  arms of the overnight joint-statistic campaign the gate was written for — `pair2dq_nobnt`,
  `jointl1q_nobnt`, `pair2dq_bnt`, `jointl1q_bnt`, i.e. the joint $\ell_1$ and the
  *rejected* unweighted `pair2d` count variant (§7), in both frames — **not** the twelve
  Table-1 arms, and the matching preflight asserts a 3000-D data vector with
  `stat/basis/k=10/snr_range=5.0`, so nothing else will pass it. Those `overnight_menu`
  caches are also not part of the released data bundle (`data/ZENODO_MANIFEST.md`). To gate
  a different arm, edit `ARMS` (and the preflight's dimension assertion) or drive
  `tarp_stratified_val.py` / `run_tarp_coverage.py` directly, which is the portable route.
* **No gate output ships with the repository.** The TARP band arrays
  (`tarp_proper_bands{,_bnt}.npz`) and the SBC dumps
  (`tarp_drp/dumps/**/posterior_samples.npz`) that `figures/fig_tarp.py` and
  `figures/fig_sbc.py` read are not among the recovered products. The numbers quoted above
  are the manuscript's; regenerating the two calibration figures requires re-running the
  gate. Treat that as the outstanding reproduction task for §14, not as a settled artefact.
* `ensemble_eval.py` has the same shape of limitation, and the same fix. Its root is
  resolved, not hardcoded — `$CNN_SBI_ARMS`, defaulting to
  `data_root()/analysis/flatsky_cross_2026_06/analytical_nde_match` — but the module-level
  `ARMS` dict still spells out the campaign's own directory *names*: six modes
  (`nobnt`, `bnt`, `product_bnt`, `auto_bnt`, `product_nobnt`, `auto_nobnt`) × three
  compressor seeds = 18 paths, selected by `--mode`. Point `$CNN_SBI_ARMS` at a tree with
  those names, or edit the dict. It is in any case not on the critical path: the ensemble
  rows of Table 1 are produced by `population_sweep` with three `--arm-dir` flags.

---

## 15. Reproduction checklist and known failure modes

Ordered roughly by how often they have actually bitten this pipeline.

1. **Scenario mismatch.** Geometry, TFDS config and tomographic bin selection must agree
   between the observed datum, the training set and the validation set. Assert it.
2. **Stale cache reuse.** Every cache carries metadata (SNR range, number of bins, scales,
   clamp behaviour, coarse-mean handling, TFDS name, bin indices, compressor checkpoint
   SHA256, zero-mean flag). Reuse only on an exact match — "wrong-but-stable" posteriors
   are the classic symptom.
3. **Preprocessing / checkpoint mixing.** A flow checkpoint loaded under a different
   summary transform is silently wrong. The stored preprocessing (and PCA/feature mask, if
   any) is authoritative and overrides conflicting CLI values in evaluation mode.
4. **Forgetting the mean subtraction**, or reusing a compressor checkpoint across a
   demeaning flip. Verify `map.mean(axis=(0,1)) < 1e-6` per channel on the first batch.
5. **A single shared SNR threshold across auto and cross channels** — empties the
   cross-channel histograms and throws away exactly the information the cross-maps exist to
   supply. Use the per-channel frozen noise table.
6. **PCA on the $\ell_1$ data vector.** Never. Use VMIM.
7. **Batch normalisation on the many-channel input.** Batches mix cosmologies; BN averages
   the signal away.
8. **Shape noise applied after BNT**, or BNT requested with fewer than four bins.
9. **Compressor/NDE training-set overlap** wherever a trainable compression stage exists.
10. **Under-converged flow**, or a best-validation checkpoint that sits at the final step;
    and last-step rather than best-validation compressor checkpoints.
11. **Quoting $\mathrm{FoM}_3$ alone.** Always with the marginals, always with the
    fragility caveat.
12. **Quoting an ungated number.** A tighter-but-miscalibrated posterior is fool's gold;
    the calibrated value is lower.

Six gates are built into the job scripts precisely so that these stop the pipeline instead
of producing a plausible wrong number. Keep them: the joint-arm dimension/finiteness/
non-negativity assertions (`build_joint_arm.slurm`); the observation-cache shape, polar-leak
and pinned-centres check (`setup/build_obs_cache.slurm`); the nine-field σ-table twin check
(`setup/freeze_sigma.slurm`); the byte-identical auto-block assertion
(`conv_arms/slice_conv.slurm`); the CNN G1 tolerance plus checkpoint SHA-256 re-hash
(`cnn_fidsum.slurm`); and the campaign gate `train_l1product_s41.slurm`, which retrains
ℓ1+product no-BNT seed 41 and compares its population median FoM₃ against the published
3045 — landing within ~10–15 % validates the whole train → eval → FoM → bootstrap chain.

---

## 16. Open questions / items not fixed by the sources

Listed so that nothing here has to be guessed by a reader:

- **Prior ranges.** The Sobol sampling ranges of the six CosmoGridV1 grid parameters are
  not stated in the manuscript sources; only the fiducial point and the sampling scheme
  are. Fill in from the CosmoGridV1 release before publication.
- **Redshift distributions.** The maps are read from `kg/stage3_lensing{1..4}` and the
  paper quotes mean redshifts $z\approx0.3,0.5,0.7,0.9$, but the per-bin $n(z)$ itself is
  not given in the sources, and the "stage3" map label sits alongside a *Euclid*-like noise
  configuration. Worth an explicit sentence in a public release.
- **Source density: the manuscript and the code disagree, and the code wins.** The
  manuscript describes 30 arcmin⁻² split over four bins (7.5 per bin); the maps were
  actually built with the pinned builder default of **10 arcmin⁻² per bin, 40 total**
  (§2 — verified in `build_full_sphere_cross_cache.py` and in the absence of any override
  in `setup/build_obs_cache.slurm`). Nothing downstream depends on which number is
  *written*, since the noise realised in the maps is the same either way, but the
  manuscript sentence must be corrected before publication: as written, a reader
  recomputing $\sigma_{\rm pix}$ gets 0.0138 instead of 0.0120, ~15 % too much shape noise.
- **"50 independent noise realisations".** In the code these are 50 distinct CosmoGridV1
  realisation indices (`perm`) of the fiducial cosmology, each carrying its own
  deterministic shape-noise draw. Whether the underlying light-cone realisation is also
  different between `perm`s (rather than only the noise) is not settled by the sources; the
  manuscript's phrasing describes them as noise realisations.
- **Joint $\ell_1$ dimension.** From the code the joint $\ell_1$ data vector is
  $6\ \text{pairs}\times5\ \text{scales}\times K^2{=}100 = \mathbf{3000}$ features with the
  production $K=10$. The figure **3200** that circulates in the project notes is the
  dimension of the *$\ell_1$ +conv+product* (16-channel) data vector,
  $16\times5\times40$ — the two should not be conflated. One planning document does label
  "$\ell_1$+product" as 3200-d, which contradicts its own channel count
  ($10\times5\times40 = 2000$); treat that as a slip.
- **Joint $\ell_1$ grid ranges.** The code supports both a fixed symmetric $[-5,5]$ SNR
  grid and signal-adapted percentile ranges (and a rotated/whitened variant that was tested
  and rejected). Which of the first two produced the published joint-$\ell_1$ numbers is
  not stated in the sources consulted. `build_joint_arm.slurm` runs `--adaptive-ranges`,
  matching the surviving cache metadata.
- **Table 1 error bars are not yet in the manuscript.** `05-results.tex` carries a
  `% TODO(cluster, G-03)` for the $\mathrm{FoM}_3$ column, and the values in §13.4 above
  come from the dedicated retraining sweep. They are the intended content of that column.
- **The retraining sweep central values differ from the published ones** by up to +13%
  (e.g. $\ell_1$ auto no-BNT 2448 → 2777). The published centrals are retained everywhere;
  the retrained run supplies only the *relative* spread. The origin of the shift is not
  documented in the sources.
- **BNT-with-noiseless-maps run.** The claim that BNT and no-BNT contours coincide for
  noiseless maps is stated in the manuscript with a `% TODO(cluster)` against locating the
  run artefact; the provenance is not currently recoverable from these sources.
- **`optuna` hyper-parameter study artefacts** (referenced in the architecture-search
  appendix) were run on a different cluster and are likewise flagged as missing provenance.
- **Number of posterior samples in the published corner figures** vs. the sweep
  ($M=2000$ per (obs, seed), pooled over 3 seeds) is not stated uniformly in the sources.

---

## 17. Provenance

**Manuscript sections.** `main.tex`, `sections/02-simulations.tex`,
`sections/03-statistics.tex`, `sections/04-inference.tex`, `sections/05-results.tex`,
`sections/06-bnt.tex`, `sections/07-discussion.tex`, `sections/A0-calibration.tex`,
`sections/A1-appendix.tex`, `sections/A2-appendix.tex`, `sections/A4-fisher.tex`;
error-bar specification `docs/NOTE_FOM_ERROR_BARS.md` (shipped in this repository).

**Production code** — the release module for each stage:

| stage | module |
|---|---|
| full-sphere maps, noise, BNT in $a_{\ell m}$, patch extraction | `src/nn_l1_sbi/data/build_full_sphere_cross_cache.py` |
| TFDS builder (`grid_10deg_80px_nonoverlap180`) + loader | `src/nn_l1_sbi/data/tf_dataset_nbody_tomo{,_cross}.py`, `tfds_cross_tfdata_loader.py` |
| patch-local channel operators (conv / product / BNT) | `src/nn_l1_sbi/flatsky/flatsky_cross.py`, `flatsky/bnt_utils.py` |
| frozen per-(channel, scale) noise table | `src/nn_l1_sbi/flatsky/freeze_flatsky_cross_noise.py` |
| starlet $\ell_1$ under frozen $\sigma$, per-channel SNR ranges | `src/nn_l1_sbi/flatsky/flatsky_cross_l1.py` |
| joint $\ell_1$ / pair2d / covariance reductions | `src/nn_l1_sbi/flatsky/flatsky_joint_stats.py`, `src/nn_l1_sbi/data/build_flatsky_joint_arm.py` |
| observation data vectors | `src/nn_l1_sbi/data/precompute_fiducial_both_datavectors.py` |
| $\ell_1$ → 10-D VMIM compression | `src/nn_l1_sbi/compression/vmim_from_cache.py` (companion `vmim_maf_companion.py`) |
| CNN-VMIM compressor + RealNVP NPE driver | `src/nn_l1_sbi/cnn/npe_cnn_nbody_tomo.py`, `cnn/build_fiducial_summaries_cnn.py` |
| $\ell_1$ cross-arm NPE driver | `src/nn_l1_sbi/inference/npe_l1norm_cross_jaxili_nbody_tomo.py` |
| RealNVP training / FoM helpers | `src/nn_l1_sbi/compression/train_nde_from_compressed.py`, `src/nn_l1_sbi/cnn/train_jaxili_from_compressed.py` |
| 9000-obs population sweep, per-obs metrics | `src/nn_l1_sbi/inference/population_sweep.py` |
| compressor deep ensemble | `src/nn_l1_sbi/inference/ensemble_eval.py` |
| TARP / SBC gates | `src/nn_l1_sbi/inference/tarp_stratified_val.py`, `run_tarp_coverage.py`, `run_joint_gate_c.py` |
| figures | `src/nn_l1_sbi/figures/fig_*.py`, shared style in `figures/style.py`, data loading in `figures/_data.py`, corner dumps via `figures/dump_contour_samples.py` + `figures/gen_auto_contour.sh` |

**Result records shipped with the repository.** `data/table1/` — `arms.json`, the 54
`sweeps/*/median_summary.json`, `TABLE1_ERRORBARS_V2.json`, `TABLE1_CONV_ARMS.json`,
`TABLE1_PAPER.json`, `TABLE1_MARGINALS_NEW.json`, `final_bars_all_v2.json`,
`rows_table1_v2.json`, and `README.md`. The five
`TABLE1_*` / `final_bars_*` files and all 54 `median_summary.json` are byte-for-byte copies
of the originals; only `arms.json` was modified, to rewrite its paths as relative.
