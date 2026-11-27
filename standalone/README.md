# Joint wavelet ℓ1-norm, standalone

The joint ℓ1-norm of the paper as two self-contained files, for use outside this
repository: `joint_l1.py` for flat-sky maps (NumPy only) and `joint_l1_sphere.py` for
HEALPix maps (NumPy and healpy). Copy them next to your code; nothing else from the
repository is needed.

## What it computes

Each tomographic convergence map κ_i is decomposed with the starlet transform. At scale s
the coefficient of map i at pixel p, in signal-to-noise units, is
u_i(p) = (W_s κ_i)(p) / σ_{i,s}, with σ_{i,s} the noise level of that bin and scale. For
every pair of maps i < j and every scale, the (u_i, u_j) plane is divided into a K × K
grid of SNR cells, and each cell holds the ℓ1 weight of the pixels that fall in it:

    L^{ij}_{ab} = Σ_{p : (u_i(p), u_j(p)) ∈ B_a × B_b}  ½ (|u_i(p)| + |u_j(p)|)

Coefficients outside the grid go to the edge cells. The data vector concatenates the K²
cells over all pairs and scales: 6 pairs × 5 scales × 100 cells = 3000 numbers for four
bins and K = 10.

## Usage

```python
from joint_l1 import JointL1                 # flat maps: (n_maps, n_bins, H, W)

jl1 = JointL1(n_bins=10, n_scales=5)
jl1.fit_noise_sigma(noise_maps)              # noise-only maps, made like the data
jl1.fit_snr_range(training_maps)             # once, on simulations
jl1.save("joint_l1_setup.npz")               # freezes sigma and the grid

x = jl1(maps)                                # (n_maps, 3000)
cells = jl1.unflatten(x)                     # (n_maps, n_pairs, n_scales, K, K)
```

```python
from joint_l1_sphere import JointL1Sphere    # HEALPix RING maps: (n_maps, n_bins, npix)

jl1 = JointL1Sphere(nside=512, n_bins=10, n_scales=5, mask=footprint)
jl1.fit_noise_sigma(noise_maps)
jl1.fit_snr_range(training_maps)
x = jl1(maps)
```

The noise σ and the SNR grid are part of the statistic's definition: fit them once, on
noise-only maps and on training simulations, and use the same values for every simulation
and for the observed data (`save` / `load`). Measure σ on noise maps passed through the
same map-making as the data. In the paper, the band limit of the maps lowered the
finest-scale σ to about half its white-noise value. For BNT, pass the matrix as
`mixing_matrix`; it is applied to the noise maps too.

Running `python joint_l1.py` or `python joint_l1_sphere.py` runs a short demonstration on
toy maps.

## Flat sky: the paper's settings

`JointL1` defaults to the paper: 5 starlet scales (four wavelet scales and the coarse one,
whose mean is removed), mirror boundaries, K = 10, and one SNR range for all bins and
scales between the 0.5 and 99.5 percentiles of u on training simulations, widened by 5% on
each side. The paper's code described that range as fitted per bin and scale, but a
reshape in `calibrate_joint_ranges` pooled all bins and scales, so the published numbers
use one shared range; `fit_snr_range(..., per_bin_and_scale=True)` gives the separate
ranges. The module docstring lists the rest of the paper's configuration.

`validate_against_paper_code.py` runs `src/nn_l1_sbi/flatsky/flatsky_joint_stats.py` and
`wl_stats_torch` at the pinned commit on the same maps. The starlet coefficients agree to
1e-16 and every feature cell is identical, with and without BNT.

## On the sphere

`JointL1Sphere` uses the spherical starlet of CosmoStat's `CMRStarlet` (B3 splines in ℓ
with cutoffs at ℓmax/2^j, ℓmax = 3 nside), re-implemented with healpy, and the same cell
code as the flat version. Three differences from the flat defaults:

- **Mask.** Pixels outside the footprint are set to zero before the transform and left out
  of the cells, σ and the grid. Apply the same mask to the simulations and the data.
- **Map mean.** Each map's mean over the footprint is removed before the transform, so a
  constant offset changes nothing. CosmoGrid's stored convergence maps have a mean of up to
  0.055. Without the removal, a mask turns that offset into a step at the footprint edge,
  and the coarse-scale coefficients there reached −110σ.
- **Grid.** One SNR range per bin and scale by default. On CosmoGrid full-sky maps the
  spread of u grows about tenfold from the finest scale to the coarse one, and a shared
  range left the finest scale in a few central cells.

`validate_sphere.py` checks the transform against `CMRStarlet` (identical to machine
precision), the analytic white-noise σ, the mask, the mixing matrix and `save` / `load`.
On 32 CosmoGrid fiducial realisations (nside 512, a 14 000 deg² footprint), the statistic
responds to σ8 and to baryons in the expected directions. With 16 CPU threads a 4-bin map
set takes about 3 s at nside 512 and about 15 s at nside 1024 (peak memory about 7 GB). No
spherical configuration has been used for inference yet.

## Citation

Tersenov, Starck & Kilbinger, *The joint wavelet ℓ1-norm matches neural compression for
tomographic weak-lensing inference* (submitted).
