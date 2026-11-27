"""Joint wavelet l1-norm on the sphere, for tomographic HEALPix convergence maps.

The statistic is the one defined in joint_l1.py (see its docstring) with the flat starlet
replaced by the spherical starlet of CosmoStat's ``pycs.sparsity.mrs.mrs_starlet.CMRStarlet``,
the transform behind ``get_wtl1_sphere`` in ``pycs/astro/wl/hos_peaks_l1.py``. The cells are
computed by the same code as the flat version, so the two share one definition:

    u_i(p) = (W_s kappa_i)(p) / sigma_{i,s},
    L^{ij}_{ab} = sum over pixels p with (u_i(p), u_j(p)) in B_a x B_b of (|u_i| + |u_j|) / 2,

with every coefficient outside the SNR grid counted in the edge cells, and the data vector
ordered (pair, scale, a * K + b). Only healpy and NumPy are needed; pycs is not imported.

Spherical starlet
-----------------
The transform works in harmonic space. The scaling functions are B3 splines in ell,

    phi_j(ell) = (3/2) B3(2 ell / ell_j),   ell_j = lmax / 2^j,   lmax = 3 nside by default,

so phi_j vanishes for ell >= ell_j. With c_0 = kappa and c_j = kappa filtered by phi_j
(j >= 1), wavelet scale j is c_j - c_{j+1} for j = 0 .. n_scales - 2, and the last output
is the coarse scale c_{n_scales - 1}. The finest scale keeps everything above lmax / 2,
including the part of the map above lmax. This is CMRStarlet's default python transform
(map2alm with iter=3, the same filters); validate_sphere.py checks the coefficients
against it. As in the flat statistic, the mean of the coarse scale is removed (over the
observed pixels); CMRStarlet itself does not do this, set subtract_coarse_mean=False to
keep its coarse scale as it is. The dyadic bands are set by lmax, so use the same nside
and lmax for the simulations and the data.

Footprint and noise
-------------------
* mask: boolean HEALPix map, True where the sky is observed. Unobserved pixels are set to
  zero before the transform (so hp.UNSEEN or NaN there is harmless) and are left out of
  the cells, the noise sigma, the SNR-grid fit and the coarse mean. Coefficients near the
  edge of the footprint are affected by these zeros; that is part of the statistic, so
  apply the same mask to every simulation and to the data.
* map mean: each map's mean over the observed pixels is removed before the transform
  (subtract_map_mean=True), so that, as in the flat statistic, a constant offset does not
  change the result. Without it, an offset becomes a step at the footprint edge. CosmoGrid's
  stored convergence maps have a mean of up to 0.055 (six times their pixel scatter in the
  deepest Stage-III bin), and the step gave coarse-scale coefficients of -110 sigma within
  3 degrees of the edge of a 14 000 deg^2 cap.
* noise sigma: one number per (bin, scale), measured on noise-only maps passed through
  the same map-making as the data (fit_noise_sigma). With a mass-mapping step (e.g.
  Kaiser-Squires from shear) the noise in kappa is not white, which is why sigma is
  measured rather than propagated. For white noise of uniform depth added directly to
  kappa, white_noise_sigma gives it analytically (cosmostat's normalisation, within about
  1% of the measured value). With strongly varying depth across the footprint a
  single sigma per (bin, scale) gives an SNR that varies with depth; per-pixel noise
  levels are not supported here.
* Maps must be in RING ordering (healpy's default); use hp.reorder for NEST maps.

The SNR grid defaults to one range per bin and scale (0.5 and 99.5 percentiles, 5% margin),
unlike the flat paper setting of one range for all of them. On CosmoGrid full-sky maps
(nside 512, Stage-III noise) the spread of u grows about tenfold from the finest scale to
the coarse one, so a shared range left the finest scale, where baryons act, in a few
central cells. per_bin_and_scale=False restores the paper's shared range.

Usage
-----
    import numpy as np
    from joint_l1_sphere import JointL1Sphere

    # maps: (n_maps, n_bins, npix) HEALPix RING maps with noise; mask: (npix,) bool
    jl1 = JointL1Sphere(nside=1024, n_bins=10, n_scales=5, mask=mask)
    jl1.fit_noise_sigma(noise_maps)         # a few noise-only map sets
    jl1.fit_snr_range(training_maps)        # once, on simulations
    jl1.save("joint_l1_sphere_setup.npz")   # sigma, grid, mask, nside, lmax

    x = jl1(maps)                           # (n_maps, 3000) for 4 bins and K = 10
"""

from __future__ import annotations

import healpy as hp
import numpy as np

from joint_l1 import JointL1, _check_cells_conserve_weight

__all__ = ["JointL1Sphere", "spherical_starlet_filters", "spherical_starlet_transform",
           "white_noise_sigma"]


def spherical_starlet_filters(lmax: int, n_scales: int) -> np.ndarray:
    """Scaling functions phi_j(ell), shape (n_scales, lmax + 1).

    phi_j is the B3 spline (3/2) B3(2 ell / ell_j), equal to 1 at ell = 0 and vanishing at
    ell_j = lmax / 2^j, written exactly as cosmostat's spline2 so the transforms agree.
    """
    ell = np.arange(lmax + 1, dtype=np.float64)
    filters = np.empty((n_scales, lmax + 1))
    for scale in range(n_scales):
        x = 2.0 * ell / (lmax / 2 ** scale)
        filters[scale] = (3 / 2) * 1 / 12 * (
            np.abs(x - 2) ** 3 - 4 * np.abs(x - 1) ** 3 + 6 * np.abs(x) ** 3
            - 4 * np.abs(x + 1) ** 3 + np.abs(x + 2) ** 3)
    return filters


def spherical_starlet_transform(healpix_map, n_scales: int = 5, lmax: int | None = None,
                                alm_iterations: int = 3, filters=None) -> np.ndarray:
    """Spherical starlet transform of one RING HEALPix map -> (n_scales, npix).

    Outputs n_scales - 1 wavelet scales from fine to coarse, then the coarse scale (its
    mean is kept here). Same as cosmostat's CMRStarlet.transform with default filters.
    """
    healpix_map = np.asarray(healpix_map, dtype=np.float64)
    nside = hp.npix2nside(healpix_map.size)
    lmax = 3 * nside if lmax is None else lmax
    if filters is None:
        filters = spherical_starlet_filters(lmax, n_scales)

    alm = hp.map2alm(healpix_map, lmax=lmax, iter=alm_iterations)
    coefficients = np.empty((n_scales, healpix_map.size))
    current = healpix_map
    for scale in range(n_scales - 1):
        smoother = hp.alm2map(hp.almxfl(alm, filters[scale + 1]), nside, lmax=lmax)
        coefficients[scale] = current - smoother
        current = smoother
    coefficients[n_scales - 1] = current
    return coefficients


def white_noise_sigma(pixel_noise_std, nside: int, n_scales: int = 5,
                      lmax: int | None = None) -> np.ndarray:
    """Noise sigma per (bin, scale) for white pixel noise, shape (n_bins, n_scales).

    Propagates a per-pixel noise std (one number per bin) through the spherical starlet
    analytically, with cosmostat's normalisation (CMRStarlet.TabNorm, the one behind
    get_wtl1_sphere(noise_std=...)). Valid only if the noise in the kappa maps is white
    and of uniform depth, e.g. Gaussian noise added directly to kappa; it agrees with
    fit_noise_sigma on such noise to about 1%. After a mass-mapping step the noise is no
    longer white: measure sigma on noise maps with fit_noise_sigma instead.
    """
    lmax = 3 * nside if lmax is None else lmax
    npix = hp.nside2npix(nside)
    phi = spherical_starlet_filters(lmax, n_scales)
    ell = np.arange(lmax + 1)

    def white_noise_std(filter_ell):            # std of unit white noise after filtering
        return np.sqrt(np.sum((2 * ell + 1) * filter_ell ** 2) / npix)

    norms = np.empty(n_scales)
    for scale in range(n_scales - 1):
        norms[scale] = white_noise_std(phi[scale] - phi[scale + 1])
    norms[n_scales - 1] = white_noise_std(phi[n_scales - 1])
    # the finest scale also holds the noise above lmax, which phi_0 does not reach
    norms[0] = np.sqrt(1.0 - white_noise_std(phi[0]) ** 2 + norms[0] ** 2)
    return np.outer(np.atleast_1d(np.asarray(pixel_noise_std, dtype=np.float64)), norms)


class JointL1Sphere(JointL1):
    """Joint wavelet l1-norm of a set of tomographic HEALPix convergence maps.

    Maps are arrays (n_maps, n_bins, npix) in RING ordering, or (n_bins, npix).

    Parameters
    ----------
    nside : int
        HEALPix resolution of the maps.
    noise_sigma, n_bins, snr_range, mixing_matrix, n_scales
        As in joint_l1.JointL1.
    mask : None or array (npix,)
        Observed footprint, nonzero where observed. None uses the full sky.
    lmax : int or None
        Band limit of the transform; the scales are the dyadic bands below it.
        Default 3 * nside, as in cosmostat.
    subtract_map_mean : bool
        Remove each map's mean over the observed pixels before the transform (default
        True); see "map mean" above.
    subtract_coarse_mean : bool
        Remove the mean of the coarse scale over the observed pixels (default True, as
        in the flat statistic).
    alm_iterations : int
        Iterations of healpy's map2alm (cosmostat uses 3).
    batch_size : int
        Map sets transformed at once (memory control only; default 1).
    """

    _pixel_axes = 1

    def __init__(self, nside: int, noise_sigma=None, n_bins: int = 10, snr_range=None,
                 mixing_matrix=None, n_scales: int | None = None, mask=None,
                 lmax: int | None = None, subtract_map_mean: bool = True,
                 subtract_coarse_mean: bool = True, alm_iterations: int = 3,
                 batch_size: int = 1):
        self.nside = int(nside)
        self.npix = hp.nside2npix(self.nside)
        self.lmax = 3 * self.nside if lmax is None else int(lmax)
        self.subtract_map_mean = bool(subtract_map_mean)
        self.subtract_coarse_mean = bool(subtract_coarse_mean)
        self.alm_iterations = int(alm_iterations)
        if mask is None:
            self.mask = None
        else:
            mask = np.asarray(mask)
            if mask.shape != (self.npix,):
                raise ValueError(f"mask must have shape ({self.npix},), got {mask.shape}")
            self.mask = mask.astype(bool)
            if not self.mask.any():
                raise ValueError("mask has no observed pixel")
        super().__init__(noise_sigma=noise_sigma, n_bins=n_bins, snr_range=snr_range,
                         mixing_matrix=mixing_matrix, n_scales=n_scales,
                         batch_size=batch_size)
        self._filters = spherical_starlet_filters(self.lmax, self.n_scales)

    @classmethod
    def from_noise_maps(cls, noise_maps, **kwargs):
        """Build the statistic (nside taken from the maps) and measure its noise sigma."""
        kwargs.setdefault("nside", hp.npix2nside(np.shape(noise_maps)[-1]))
        return super().from_noise_maps(noise_maps, **kwargs)

    # -- geometry hooks ------------------------------------------------------
    def _prepare_maps(self, maps) -> np.ndarray:
        maps = np.asarray(maps, dtype=np.float64)
        if maps.shape[-1] != self.npix:
            raise ValueError(f"maps must have {self.npix} pixels (nside {self.nside}) "
                             f"on their last axis, got shape {maps.shape}")
        if self.mask is not None:
            maps = np.where(self.mask, maps, 0.0)
        maps = super()._prepare_maps(maps)                 # mixing and checks
        if self.subtract_map_mean:
            observed = maps if self.mask is None else maps[..., self.mask]
            maps = maps - observed.mean(axis=-1, keepdims=True)
            if self.mask is not None:
                maps[..., ~self.mask] = 0.0
        return maps

    def _wavelet(self, maps) -> np.ndarray:
        """(n_maps, n_bins, npix) -> (n_maps, n_bins, n_scales, npix)."""
        coefficients = np.empty(maps.shape[:2] + (self.n_scales, self.npix))
        for map_index in range(maps.shape[0]):
            for bin_index in range(maps.shape[1]):
                coefficients[map_index, bin_index] = spherical_starlet_transform(
                    maps[map_index, bin_index], self.n_scales, self.lmax,
                    self.alm_iterations, filters=self._filters)
        if self.subtract_coarse_mean:
            coarse = coefficients[:, :, -1]
            observed = coarse if self.mask is None else coarse[..., self.mask]
            coarse -= observed.mean(axis=-1, keepdims=True)
        return coefficients

    def _used_pixels(self, coefficients) -> np.ndarray:
        if self.mask is None:
            return coefficients
        return coefficients[..., self.mask]

    def fit_snr_range(self, training_maps, per_bin_and_scale: bool = True,
                      percentiles=(0.5, 99.5), margin: float = 0.05,
                      pixel_stride: int | None = None) -> np.ndarray:
        """As JointL1.fit_snr_range, but one range per bin and scale by default (see the
        module docstring); about 50 000 pixels per map are sampled."""
        return super().fit_snr_range(training_maps, per_bin_and_scale, percentiles,
                                     margin, pixel_stride)

    # -- freezing the setup -------------------------------------------------
    def _extra_state(self) -> dict:
        return {"nside": self.nside, "lmax": self.lmax,
                "mask": np.zeros(0, bool) if self.mask is None else self.mask,
                "subtract_map_mean": self.subtract_map_mean,
                "subtract_coarse_mean": self.subtract_coarse_mean,
                "alm_iterations": self.alm_iterations}

    @classmethod
    def _extra_init_kwargs(cls, stored) -> dict:
        mask = stored["mask"]
        return {"nside": int(stored["nside"]), "lmax": int(stored["lmax"]),
                "mask": None if mask.size == 0 else mask.astype(bool),
                "subtract_map_mean": bool(stored["subtract_map_mean"]),
                "subtract_coarse_mean": bool(stored["subtract_coarse_mean"]),
                "alm_iterations": int(stored["alm_iterations"])}


# ---------------------------------------------------------------------------
# Demonstration on toy maps: python joint_l1_sphere.py
# ---------------------------------------------------------------------------
def _toy_tomographic_healpix_maps(n_maps, nside, n_tomo=4, amplitude=0.02, seed=0):
    """Correlated, mildly non-Gaussian toy HEALPix maps (n_maps, n_tomo, npix).

    Lognormal transform of cumulative sums of independent Gaussian layers, as in the
    flat toy maps. For testing the code only, not a lensing simulation.
    """
    lmax = 3 * nside - 1
    ell = np.arange(lmax + 1)
    cl = 1.0 / (ell + 10.0) ** 2
    np.random.seed(seed)                                   # hp.synfast uses NumPy's global RNG
    maps = np.empty((n_maps, n_tomo, hp.nside2npix(nside)))
    for map_index in range(n_maps):
        layers = np.array([hp.synfast(cl, nside, lmax=lmax) for _ in range(n_tomo)])
        layers /= layers.std(axis=-1, keepdims=True)
        gaussian = np.cumsum(layers, axis=0) / np.sqrt(np.arange(1, n_tomo + 1))[:, None]
        maps[map_index] = np.exp(gaussian) - np.exp(0.5)
    return amplitude * maps / maps.std()


if __name__ == "__main__":
    nside, n_tomo, pixel_noise = 128, 4, 0.01
    rng = np.random.default_rng(1)

    signal = _toy_tomographic_healpix_maps(12, nside, n_tomo)
    maps = signal + pixel_noise * rng.standard_normal(signal.shape)
    noise_only = pixel_noise * rng.standard_normal((4, n_tomo, hp.nside2npix(nside)))
    theta, _ = hp.pix2ang(nside, np.arange(hp.nside2npix(nside)))
    mask = np.abs(np.pi / 2 - theta) > np.radians(20)      # cut a band around the equator

    jl1 = JointL1Sphere(nside, n_bins=10, n_scales=5, mask=mask)
    jl1.fit_noise_sigma(noise_only)
    jl1.fit_snr_range(maps[:8])                            # "training simulations"
    x = jl1(maps[8:])                                      # "data"
    print("noise sigma per (bin, scale):\n", np.round(jl1.noise_sigma, 5))
    print("SNR grid:", np.round(jl1.snr_range[0, 0], 2))
    print("data vector shape:", x.shape)
    _check_cells_conserve_weight(jl1, maps[8:], x)
    print("check passed: the cells conserve the total l1 weight of each pair and scale")
