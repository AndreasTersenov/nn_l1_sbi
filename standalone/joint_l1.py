"""Joint wavelet l1-norm for tomographic weak-lensing convergence maps.

Standalone reference implementation of the summary statistic introduced in
Tersenov, Starck & Kilbinger, "The joint wavelet l1-norm matches neural compression for
tomographic weak-lensing inference" (submitted). The paper's full pipeline lives in
https://github.com/AndreasTersenov/nn_l1_sbi; this file depends on NumPy only and gives
the same numbers as that pipeline (validate_against_paper_code.py checks it).
The same statistic on HEALPix maps is in joint_l1_sphere.py.

Definition
----------
Each tomographic convergence map kappa_i is decomposed with the starlet transform
(isotropic undecimated wavelet, B3-spline "a trous" algorithm) into n_scales - 1
wavelet scales plus a coarse scale. At scale s, the coefficient of map i at pixel p in
signal-to-noise units is

    u_i(p) = (W_s kappa_i)(p) / sigma_{i,s},

where sigma_{i,s} is the standard deviation of the scale-s coefficients of noise-only
maps of bin i, measured once and then kept fixed. For every pair of maps i < j and
every scale, the (u_i, u_j) plane is divided into a fixed K x K grid of SNR cells
B_a x B_b, and each cell holds the l1 weight of the pixels that fall in it,

    L^{ij}_{ab} = sum over pixels p with (u_i(p), u_j(p)) in B_a x B_b of
                  (|u_i(p)| + |u_j(p)|) / 2 .

Coefficients outside the grid are counted in the edge cells, so every pixel is counted.
The data vector concatenates the K^2 cells over all pairs and scales, in the order
(pair, scale, a * K + b), with pairs (0,1), (0,2), ..., (n-2, n-1). Row index a bins
u_i, column index b bins u_j. With 4 tomographic bins, 5 scales and K = 10 this gives
6 x 5 x 100 = 3000 numbers per set of maps.

Summing the counts in each cell instead of the l1 weight would give a 2-D histogram,
i.e. an estimate of the joint PDF. In the paper that version over-fitted and failed the
coverage tests, while the l1-weighted one passed them.

Settings used in the paper
--------------------------
* Maps: 4 tomographic bins, 80 x 80 pixel flat-sky patches of 10 x 10 deg
  (7.5 arcmin pixels), Gaussian shape noise (sigma_e = 0.26, n_gal = 10 arcmin^-2).
* Starlet: n_scales = 5 (four wavelet scales + coarse), mirror boundaries, spatial mean
  of the coarse scale removed. This is ``starlet_transform`` below.
* sigma_{i,s}: measured from 48 noise-only realisations passed through the same map
  pipeline as the signal (``fit_noise_sigma``). Measure sigma from noise maps made the
  same way as your data: in the paper, the harmonic band limit lowered the finest-scale
  sigma to about half of the white-noise value.
* Grid: K = 10, one SNR range shared by all bins and scales, set from the 0.5 and 99.5
  percentiles of u over about 3600 training simulations (all cosmologies) and widened by
  5% of its span on each side (``fit_snr_range`` with its defaults). This gave about
  [-7.5, 8.3] without BNT and [-3.4, 3.5] in the BNT basis.
* BNT: a fixed 4 x 4 matrix applied to the maps before the wavelet transform
  (``mixing_matrix``). The noise sigma must then be measured in the same basis, which
  ``fit_noise_sigma`` does, since it mixes the noise maps with the same matrix.
* Downstream, not part of the statistic: log(1 + x), z-scoring with the training-set
  mean and std, clipping at 5 std, removing near-constant features, then a VMIM
  compression to 10 numbers before the neural density estimator.

The noise sigma and the SNR grid are part of the statistic's definition. Fix them once,
from noise maps and training simulations, and use the same values for every simulation
and for the observed data. ``save`` / ``load`` store them together.

Usage
-----
    import numpy as np
    from joint_l1 import JointL1

    # maps: (n_maps, n_bins, H, W) convergence maps with noise, float
    jl1 = JointL1(n_bins=10, n_scales=5)
    jl1.fit_noise_sigma(noise_maps)           # noise-only maps, same shape convention
    jl1.fit_snr_range(training_maps[:3600])   # once, on simulations
    jl1.save("joint_l1_setup.npz")            # freeze sigma and grid

    x = jl1(maps)                    # (n_maps, 3000) for 4 bins
    cells = jl1.unflatten(x)         # (n_maps, n_pairs, n_scales, K, K)
"""

from __future__ import annotations

from itertools import combinations

import numpy as np

__all__ = ["JointL1", "starlet_transform", "noise_sigma_from_maps"]

# B3-spline filter of the starlet transform: [1, 4, 6, 4, 1] / 16
_B3_SPLINE = np.array([1.0, 4.0, 6.0, 4.0, 1.0]) / 16.0


# ---------------------------------------------------------------------------
# Starlet transform
# ---------------------------------------------------------------------------
def _b3_smooth(images: np.ndarray, step: int) -> np.ndarray:
    """Smooth the last two axes with the B3 spline, taps `step` pixels apart.

    Mirror boundaries without repeating the edge pixel (NumPy 'reflect', the same as
    PyTorch's 'reflect' padding used in the paper's code).
    """
    pad = 2 * step
    height, width = images.shape[-2:]
    pad_width = [(0, 0)] * (images.ndim - 2) + [(pad, pad), (pad, pad)]
    padded = np.pad(images, pad_width, mode="reflect")

    # The 2-D filter is separable: filter along rows, then along columns.
    smoothed_rows = np.zeros(images.shape[:-2] + (height, padded.shape[-1]))
    for tap, coefficient in enumerate(_B3_SPLINE):
        offset = tap * step
        smoothed_rows += coefficient * padded[..., offset:offset + height, :]

    smoothed = np.zeros(images.shape)
    for tap, coefficient in enumerate(_B3_SPLINE):
        offset = tap * step
        smoothed += coefficient * smoothed_rows[..., :, offset:offset + width]
    return smoothed


def starlet_transform(images: np.ndarray, n_scales: int = 5) -> np.ndarray:
    """Starlet transform of images (..., H, W) -> (..., n_scales, H, W).

    The first n_scales - 1 outputs are the wavelet scales from fine to coarse (the
    differences of successive B3-spline smoothings), the last one is the coarse scale
    with its spatial mean removed. Adding all outputs back gives the input minus its mean.
    """
    if n_scales < 2:
        raise ValueError(f"n_scales must be at least 2, got {n_scales}")
    images = np.asarray(images, dtype=np.float64)
    largest_pad = 2 * 2 ** (n_scales - 2)
    if largest_pad >= min(images.shape[-2:]):
        raise ValueError(
            f"maps of shape {images.shape[-2:]} are too small for {n_scales} scales "
            f"(the coarsest filter reaches {largest_pad} pixels past the edge)")

    scales = []
    current = images
    for scale_index in range(n_scales - 1):
        smoother = _b3_smooth(current, step=2 ** scale_index)
        scales.append(current - smoother)
        current = smoother
    coarse = current - current.mean(axis=(-2, -1), keepdims=True)
    scales.append(coarse)
    return np.stack(scales, axis=-3)


def _as_mixing_matrix(mixing_matrix):
    if mixing_matrix is None:
        return None
    matrix = np.asarray(mixing_matrix, dtype=np.float64)
    if matrix.ndim != 2:
        raise ValueError(f"mixing_matrix must be 2-D, got shape {matrix.shape}")
    return matrix


def noise_sigma_from_maps(noise_maps, n_scales: int = 5, mixing_matrix=None,
                          batch_size: int = 64) -> np.ndarray:
    """Noise level per (bin, scale) of flat maps, shape (n_bins, n_scales).

    Shortcut for JointL1(n_scales=..., mixing_matrix=...).fit_noise_sigma(noise_maps).
    """
    statistic = JointL1(n_scales=n_scales, mixing_matrix=mixing_matrix,
                        batch_size=batch_size)
    return statistic.fit_noise_sigma(noise_maps)


# ---------------------------------------------------------------------------
# The statistic
# ---------------------------------------------------------------------------
class JointL1:
    """Joint wavelet l1-norm of a set of tomographic flat-sky convergence maps.

    Maps are arrays (n_maps, n_bins, H, W), or (n_bins, H, W) for a single set.

    Parameters
    ----------
    noise_sigma : None or array (n_bins, n_scales)
        Noise level of the starlet coefficients of each bin and scale, in the basis the
        statistic is computed in (after mixing_matrix). None means not set yet: call
        ``fit_noise_sigma`` on noise-only maps.
    n_bins : int
        K, the number of SNR bins along each axis of the grid (paper: 10).
    snr_range : None, (lo, hi), or array (n_bins, n_scales, 2)
        Range of the SNR grid. A pair is shared by all bins and scales, as in the paper.
        None means not set yet: call ``fit_snr_range`` on training simulations.
    mixing_matrix : None or array (n_bins, n_input_bins)
        Optional fixed linear recombination of the input maps applied before the wavelet
        transform, e.g. the BNT matrix: mixed_a = sum_b M[a, b] kappa_b.
    n_scales : int
        Number of starlet scales including the coarse one (paper: 5). Taken from
        noise_sigma when that is given.
    batch_size : int
        Number of map sets processed at once (memory control only).
    """

    # number of pixel axes of one map: 2 for flat (H, W), 1 for HEALPix
    _pixel_axes = 2

    def __init__(self, noise_sigma=None, n_bins: int = 10, snr_range=None,
                 mixing_matrix=None, n_scales: int | None = None, batch_size: int = 64):
        self.n_bins = int(n_bins)
        self.mixing_matrix = _as_mixing_matrix(mixing_matrix)
        self.batch_size = int(batch_size)
        self.n_scales = 5 if n_scales is None else int(n_scales)
        self.noise_sigma = None
        self.n_tomo = None if self.mixing_matrix is None else self.mixing_matrix.shape[0]
        self.pairs = None
        self.snr_range = None
        if noise_sigma is not None:
            self._set_noise_sigma(noise_sigma, n_scales)
        if snr_range is not None:
            if self.noise_sigma is None:
                raise ValueError("pass noise_sigma together with snr_range")
            self.snr_range = self._as_range_table(snr_range)

    # -- construction helpers ----------------------------------------------
    @classmethod
    def from_noise_maps(cls, noise_maps, **kwargs):
        """Build the statistic and measure its noise sigma on noise-only maps."""
        statistic = cls(**kwargs)
        statistic.fit_noise_sigma(noise_maps)
        return statistic

    def _set_noise_sigma(self, noise_sigma, n_scales_requested=None):
        noise_sigma = np.asarray(noise_sigma, dtype=np.float64)
        if noise_sigma.ndim != 2:
            raise ValueError(
                f"noise_sigma must have shape (n_bins, n_scales), got {noise_sigma.shape}")
        if not np.all(noise_sigma > 0):
            raise ValueError("noise_sigma must be strictly positive")
        if n_scales_requested is not None and noise_sigma.shape[1] != n_scales_requested:
            raise ValueError(f"noise_sigma has {noise_sigma.shape[1]} scales, "
                             f"n_scales is {n_scales_requested}")
        if self.n_tomo is not None and noise_sigma.shape[0] != self.n_tomo:
            raise ValueError(f"noise_sigma has {noise_sigma.shape[0]} bins, "
                             f"the mixing matrix gives {self.n_tomo}")
        self.noise_sigma = noise_sigma
        self.n_tomo, self.n_scales = noise_sigma.shape
        self.pairs = list(combinations(range(self.n_tomo), 2))

    def _as_range_table(self, snr_range) -> np.ndarray:
        table = np.asarray(snr_range, dtype=np.float64)
        if table.shape == (2,):
            table = np.broadcast_to(table, (self.n_tomo, self.n_scales, 2)).copy()
        if table.shape != (self.n_tomo, self.n_scales, 2):
            raise ValueError(
                f"snr_range must be (lo, hi) or have shape ({self.n_tomo}, "
                f"{self.n_scales}, 2), got {table.shape}")
        if not np.all(table[..., 1] > table[..., 0]):
            raise ValueError("snr_range needs hi > lo everywhere")
        return table

    def _require(self, noise=True, grid=False):
        if noise and self.noise_sigma is None:
            raise RuntimeError("the noise sigma is not set: pass noise_sigma=... or call "
                               "fit_noise_sigma() on noise-only maps first")
        if grid and self.snr_range is None:
            raise RuntimeError("the SNR grid is not set: pass snr_range=... or call "
                               "fit_snr_range() on training simulations first")

    # -- geometry hooks (overridden for HEALPix maps) ------------------------
    def _prepare_maps(self, maps) -> np.ndarray:
        """Maps as float64 (n_maps, n_bins, *pixel_shape), mixed across bins if asked."""
        maps = np.asarray(maps, dtype=np.float64)
        if maps.ndim == self._pixel_axes + 1:
            maps = maps[np.newaxis]
        if maps.ndim != self._pixel_axes + 2:
            raise ValueError(f"maps must have {self._pixel_axes + 2} axes "
                             f"(n_maps, n_bins, pixels...), got shape {maps.shape}")
        if self.mixing_matrix is not None:
            # mixed[n, a] = sum_b M[a, b] * maps[n, b], e.g. the BNT transform
            maps = np.einsum("ab,nb...->na...", self.mixing_matrix, maps)
        if self.n_tomo is not None and maps.shape[1] != self.n_tomo:
            raise ValueError(f"expected {self.n_tomo} tomographic bins (after mixing), "
                             f"got {maps.shape[1]}")
        if not np.all(np.isfinite(maps)):
            raise ValueError("maps contain NaN or inf values")
        return maps

    def _wavelet(self, maps) -> np.ndarray:
        """(n_maps, n_bins, *pixel_shape) -> (n_maps, n_bins, n_scales, *pixel_shape)."""
        return starlet_transform(maps, self.n_scales)

    def _used_pixels(self, coefficients) -> np.ndarray:
        """Flatten the pixel axes to (..., n_pixels), keeping only the pixels used."""
        return coefficients.reshape(coefficients.shape[:3] + (-1,))

    def _coefficients(self, maps) -> np.ndarray:
        """Wavelet coefficients of the used pixels, (n_maps, n_bins, n_scales, n_pixels)."""
        return self._used_pixels(self._wavelet(self._prepare_maps(maps)))

    def _batches(self, maps):
        maps = np.asarray(maps)
        if maps.ndim == self._pixel_axes + 1:
            maps = maps[np.newaxis]
        for start in range(0, maps.shape[0], self.batch_size):
            yield maps[start:start + self.batch_size]

    # -- fitting the fixed parts ---------------------------------------------
    def fit_noise_sigma(self, noise_maps) -> np.ndarray:
        """Measure the noise level of each (bin, scale) on noise-only maps.

        noise_maps: noise-only convergence maps made by the same pipeline as the data
        (same projection, smoothing, band limit, mask), in the input basis; they are
        mixed with mixing_matrix like the data. The noise level is the root mean square
        of the coefficients over realisations and pixels (the noise has zero mean).
        Returns the (n_bins, n_scales) table, also stored in self.noise_sigma.
        """
        sum_of_squares = None
        n_values = 0
        for batch in self._batches(noise_maps):
            coefficients = self._coefficients(batch)        # (n, bins, scales, pixels)
            batch_sum = np.sum(coefficients ** 2, axis=(0, 3))
            sum_of_squares = batch_sum if sum_of_squares is None else sum_of_squares + batch_sum
            n_values += coefficients.shape[0] * coefficients.shape[3]
        self._set_noise_sigma(np.sqrt(sum_of_squares / n_values))
        if self.snr_range is not None:
            self.snr_range = None      # a grid in SNR units is tied to the old sigma
        return self.noise_sigma

    def fit_snr_range(self, training_maps, per_bin_and_scale: bool = False,
                      percentiles=(0.5, 99.5), margin: float = 0.05,
                      pixel_stride: int | None = 37) -> np.ndarray:
        """Set the SNR grid from percentiles of u over training simulations.

        Use a representative subset of the training simulations (all cosmologies), never
        the observed data, and call this once. The defaults reproduce the paper: one range
        shared by all bins and scales, between the 0.5 and 99.5 percentiles of u (every
        `pixel_stride`-th pixel), widened by `margin` times its span on each side.
        per_bin_and_scale=True sets a separate range for each bin and scale instead.
        Returns the (n_bins, n_scales, 2) range table, also stored in self.snr_range.
        """
        self._require(noise=True)
        samples = []
        for batch in self._batches(training_maps):
            snr = self._snr_pixels(batch)                  # (n, bins, scales, pixels)
            stride = pixel_stride if pixel_stride is not None else self._auto_stride(snr)
            snr = snr[..., ::stride]
            # (maps, bins, scales, pixels) -> (maps * pixels, bins, scales)
            samples.append(snr.transpose(0, 3, 1, 2).reshape(-1, self.n_tomo, self.n_scales))
        samples = np.concatenate(samples, axis=0)

        if per_bin_and_scale:
            low = np.percentile(samples, percentiles[0], axis=0)     # (bins, scales)
            high = np.percentile(samples, percentiles[1], axis=0)
        else:
            low = np.full((self.n_tomo, self.n_scales), np.percentile(samples, percentiles[0]))
            high = np.full((self.n_tomo, self.n_scales), np.percentile(samples, percentiles[1]))
        span = high - low
        self.snr_range = np.stack([low - margin * span, high + margin * span], axis=-1)
        return self.snr_range

    @staticmethod
    def _auto_stride(snr, samples_per_map=50_000):
        return max(1, snr.shape[-1] // samples_per_map)

    # -- the statistic ---------------------------------------------------------
    def _snr_pixels(self, maps) -> np.ndarray:
        coefficients = self._coefficients(maps)       # a fresh array: divide in place
        coefficients /= self.noise_sigma[np.newaxis, :, :, np.newaxis]
        return coefficients

    def snr_coefficients(self, maps) -> np.ndarray:
        """Starlet coefficients in SNR units, (n_maps, n_bins, n_scales, *pixel_shape)."""
        self._require(noise=True)
        coefficients = self._wavelet(self._prepare_maps(maps))
        sigma_shape = self.noise_sigma.shape + (1,) * self._pixel_axes
        return coefficients / self.noise_sigma.reshape(sigma_shape)[np.newaxis]

    @property
    def n_features(self) -> int:
        self._require(noise=True)
        return len(self.pairs) * self.n_scales * self.n_bins ** 2

    def compute(self, maps) -> np.ndarray:
        """Joint l1-norm data vectors, (n_maps, n_features), float64."""
        self._require(noise=True, grid=True)
        rows = [self._cells(self._snr_pixels(batch)) for batch in self._batches(maps)]
        return np.concatenate(rows, axis=0)

    __call__ = compute

    def _cells(self, snr) -> np.ndarray:
        """snr (n_maps, n_bins, n_scales, n_pixels) -> (n_maps, n_features)."""
        n_maps = snr.shape[0]
        n_cells = self.n_bins ** 2
        # each map gets its own block of K^2 cells, so one bincount fills all maps
        block_start = np.arange(n_maps)[:, np.newaxis] * n_cells
        features = np.empty((n_maps, len(self.pairs), self.n_scales, n_cells))

        for scale in range(self.n_scales):
            snr_scale = snr[:, :, scale]                               # (maps, bins, pixels)
            low = self.snr_range[np.newaxis, :, scale, 0, np.newaxis]
            high = self.snr_range[np.newaxis, :, scale, 1, np.newaxis]
            # SNR bin index along each axis; out-of-range values go to the edge bins
            bin_index = np.floor((snr_scale - low) / (high - low) * self.n_bins)
            bin_index = np.clip(bin_index, 0, self.n_bins - 1).astype(np.int64)
            abs_snr = np.abs(snr_scale)

            for pair_index, (i, j) in enumerate(self.pairs):
                cell = bin_index[:, i] * self.n_bins + bin_index[:, j]   # (maps, pixels)
                weight = 0.5 * (abs_snr[:, i] + abs_snr[:, j])
                cell_sums = np.bincount((cell + block_start).ravel(), weights=weight.ravel(),
                                        minlength=n_maps * n_cells)
                features[:, pair_index, scale] = cell_sums.reshape(n_maps, n_cells)
        return features.reshape(n_maps, -1)

    def unflatten(self, features) -> np.ndarray:
        """(n_maps, n_features) -> (n_maps, n_pairs, n_scales, K, K); axis -2 bins u_i."""
        self._require(noise=True)
        features = np.asarray(features)
        return features.reshape(features.shape[:-1] + (
            len(self.pairs), self.n_scales, self.n_bins, self.n_bins))

    # -- freezing the setup -------------------------------------------------
    def _extra_state(self) -> dict:
        return {}

    @classmethod
    def _extra_init_kwargs(cls, stored) -> dict:
        return {}

    def save(self, path) -> None:
        """Store everything that defines the statistic: sigma, grid, mixing, geometry."""
        self._require(noise=True, grid=True)
        np.savez(path, noise_sigma=self.noise_sigma, snr_range=self.snr_range,
                 n_bins=self.n_bins,
                 mixing_matrix=(np.zeros((0, 0)) if self.mixing_matrix is None
                                else self.mixing_matrix),
                 **self._extra_state())

    @classmethod
    def load(cls, path, **kwargs):
        with np.load(path) as stored:
            mixing_matrix = stored["mixing_matrix"]
            init_kwargs = cls._extra_init_kwargs(stored)
            init_kwargs.update(kwargs)
            return cls(noise_sigma=stored["noise_sigma"], n_bins=int(stored["n_bins"]),
                       snr_range=stored["snr_range"],
                       mixing_matrix=None if mixing_matrix.size == 0 else mixing_matrix,
                       **init_kwargs)


# ---------------------------------------------------------------------------
# Demonstration on toy maps: python joint_l1.py
# ---------------------------------------------------------------------------
def _toy_tomographic_maps(n_maps, n_tomo=4, npix=80, amplitude=0.02, seed=0):
    """Correlated, mildly non-Gaussian toy convergence maps (n_maps, n_tomo, npix, npix).

    Each bin is a lognormal transform of a Gaussian field made of a cumulative sum of
    independent layers, so neighbouring bins are correlated as in tomographic lensing.
    For testing the code only, not a lensing simulation.
    """
    rng = np.random.default_rng(seed)
    kx = np.fft.fftfreq(npix)[:, np.newaxis]
    ky = np.fft.rfftfreq(npix)[np.newaxis, :]
    k = np.sqrt(kx ** 2 + ky ** 2)
    k[0, 0] = np.inf
    filter_k = k ** -1.0                                  # P(k) ~ k^-2
    white = rng.standard_normal((n_maps, n_tomo, npix, npix))
    layers = np.fft.irfft2(np.fft.rfft2(white) * filter_k, s=(npix, npix))
    layers /= layers.std(axis=(-2, -1), keepdims=True)
    gaussian = np.cumsum(layers, axis=1) / np.sqrt(np.arange(1, n_tomo + 1))[:, None, None]
    lognormal = np.exp(gaussian) - np.exp(0.5)            # mean ~ 0, skewed
    return amplitude * lognormal / lognormal.std()


def _check_cells_conserve_weight(statistic, maps, features):
    """Every used pixel lands in exactly one cell, so the cells of one (pair, scale) add
    up to the total l1 weight of that pair at that scale."""
    snr = statistic._snr_pixels(maps)
    cells = statistic.unflatten(features)
    for pair_index, (i, j) in enumerate(statistic.pairs):
        total = 0.5 * (np.abs(snr[:, i]) + np.abs(snr[:, j])).sum(axis=-1)
        assert np.allclose(cells[:, pair_index].sum(axis=(-2, -1)), total)


if __name__ == "__main__":
    n_tomo, npix, pixel_noise = 4, 80, 0.012
    rng = np.random.default_rng(1)

    signal = _toy_tomographic_maps(400, n_tomo, npix)
    maps = signal + pixel_noise * rng.standard_normal(signal.shape)
    noise_only = pixel_noise * rng.standard_normal((48, n_tomo, npix, npix))

    jl1 = JointL1(n_bins=10, n_scales=5)
    jl1.fit_noise_sigma(noise_only)
    jl1.fit_snr_range(maps[:300])                         # "training simulations"
    x = jl1(maps[300:])                                   # "data"
    print("noise sigma per (bin, scale):\n", np.round(jl1.noise_sigma, 5))
    print("SNR grid:", np.round(jl1.snr_range[0, 0], 2))
    print("data vector shape:", x.shape)
    _check_cells_conserve_weight(jl1, maps[300:], x)
    print("check passed: the cells conserve the total l1 weight of each pair and scale")
