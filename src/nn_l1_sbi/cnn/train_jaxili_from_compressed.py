"""Reimplementation of the FoM3 helpers imported by `ensemble_eval.py` and
`train_nde_from_compressed.py`.

Supplies `setup_env`, `compute_fom3`, `marginal_stats` and `fom2d`. Each function is
reconstructed against a persisted artefact rather than from memory, as documented
per-function below:

* `compute_fom3` -- VERIFIED EXACTLY. Across 8 persisted `posterior.fom.json` records,
  `fom3 == 1/sqrt(det_cov3)` to machine precision (worst relative error 1.7e-16). The
  definition used throughout the pipeline is
  `FoM3 = 1/sqrt(det(np.cov(samples[:, :3].T)))` with parameter order
  (Omega_m, sigma_8, w_0). The returned key set matches the persisted JSON exactly:
  fom3, det_cov3, logdet_cov3, valid_fom3.

* `marginal_stats` / `fom2d` -- CONVENTION NOT FULLY PINNED DOWN. Reconstructing the
  per-observation correlation matrix from the persisted `sigma` + `fom2d` arrays and
  predicting `fom3` leaves a ~0.2% median residual, so the originals were not plain
  sqrt(diag) / one-over-sqrt-det-of-2x2 (most likely getdist-based, since getdist is in
  the import list). These two are REPORT-ONLY: they enter neither FoM3 nor any error-bar
  term, so the standard definitions are used and the discrepancy is documented rather
  than chased. Downstream code should prefer the persisted covariance over these.

* `setup_env` -- mirrors `setup_environment` in
  npe_l1norm_cross_jaxili_nbody_tomo.py (lines 140-152), which does the same job.

`posterior_moments` is NEW: the original pipeline persisted only fom3 + marginal sigma,
but the ensemble jackknife needs each member's per-observation posterior MEAN and full
3x3 COVARIANCE (a 50/50 pooled mixture has covariance
0.5*(C1+C2) + 0.25*(m1-m2)(m1-m2)^T, so the means are essential). Persist these.

Parameter order throughout: 0=Omega_m, 1=sigma_8, 2=w_0.
"""
from __future__ import annotations

import os

import numpy as np

PARAM_NAMES = ("Omega_m", "sigma_8", "w_0")
PAIRS = ((0, 1), (0, 2), (1, 2))          # (Om,s8), (Om,w0), (s8,w0) -- ensemble_eval order


def setup_env(cuda_devices: str = "0"):
    """Pin the visible GPU, keep TF from grabbing all of it, report the backends.

    Mirrors `setup_environment` in the L1 driver. Returns the torch device
    (the L1 pipeline runs wl_stats_torch on GPU), or None if torch is unavailable.
    """
    os.environ["CUDA_VISIBLE_DEVICES"] = str(cuda_devices)
    print(f"CUDA_VISIBLE_DEVICES = {cuda_devices}")
    try:
        import tensorflow as tf
        for gpu in tf.config.experimental.list_physical_devices("GPU"):
            try:
                tf.config.experimental.set_memory_growth(gpu, True)
            except RuntimeError as exc:
                print(f"TF GPU config: {exc}")
    except Exception as exc:                                  # pragma: no cover
        print(f"TF unavailable: {exc}")
    device = None
    try:
        import torch
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"PyTorch device : {device}")
    except Exception as exc:                                  # pragma: no cover
        print(f"torch unavailable: {exc}")
    try:
        from jax.lib import xla_bridge
        print(f"JAX backend    : {xla_bridge.get_backend().platform}")
    except Exception as exc:                                  # pragma: no cover
        print(f"JAX unavailable: {exc}")
    return device


def _cov3(samples: np.ndarray) -> np.ndarray:
    s = np.asarray(samples, dtype=np.float64)[:, :3]
    return np.cov(s.T)


def compute_fom3(samples: np.ndarray) -> dict:
    """FoM3 = 1/sqrt(det(cov(Omega_m, sigma_8, w_0))).

    Verified against the persisted `posterior.fom.json` records (see module docstring).
    """
    c = _cov3(samples)
    det = float(np.linalg.det(c))
    valid = bool(np.isfinite(det) and det > 0.0)
    sign, logdet = np.linalg.slogdet(c)
    return {
        "fom3": float(1.0 / np.sqrt(det)) if valid else float("nan"),
        "det_cov3": det,
        "logdet_cov3": float(logdet) if sign > 0 else float("nan"),
        "valid_fom3": valid,
    }


def marginal_stats(samples: np.ndarray) -> dict:
    """Per-parameter marginal mean/sigma. Report-only; see module docstring."""
    s = np.asarray(samples, dtype=np.float64)
    mean = s.mean(axis=0)
    sd = s.std(axis=0, ddof=1)
    return {
        "mean": {n: float(mean[i]) for i, n in enumerate(PARAM_NAMES)},
        "sigma": {n: float(sd[i]) for i, n in enumerate(PARAM_NAMES)},
    }


def fom2d(samples: np.ndarray) -> dict:
    """Pairwise 2D FoMs, 1/sqrt(det C2). Report-only; see module docstring."""
    c = _cov3(samples)
    out = {}
    for i, j in PAIRS:
        sub = c[np.ix_((i, j), (i, j))]
        det = float(np.linalg.det(sub))
        key = f"fom2d_{PARAM_NAMES[i]}_{PARAM_NAMES[j]}"
        out[key] = float(1.0 / np.sqrt(det)) if np.isfinite(det) and det > 0 else float("nan")
    return out


def posterior_moments(samples: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """NEW: per-observation posterior mean (3,) and full covariance (3,3).

    The ensemble jackknife cannot be reconstructed from fom3 + marginal sigma alone, so
    persist this pair for every observation.
    """
    s = np.asarray(samples, dtype=np.float64)[:, :3]
    return s.mean(axis=0), np.cov(s.T)


def self_check(seed: int = 0) -> bool:
    """Internal consistency: fom3 must equal 1/sqrt(det) of the returned covariance, the
    marginals must equal sqrt(diag), and the 2D FoMs must match their 2x2 sub-blocks."""
    rng = np.random.default_rng(seed)
    s_om, s_s8, s_w0 = 2.0e-2, 3.0e-2, 1.5e-1
    true_cov = np.array([
        [s_om ** 2,            0.93 * s_om * s_s8, -0.40 * s_om * s_w0],
        [0.93 * s_om * s_s8,   s_s8 ** 2,          -0.20 * s_s8 * s_w0],
        [-0.40 * s_om * s_w0, -0.20 * s_s8 * s_w0,  s_w0 ** 2],
    ])
    x = rng.multivariate_normal([0.26, 0.84, -1.0], true_cov, size=200_000)
    f3 = compute_fom3(x)
    mean, cov = posterior_moments(x)
    d = np.linalg.det(cov)
    ok_det = abs(f3["det_cov3"] - d) / abs(d) < 1e-12
    ok_fom = abs(f3["fom3"] - 1.0 / np.sqrt(d)) / f3["fom3"] < 1e-12
    ms, f2 = marginal_stats(x), fom2d(x)
    ok_sig = all(abs(ms["sigma"][PARAM_NAMES[i]] - np.sqrt(cov[i, i])) /
                 np.sqrt(cov[i, i]) < 1e-6 for i in range(3))
    ok_2d = True
    for i, j in PAIRS:
        sub = cov[np.ix_((i, j), (i, j))]
        got = f2[f"fom2d_{PARAM_NAMES[i]}_{PARAM_NAMES[j]}"]
        ok_2d &= abs(got - 1.0 / np.sqrt(np.linalg.det(sub))) / got < 1e-12
    print(f"  det match  : {ok_det}")
    print(f"  fom3 match : {ok_fom}   (fom3={f3['fom3']:.1f})")
    print(f"  sigma match: {ok_sig}")
    print(f"  fom2d match: {ok_2d}")
    print(f"  recovered rho(Om,s8) = {cov[0,1]/np.sqrt(cov[0,0]*cov[1,1]):.4f} (input 0.93)")
    print(f"  mean = {np.round(mean,4)}  (input [0.26, 0.84, -1.0])")
    return bool(ok_det and ok_fom and ok_sig and ok_2d)


if __name__ == "__main__":
    print("self-check:")
    raise SystemExit(0 if self_check() else 1)
