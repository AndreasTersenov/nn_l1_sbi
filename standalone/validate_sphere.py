"""Checks for joint_l1_sphere.py.

  1. spherical starlet   vs cosmostat CMRStarlet.transform (pycs, default python path)
  2. noise sigma         measured on white noise vs cosmostat's analytic CMRStarlet.TabNorm
  3. mask                hp.UNSEEN or NaN outside the footprint give the same features as zeros
  4. mixing matrix       mixing inside the statistic == feeding it pre-mixed maps
  5. save / load         reloaded statistic gives identical features
  6. conservation        cells add up to the total l1 weight of each pair and scale

Needs healpy and a cosmostat checkout (only pycs.sparsity.mrs.mrs_starlet is imported):
    python validate_sphere.py --cosmostat-src <cosmostat checkout> [--nside 128]
"""
import argparse
import os
import sys
import tempfile

import healpy as hp
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import joint_l1_sphere as sphere  # noqa: E402
from joint_l1 import _check_cells_conserve_weight  # noqa: E402


def report(name, ours, theirs, tolerance):
    relative = np.abs(ours - theirs).max() / np.abs(theirs).max()
    ok = relative <= tolerance
    print(f"  [{'OK ' if ok else 'FAIL'}] {name}: max |diff| / max |value| = {relative:.2e} "
          f"(tolerance {tolerance:.0e})")
    return ok


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cosmostat-src", required=True)
    parser.add_argument("--nside", type=int, default=128)
    args = parser.parse_args()
    sys.path.insert(0, args.cosmostat_src)
    from pycs.sparsity.mrs.mrs_starlet import CMRStarlet

    nside, n_tomo, n_scales, pixel_noise = args.nside, 4, 5, 0.01
    npix = hp.nside2npix(nside)
    rng = np.random.default_rng(5)
    signal = sphere._toy_tomographic_healpix_maps(6, nside, n_tomo, seed=2)
    maps = signal + pixel_noise * rng.standard_normal(signal.shape)
    all_ok = True

    print(f"1. spherical starlet vs cosmostat CMRStarlet (nside {nside}, {n_scales} scales)")
    cosmostat = CMRStarlet()
    cosmostat.init_starlet(nside, nscale=n_scales)
    assert cosmostat.TypeCode == 0, "expected cosmostat's python transform"
    theirs = []
    for bin_index in range(n_tomo):
        cosmostat.transform(maps[0, bin_index])
        theirs.append(np.array(cosmostat.coef))
    theirs = np.array(theirs)                                  # (bins, scales, npix)
    raw = sphere.JointL1Sphere(nside, n_scales=n_scales, subtract_map_mean=False,
                               subtract_coarse_mean=False)
    ours = raw._wavelet(raw._prepare_maps(maps[:1]))[0]
    all_ok &= report("coefficients", ours, theirs, 1e-12)
    for scale in range(n_scales):
        all_ok &= report(f"  scale {scale}", ours[:, scale], theirs[:, scale], 1e-12)

    print("2. noise sigma: white_noise_sigma vs cosmostat TabNorm, and vs measured")
    analytic = sphere.white_noise_sigma([pixel_noise] * n_tomo, nside, n_scales)
    all_ok &= report("white_noise_sigma vs sigma_pix * TabNorm", analytic,
                     pixel_noise * np.tile(np.asarray(cosmostat.TabNorm), (n_tomo, 1)), 1e-12)
    noise = pixel_noise * rng.standard_normal((8, n_tomo, npix))
    measured = sphere.JointL1Sphere(nside, n_scales=n_scales).fit_noise_sigma(noise)
    # averaged over the bins: the coarse scale has few modes, so single bins scatter by ~2%
    ratio = (measured / analytic).mean(axis=0)
    print("         measured / analytic per scale (mean over bins):",
          np.array2string(ratio, precision=3))
    ok = np.all(np.abs(ratio - 1) < 0.02)
    print(f"  [{'OK ' if ok else 'FAIL'}] agree within 2% on white noise")
    all_ok &= ok

    print("3. mask: values outside the footprint do not matter")
    theta, phi = hp.pix2ang(nside, np.arange(npix))
    mask = (np.abs(np.pi / 2 - theta) > np.radians(25)) & (phi > np.radians(40))
    statistic = sphere.JointL1Sphere(nside, n_scales=n_scales, mask=mask)
    statistic.fit_noise_sigma(noise)
    statistic.fit_snr_range(maps[:3])
    zeros_outside = np.where(mask, maps[3:], 0.0)
    unseen_outside = np.where(mask, maps[3:], hp.UNSEEN)
    nan_outside = np.where(mask, maps[3:], np.nan)
    reference = statistic(zeros_outside)
    all_ok &= report("UNSEEN outside vs zeros", statistic(unseen_outside), reference, 1e-14)
    all_ok &= report("NaN outside vs zeros", statistic(nan_outside), reference, 1e-14)
    print(f"         observed fraction {mask.mean():.2f}; features {reference.shape}")
    # a constant offset per bin (CosmoGrid maps carry one) must not change anything
    offset = np.array([0.0076, 0.0164, 0.0361, 0.0545])[None, :, None]
    all_ok &= report("constant offset per bin", statistic(maps[3:] + offset), reference, 1e-9)

    print("4. mixing matrix inside the statistic == pre-mixed maps")
    mixing = np.array([[1.0, 0.0, 0.0, 0.0], [-1.0, 1.0, 0.0, 0.0],
                       [0.4521097, -1.4521097, 1.0, 0.0], [0.0, 0.25127807, -1.251278, 1.0]])
    mixed_inside = sphere.JointL1Sphere(nside, n_scales=n_scales, mixing_matrix=mixing)
    mixed_inside.fit_noise_sigma(noise)
    mixed_inside.fit_snr_range(maps[:3])
    premixed = sphere.JointL1Sphere(nside, n_scales=n_scales)
    premixed.fit_noise_sigma(np.einsum("ab,nbp->nap", mixing, noise))
    premixed.fit_snr_range(np.einsum("ab,nbp->nap", mixing, maps[:3]))
    all_ok &= report("noise sigma", mixed_inside.noise_sigma, premixed.noise_sigma, 1e-12)
    all_ok &= report("features", mixed_inside(maps[3:]),
                     premixed(np.einsum("ab,nbp->nap", mixing, maps[3:])), 1e-10)

    print("5. save / load")
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "setup.npz")
        statistic.save(path)
        reloaded = sphere.JointL1Sphere.load(path)
    all_ok &= report("features after reload", reloaded(maps[3:]), statistic(maps[3:]), 0)
    print(f"         reloaded nside {reloaded.nside}, lmax {reloaded.lmax}, "
          f"mask kept: {np.array_equal(reloaded.mask, mask)}")
    all_ok &= np.array_equal(reloaded.mask, mask)

    print("6. conservation of the l1 weight, with the mask")
    _check_cells_conserve_weight(statistic, maps[3:], statistic(maps[3:]))
    print("  [OK ] cells add up to the total weight of each pair and scale")

    print("ALL CHECKS PASSED" if all_ok else "SOME CHECKS FAILED")
    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
