#!/usr/bin/env python3
"""Step 4/5 — identify the 180 patch centres by bit-exact reproduction against the TFDS.

===============================================================================
 DO NOT REGENERATE THE PATCH CENTRES.  USE inputs/centers_PAPER_180.npy.
 A regenerated tiling is not reproducible across numpy builds (unstable argsort
 over HEALPix tie blocks).  This driver is the PROOF step: it reprojects patches
 from the raw CosmoGridV1 maps and compares them to the TFDS bit for bit, so it
 is what lets you VERIFY the pinned file rather than trust it.
===============================================================================

The 10deg cache recorded `patch_centers` in every .npz, but no 10deg cache file
survived, so the geometry has to be identified rather than read.

This is possible because the whole cache pipeline is deterministic -- including the
shape noise, whose seed is `noise_seed_base + 100*cosmo_idx + perm`. And the cross
TFDS is a bit-exact reserialization of that cache. So for a candidate center set we
can recompute a patch from the raw CosmoGridV1 full-sphere maps and compare to the
TFDS array element-wise. The correct geometry reproduces it exactly; any other
geometry is projecting a different piece of sky and will not come close.

Pipeline reproduced verbatim from build_full_sphere_cross_cache:
   noiseless (4, npix)            <- h5 kg/stage3_lensing{1..4}
   seed = base + 100*cosmo + perm
   noisy = noiseless + N(0, sigma_e/sqrt(n_gal * pixarea_arcmin2(nside)))
   alm   = map2alm(noisy[b], lmax, iter=0)
   map   = alm2map(alm, nside, lmax)                      # channel 0 = auto bin 1
   patch = GnomonicProj(rot=[lon,lat,0], xsize, ysize, reso).projmap(map)
   patch -= patch.mean()                                  # per (patch, channel)

Only channel 0 is needed, but the RNG draw must cover the full (4, npix) array to
land on the same noise realisation.

Candidate geometries are read as centers_*.npy from the work directory (run
find_centers.py first). Outputs go to the work directory, never to inputs/.
"""
import glob
import json
import os
import sys
from functools import partial
from pathlib import Path

import numpy as np
import h5py
import healpy as hp

from nn_l1_sbi.paths import cosmogrid_metainfo, cosmogrid_root, data_root, tfds_dir

# ---------------------------------------------------------------- paths (ported)
# Work directory: $NN_L1_SBI_CENTERS_OUT, else $NN_L1_SBI_DATA/centers.
WORK = Path(os.environ.get("NN_L1_SBI_CENTERS_OUT") or (data_root() / "centers"))
WORK.mkdir(parents=True, exist_ok=True)
REC = str(WORK)

TFDS_NAME = "nbody_cosmogrid_dataset_tomo_cross"
TFDS_CONFIG = "grid_10deg_80px_nonoverlap180"
TFDS_VERSION = "0.0.1"
TFDS_DIR = str(tfds_dir() / TFDS_NAME / TFDS_CONFIG / TFDS_VERSION)
META = str(cosmogrid_metainfo())

# pinned by verification against the intact _bnt frozen sigma table (sigma_pix match
# to 6 decimals) and the surviving 20deg manifest
NSIDE, LMAX = 512, 1024
SIGMA_E, N_GAL = 0.26, 10.0
NOISE_SEED_BASE = 12345
FIELD_NPIX, RESO_ARCMIN = 80, 7.5
MAP_LABEL = "nobaryons"
N_AUTO = 4
N_PATCH = 180


def cg_parent():
    """Directory CONTAINING the CosmoGridV1 tree; `path_par` is relative to it.

    Same rule as nn_l1_sbi.data.build_full_sphere_cross_cache._cg_parent: $CG_PARENT
    wins, else the parent of $NN_L1_SBI_COSMOGRID.
    """
    env = os.environ.get("CG_PARENT")
    return Path(env) if env else cosmogrid_root().parent


def log(*a):
    print(*a, flush=True)


# ---------------------------------------------------------------- TFDS reference
def load_reference():
    """Return (cosmo_idx, perm, {patch: map(80,80) channel 0}) from the TFDS.

    DISABLE_SHUFFLING=True and the builder walks sorted(cache glob), so the first
    180 examples of `train` are one (cosmo, perm) group with patch 0..179.
    """
    import warnings
    warnings.filterwarnings("ignore")
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
    import tensorflow_datasets as tfds

    b = tfds.builder_from_directory(TFDS_DIR)
    log(f"  tfds: {b.info.name}/{b.info.config_name} splits={list(b.info.splits)}")
    ds = b.as_dataset(split="train", shuffle_files=False)
    ref, ci, pm = {}, None, None
    for ex in ds.take(N_PATCH + 40):
        c, p, k = int(ex["cosmo_idx"]), int(ex["perm"]), int(ex["patch"])
        if ci is None:
            ci, pm = c, p
        if (c, p) != (ci, pm):
            break
        ref[k] = np.asarray(ex["map_nbody"])[:, :, 0].astype(np.float32)
    log(f"  reference group: cosmo_idx={ci} perm={pm}  patches={len(ref)}")
    return ci, pm, ref


# ------------------------------------------------------- full-sphere channel 0
def full_sphere_channel0(cosmo_idx, perm):
    with h5py.File(META, "r") as f:
        path_par = f["parameters"]["grid"]["path_par"][cosmo_idx].decode()
    # `path_par` carries the tree's own directory name, so it is joined onto the PARENT
    # of the CosmoGridV1 root; the "raw" -> "stage3_forecast" remap is the builder's.
    rel = (path_par.replace("CosmoGrid", "CosmoGridV1")
                   .replace("raw", "stage3_forecast"))
    h5p = os.path.join(str(cg_parent()), rel.lstrip("/"),
                       f"perm_{perm:04d}", f"projected_probes_maps_{MAP_LABEL}512.h5")
    log(f"  source: {h5p}")
    if not os.path.exists(h5p):
        sys.exit(f"MISSING source h5: {h5p}")
    with h5py.File(h5p, "r") as sim:
        kg = sim["kg"]
        noiseless = np.stack([np.asarray(kg[f"stage3_lensing{b}"], dtype=np.float64)
                              for b in (1, 2, 3, 4)], axis=0)
    log(f"  noiseless {noiseless.shape} nside={hp.npix2nside(noiseless.shape[1])}")

    seed = NOISE_SEED_BASE + 100 * int(cosmo_idx) + int(perm)
    pix_area = hp.nside2pixarea(NSIDE, degrees=True) * 3600.0
    noise_std = SIGMA_E / np.sqrt(N_GAL * pix_area)
    log(f"  seed={seed}  noise_std={noise_std:.8f}")
    rng = np.random.default_rng(seed)
    noisy = noiseless + rng.normal(0.0, noise_std, size=noiseless.shape)   # full draw

    alm0 = hp.map2alm(noisy[0], lmax=LMAX, iter=0)
    return hp.alm2map(alm0, nside=NSIDE, lmax=LMAX)


# ------------------------------------------------------------------- comparison
def project(full, lon, lat, vec2pix):
    proj = hp.projector.GnomonicProj(rot=[lon, lat, 0], xsize=FIELD_NPIX,
                                     ysize=FIELD_NPIX, reso=RESO_ARCMIN)
    p = proj.projmap(full, vec2pix_func=vec2pix).astype(np.float32)
    return p - p.mean()


def main():
    log(f"  work dir : {REC}")
    log(f"  tfds     : {TFDS_DIR}")
    log(f"  cosmogrid: {cg_parent()}  (meta {META})")
    cosmo_idx, perm, ref = load_reference()
    if len(ref) < N_PATCH:
        log(f"  WARNING: only {len(ref)} patches in the first group")
    full0 = full_sphere_channel0(cosmo_idx, perm)
    vec2pix = partial(hp.vec2pix, NSIDE)

    cands = sorted(glob.glob(f"{REC}/centers_*.npy"))
    if not cands:
        sys.exit(f"no candidate geometries in {REC} -- run find_centers.py first, or copy "
                 "scripts/slurm/inputs/centers_PAPER_180.npy there to test the pinned one")
    log(f"\n  testing {len(cands)} candidate geometries")
    probe = [k for k in (0, 1, 90, 179) if k in ref] or sorted(ref)[:3]
    log(f"  probe patches: {probe}\n")

    scored = []
    for cp in cands:
        C = np.load(cp)
        if C.shape[0] != N_PATCH:
            continue
        worst = 0.0
        for k in probe:
            mine = project(full0, float(C[k, 0]), float(C[k, 1]), vec2pix)
            worst = max(worst, float(np.max(np.abs(mine - ref[k]))))
        scored.append((worst, os.path.basename(cp)[len("centers_"):-len(".npy")]))

    scored.sort()
    log("  best 8 candidates (max abs diff over probe patches):")
    for d, k in scored[:8]:
        log(f"    {k}   max|diff| = {d:.6e}")

    best_d, best_k = scored[0]
    ok = best_d < 1e-5
    log(f"\n  => {'MATCH' if ok else 'NO MATCH'}: {best_k}  max|diff|={best_d:.3e}")
    if ok and len(scored) > 1:
        log(f"     runner-up is {scored[1][0]:.3e} -- "
            f"{scored[1][0]/max(best_d,1e-30):.1e}x worse (unambiguous)")

    if ok:
        C = np.load(f"{REC}/centers_{best_k}.npy")
        np.save(f"{REC}/centers_PAPER_180.npy", C)
        srch_path = f"{REC}/centers_search.json"
        params = []
        if os.path.exists(srch_path):
            srch = json.load(open(srch_path))
            params = [x for x in srch if x["key"] == best_k]
        json.dump(dict(geometry=best_k, max_abs_diff=best_d,
                       cosmo_idx=cosmo_idx, perm=perm, probe_patches=probe,
                       params=params,
                       pinned=dict(nside=NSIDE, lmax=LMAX, sigma_e=SIGMA_E,
                                   galaxy_density=N_GAL,
                                   noise_seed_base=NOISE_SEED_BASE,
                                   field_npix=FIELD_NPIX, reso_arcmin=RESO_ARCMIN)),
                  open(f"{REC}/centers_PAPER_180.json", "w"), indent=2)
        log(f"     wrote centers_PAPER_180.npy + .json under {REC}")
        log(f"     params: {params}")
        log("     compare against scripts/slurm/inputs/centers_PAPER_180.npy; that file "
            "stays the single source of truth")
    else:
        log("     -> no candidate reproduces the TFDS; widen the search or revisit"
            " noise_seed_base / lmax / map_label before rebuilding.")


if __name__ == "__main__":
    main()
