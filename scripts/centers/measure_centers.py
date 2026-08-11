#!/usr/bin/env python3
"""Step 3/5 — MEASURE the paper's patch centres instead of guessing the parameters.

===============================================================================
 DO NOT REGENERATE THE PATCH CENTRES.  USE inputs/centers_PAPER_180.npy.
 The parameter search cannot be trusted on its own: `_build_non_overlapping_centers`
 orders candidates with an UNSTABLE argsort over HEALPix rings (exact tie blocks),
 so the tiling depends on the numpy build.  This driver measures the centres
 directly off the TFDS, and is shipped so the pinned file can be VERIFIED.
===============================================================================

Parameter search was ambiguous: 69 distinct geometries give exactly 180 non-overlapping
10deg patches, and the best of them reproduced only 2 of 3 test patches -- so the true
min_separation_deg is near 14.142 but not equal to it.

The decisive constraint: centers are always HEALPix PIXEL CENTERS at `center_nside`.
At nside=64 that is a finite set of 49152 directions. So for any observed
(cosmo, perm, patch k) we can identify which pixel centre reproduces the stored map,
which *measures* centers[k] with no parameter guessing at all.

Two-stage matching per patch, for speed:
  1. cheap: an 8x8 gnomonic projection. For a centred grid, the 8x8 grid at the same
     reso coincides exactly with pixels [36:44, 36:44] of the 80x80 grid, so it can be
     compared against that sub-block of the stored patch. Compared after removing each
     block's own mean, since the cache demeans each patch over the full 80x80.
  2. confirm: full 80x80 projection, bit-exact.

Candidates are tried union-of-the-searched-geometries first (the true centre is very
likely among them, since those geometries nearly agree), then the full nside grid as
fallback. Run find_centers.py first: this reads its centers_*.npy from the work directory.

Noise is reproducible: seed = noise_seed_base + 100*cosmo_idx + perm.

Inputs : the cross TFDS ($NN_L1_SBI_TFDS) and CosmoGridV1 ($NN_L1_SBI_COSMOGRID)
Outputs: measured_centers.json (+ centers_PAPER_180.npy) in the work directory.
"""
import glob
import json
import os
import time
from collections import defaultdict
from functools import partial
from pathlib import Path

import numpy as np
import h5py
import healpy as hp

from nn_l1_sbi.paths import cosmogrid_metainfo, cosmogrid_root, data_root, tfds_dir

# ---------------------------------------------------------------- paths (ported)
# Work directory: $NN_L1_SBI_CENTERS_OUT, else $NN_L1_SBI_DATA/centers.  The pinned
# scripts/slurm/inputs/centers_PAPER_180.npy is never written to.
WORK = Path(os.environ.get("NN_L1_SBI_CENTERS_OUT") or (data_root() / "centers"))
WORK.mkdir(parents=True, exist_ok=True)
REC = str(WORK)

# The cross TFDS the paper trained on, under $NN_L1_SBI_TFDS (else $TFDS_DATA_DIR).
TFDS_NAME = "nbody_cosmogrid_dataset_tomo_cross"
TFDS_CONFIG = "grid_10deg_80px_nonoverlap180"
TFDS_VERSION = "0.0.1"
TFDS_DIR = str(tfds_dir() / TFDS_NAME / TFDS_CONFIG / TFDS_VERSION)
META = str(cosmogrid_metainfo())

NSIDE, LMAX = 512, 1024
SIGMA_E, N_GAL, NOISE_SEED_BASE = 0.26, 10.0, 12345
FIELD_NPIX, RESO_ARCMIN, N_AUTO = 80, 7.5, 4
CENTER_NSIDES = [64, 32, 16]        # nside=64 strongly favoured by the partial match
N_GROUPS = int(os.environ.get("N_GROUPS", "12"))
SUB = slice(36, 44)                  # central 8x8 of the 80x80 grid


def cg_parent():
    """Directory CONTAINING the CosmoGridV1 tree; `path_par` is relative to it.

    Same rule as nn_l1_sbi.data.build_full_sphere_cross_cache._cg_parent: $CG_PARENT
    wins, else the parent of $NN_L1_SBI_COSMOGRID (path_par already carries the tree's
    own directory name).
    """
    env = os.environ.get("CG_PARENT")
    return Path(env) if env else cosmogrid_root().parent


def log(*a):
    print(*a, flush=True)


def read_shard(idx=0):
    import warnings
    warnings.filterwarnings("ignore")
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
    import tensorflow as tf
    import tensorflow_datasets as tfds
    b = tfds.builder_from_directory(TFDS_DIR)
    shards = sorted(glob.glob(f"{TFDS_DIR}/*train.tfrecord-*"))
    parser = b.info.features.deserialize_example_np
    groups = defaultdict(dict)
    for raw in tf.data.TFRecordDataset([shards[idx]]):
        ex = parser(raw.numpy())
        groups[(int(ex["cosmo_idx"]), int(ex["perm"]))][int(ex["patch"])] = \
            np.asarray(ex["map_nbody"])[:, :, 0].astype(np.float32)
    log(f"  shard {idx}: {sum(len(v) for v in groups.values())} examples, {len(groups)} groups")
    return groups


def full_sphere_channel0(cosmo_idx, perm):
    with h5py.File(META, "r") as f:
        path_par = f["parameters"]["grid"]["path_par"][cosmo_idx].decode()
    rel = path_par.replace("CosmoGrid", "CosmoGridV1").replace("raw", "stage3_forecast")
    h5p = os.path.join(str(cg_parent()), rel.lstrip("/"), f"perm_{perm:04d}",
                       "projected_probes_maps_nobaryons512.h5")
    if not os.path.exists(h5p):
        return None
    with h5py.File(h5p, "r") as sim:
        kg = sim["kg"]
        noiseless = np.stack([np.asarray(kg[f"stage3_lensing{b}"], dtype=np.float64)
                              for b in (1, 2, 3, 4)], axis=0)
    seed = NOISE_SEED_BASE + 100 * int(cosmo_idx) + int(perm)
    nstd = SIGMA_E / np.sqrt(N_GAL * hp.nside2pixarea(NSIDE, degrees=True) * 3600.0)
    noisy = noiseless + np.random.default_rng(seed).normal(0.0, nstd, size=noiseless.shape)
    return hp.alm2map(hp.map2alm(noisy[0], lmax=LMAX, iter=0), nside=NSIDE, lmax=LMAX)


def proj(full, lon, lat, npix, vec2pix):
    p = hp.projector.GnomonicProj(rot=[lon, lat, 0], xsize=npix, ysize=npix,
                                  reso=RESO_ARCMIN)
    return p.projmap(full, vec2pix_func=vec2pix).astype(np.float32)


def candidate_pool():
    """Union of the searched geometries first, then full nside grids."""
    pool, seen = [], set()
    for f in sorted(glob.glob(f"{REC}/centers_*.npy")):
        try:
            C = np.load(f)
        except Exception:
            continue
        if C.ndim != 2 or C.shape[1] != 2:
            continue
        for lon, lat in C:
            k = (round(float(lon), 6), round(float(lat), 6))
            if k not in seen:
                seen.add(k); pool.append(k)
    log(f"  stage-A pool (union of searched geometries): {len(pool)} unique centres")
    grids = {}
    for ns in CENTER_NSIDES:
        th, ph = hp.pix2ang(ns, np.arange(hp.nside2npix(ns)))
        grids[ns] = np.stack([np.degrees(ph) - 180.0, 90.0 - np.degrees(th)], axis=1)
    return np.array(pool, dtype=np.float64), grids


def match_patch(full, ref, pool, grids, vec2pix):
    """Return (lon, lat, max_abs_diff, stage) for the centre reproducing `ref`."""
    tgt8 = ref[SUB, SUB].astype(np.float64); tgt8 = tgt8 - tgt8.mean()

    def scan(cands, tol8=2e-3):
        best = []
        for lon, lat in cands:
            p8 = proj(full, float(lon), float(lat), 8, vec2pix).astype(np.float64)
            p8 = p8 - p8.mean()
            d = np.max(np.abs(p8 - tgt8))
            if d < tol8:
                best.append((d, float(lon), float(lat)))
        return sorted(best)[:5]

    for stage, cands in [("A:union", pool)] + [(f"B:nside{ns}", grids[ns]) for ns in CENTER_NSIDES]:
        hits = scan(cands)
        for _, lon, lat in hits:
            p80 = proj(full, lon, lat, FIELD_NPIX, vec2pix)
            p80 = p80 - p80.mean()
            d80 = float(np.max(np.abs(p80 - ref)))
            if d80 < 1e-5:
                return lon, lat, d80, stage
    return None, None, None, None


def main():
    log(f"  work dir : {REC}")
    log(f"  tfds     : {TFDS_DIR}")
    log(f"  cosmogrid: {cg_parent()}  (meta {META})")
    vec2pix = partial(hp.vec2pix, NSIDE)
    pool, grids = candidate_pool()
    groups = read_shard(0)
    ranked = sorted(groups.items(), key=lambda kv: -len(kv[1]))[:N_GROUPS]
    log(f"  using {len(ranked)} groups, {sum(len(v) for _, v in ranked)} patches total\n")

    measured, t0 = {}, time.time()
    for (ci, pm), ref in ranked:
        full = full_sphere_channel0(ci, pm)
        if full is None:
            log(f"  cosmo {ci} perm {pm}: source h5 missing, skipped"); continue
        for k in sorted(ref):
            lon, lat, d, stage = match_patch(full, ref[k], pool, grids, vec2pix)
            if lon is None:
                log(f"  patch {k:3d} (cosmo {ci} perm {pm}): NO MATCH")
            else:
                if k in measured and (abs(measured[k][0] - lon) > 1e-6 or
                                      abs(measured[k][1] - lat) > 1e-6):
                    log(f"  patch {k:3d}: INCONSISTENT with earlier measurement!")
                measured[k] = (lon, lat)
                log(f"  patch {k:3d}: lon={lon:9.4f} lat={lat:9.4f} "
                    f"|diff|={d:.2e} [{stage}] ({time.time()-t0:.0f}s)")

    log(f"\n  MEASURED {len(measured)} distinct patch centres")
    json.dump({str(k): list(v) for k, v in sorted(measured.items())},
              open(f"{REC}/measured_centers.json", "w"), indent=2)

    # which of the searched geometries agrees with every measured centre?
    log("\n  === geometries consistent with all measured centres ===")
    ok = []
    for f in sorted(glob.glob(f"{REC}/centers_*.npy")):
        try:
            C = np.load(f)
        except Exception:
            continue
        if C.ndim != 2 or C.shape[0] != 180:
            continue
        if all(abs(float(C[k, 0]) - lo) < 1e-4 and abs(float(C[k, 1]) - la) < 1e-4
               for k, (lo, la) in measured.items()):
            ok.append(os.path.basename(f)[len("centers_"):-len(".npy")])
    if ok:
        srch = json.load(open(f"{REC}/centers_search.json"))
        for key in ok:
            log(f"    {key}: {[ (x['nside'], x['max_min_sep'], x['max_abs_lat']) for x in srch if x['key']==key ][:3]}")
        np.save(f"{REC}/centers_PAPER_180.npy", np.load(f"{REC}/centers_{ok[0]}.npy"))
        log(f"  -> centers_PAPER_180.npy set from {ok[0]}  (in {REC}; compare against "
            f"scripts/slurm/inputs/centers_PAPER_180.npy, do not replace it)")
    else:
        log("    none -- the true parameters are outside the searched grid.")
        log("    Fall back: derive min_sep/max_abs_lat from the measured centres directly.")
    log(f"  wrote measured_centers.json ({len(measured)} entries)")


if __name__ == "__main__":
    main()
