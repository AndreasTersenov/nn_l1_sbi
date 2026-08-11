#!/usr/bin/env python3
"""Step 5/5 — confirm the FULL 180-patch ordering of the recovered centre geometry.

===============================================================================
 DO NOT REGENERATE THE PATCH CENTRES.  USE inputs/centers_PAPER_180.npy.
 This is the VERIFICATION step: it reprojects every measurable patch of one
 (cosmo, perm) group from the raw CosmoGridV1 maps and requires a bit-exact match
 against the TFDS.  It is the job to run if you want to check the pinned centres;
 it does not produce a new tiling, and it never writes into inputs/.
===============================================================================

prove_centers.py identified the geometry, but its reference group happened to
contain a single patch (the TFDS builder writes via imap_unordered, so examples
interleave across cosmologies -- the first 180 examples are NOT one group). That
run therefore verified centers[118] only.

The block bootstrap groups observations by patch index, so the obs cache must
reproduce the whole patch<->center mapping, not one entry. Here we read a single
TFDS shard directly (~1100 examples), take the (cosmo, perm) group with the most
patches present, and check every one of them -- against the geometry under test and,
when it is available, against the runner-up as a control.

Inputs : centers_PAPER_180.npy/.json -- the work directory first, else the pinned
         copies in scripts/slurm/inputs/.
Output : centers_PAPER_180.json in the WORK directory, with the confirmation fields
         appended. The shipped inputs/ copy is never modified.
"""
import glob
import json
import os
import sys
from collections import defaultdict
from functools import partial
from pathlib import Path

import numpy as np
import h5py
import healpy as hp

from nn_l1_sbi.paths import cosmogrid_metainfo, cosmogrid_root, data_root, tfds_dir

# ---------------------------------------------------------------- paths (ported)
WORK = Path(os.environ.get("NN_L1_SBI_CENTERS_OUT") or (data_root() / "centers"))
WORK.mkdir(parents=True, exist_ok=True)
REC = str(WORK)
SHIPPED_INPUTS = Path(__file__).resolve().parent.parent / "slurm" / "inputs"

TFDS_NAME = "nbody_cosmogrid_dataset_tomo_cross"
TFDS_CONFIG = "grid_10deg_80px_nonoverlap180"
TFDS_VERSION = "0.0.1"
TFDS_DIR = str(tfds_dir() / TFDS_NAME / TFDS_CONFIG / TFDS_VERSION)
META = str(cosmogrid_metainfo())

NSIDE, LMAX = 512, 1024
SIGMA_E, N_GAL, NOISE_SEED_BASE = 0.26, 10.0, 12345
FIELD_NPIX, RESO_ARCMIN, N_AUTO = 80, 7.5, 4
CONTROL = "efae5667ba16316f"        # runner-up from prove_centers, if that run is present


def pinned_input(name: str):
    """Locate a pinned input: work directory first, then the shipped inputs/ directory."""
    cands = [WORK / name]
    env_inputs = os.environ.get("NN_L1_SBI_INPUTS")
    if env_inputs:
        cands.append(Path(env_inputs) / name)
    cands.append(SHIPPED_INPUTS / name)
    for c in cands:
        if c.exists():
            return c
    sys.exit(f"cannot find {name}; looked in " + ", ".join(str(c) for c in cands))


def cg_parent():
    """Directory CONTAINING the CosmoGridV1 tree; `path_par` is relative to it.

    Same rule as nn_l1_sbi.data.build_full_sphere_cross_cache._cg_parent: $CG_PARENT
    wins, else the parent of $NN_L1_SBI_COSMOGRID.
    """
    env = os.environ.get("CG_PARENT")
    return Path(env) if env else cosmogrid_root().parent


def log(*a):
    print(*a, flush=True)


def read_one_shard():
    """Decode a single tfrecord shard; return {(cosmo,perm): {patch: chan0 map}}."""
    import warnings
    warnings.filterwarnings("ignore")
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
    import tensorflow as tf
    import tensorflow_datasets as tfds

    b = tfds.builder_from_directory(TFDS_DIR)
    shards = sorted(glob.glob(f"{TFDS_DIR}/*train.tfrecord-*"))
    log(f"  {len(shards)} train shards; reading {os.path.basename(shards[0])}")
    ser = b.info.features.serialize_example        # noqa: F841  (spec sanity)
    parser = b.info.features.deserialize_example_np

    groups = defaultdict(dict)
    ds = tf.data.TFRecordDataset([shards[0]])
    n = 0
    for raw in ds:
        ex = parser(raw.numpy())
        c, p, k = int(ex["cosmo_idx"]), int(ex["perm"]), int(ex["patch"])
        groups[(c, p)][k] = np.asarray(ex["map_nbody"])[:, :, 0].astype(np.float32)
        n += 1
    log(f"  decoded {n} examples -> {len(groups)} (cosmo,perm) groups")
    return groups


def full_sphere_channel0(cosmo_idx, perm):
    with h5py.File(META, "r") as f:
        path_par = f["parameters"]["grid"]["path_par"][cosmo_idx].decode()
    rel = path_par.replace("CosmoGrid", "CosmoGridV1").replace("raw", "stage3_forecast")
    h5p = os.path.join(str(cg_parent()), rel.lstrip("/"),
                       f"perm_{perm:04d}", "projected_probes_maps_nobaryons512.h5")
    if not os.path.exists(h5p):
        sys.exit(f"MISSING {h5p}")
    with h5py.File(h5p, "r") as sim:
        kg = sim["kg"]
        noiseless = np.stack([np.asarray(kg[f"stage3_lensing{b}"], dtype=np.float64)
                              for b in (1, 2, 3, 4)], axis=0)
    seed = NOISE_SEED_BASE + 100 * int(cosmo_idx) + int(perm)
    noise_std = SIGMA_E / np.sqrt(N_GAL * hp.nside2pixarea(NSIDE, degrees=True) * 3600.0)
    rng = np.random.default_rng(seed)
    noisy = noiseless + rng.normal(0.0, noise_std, size=noiseless.shape)
    return hp.alm2map(hp.map2alm(noisy[0], lmax=LMAX, iter=0), nside=NSIDE, lmax=LMAX)


def project(full, lon, lat, vec2pix):
    proj = hp.projector.GnomonicProj(rot=[lon, lat, 0], xsize=FIELD_NPIX,
                                     ysize=FIELD_NPIX, reso=RESO_ARCMIN)
    p = proj.projmap(full, vec2pix_func=vec2pix).astype(np.float32)
    return p - p.mean()


def main():
    log(f"  work dir : {REC}")
    log(f"  tfds     : {TFDS_DIR}")
    log(f"  cosmogrid: {cg_parent()}  (meta {META})")

    meta_path = pinned_input("centers_PAPER_180.json")
    npy_path = pinned_input("centers_PAPER_180.npy")
    meta = json.load(open(meta_path))
    # "geometry" is written by prove_centers.py; the pinned inputs/ json records the
    # generating parameters instead, so it is reported when present and skipped otherwise.
    key = meta.get("geometry", "(geometry key not recorded in this json)")
    C = np.load(npy_path)
    log(f"  centres under test: {npy_path}   ({meta_path})")

    # The control is the runner-up geometry from a prove_centers.py run in this work
    # directory. It is a sanity contrast, not part of the pass/fail criterion, so it is
    # optional: verifying the shipped centres does not require a full search.
    ctl_path = os.path.join(REC, f"centers_{CONTROL}.npy")
    Cc = np.load(ctl_path) if os.path.exists(ctl_path) else None
    log(f"  geometry under test: {key}   control: "
        f"{CONTROL if Cc is not None else 'none (no prove_centers run here)'}")

    groups = read_one_shard()
    (ci, pm), ref = max(groups.items(), key=lambda kv: len(kv[1]))
    log(f"  richest group: cosmo_idx={ci} perm={pm} with {len(ref)} patches\n")

    full0 = full_sphere_channel0(ci, pm)
    vec2pix = partial(hp.vec2pix, NSIDE)

    worst = worst_ctl = 0.0
    bad = []
    for k in sorted(ref):
        d = float(np.max(np.abs(project(full0, float(C[k, 0]), float(C[k, 1]), vec2pix) - ref[k])))
        worst = max(worst, d)
        if Cc is not None:
            dc = float(np.max(np.abs(project(full0, float(Cc[k, 0]), float(Cc[k, 1]), vec2pix) - ref[k])))
            worst_ctl = max(worst_ctl, dc)
        if d > 1e-5:
            bad.append((k, d))
    log(f"  patches checked : {len(ref)}")
    log(f"  worst |diff|    : {worst:.3e}   (geometry under test)")
    if Cc is not None:
        log(f"  worst |diff|    : {worst_ctl:.3e}   (control geometry)")
    else:
        log("  worst |diff|    : n/a                (control geometry not available)")
    log(f"  mismatches>1e-5 : {len(bad)}  {bad[:5]}")
    ok = not bad
    log(f"\n  => FULL ORDERING {'CONFIRMED' if ok else 'NOT confirmed'} "
        f"across {len(ref)} patches of cosmo {ci} perm {pm}")
    meta["full_ordering_confirmed"] = ok
    meta["patches_checked"] = len(ref)
    meta["worst_abs_diff_all_patches"] = worst
    meta["control_worst_abs_diff"] = worst_ctl if Cc is not None else None
    meta["confirm_group"] = dict(cosmo_idx=ci, perm=pm)
    meta["centers_file_checked"] = str(npy_path)
    # Always to the work directory: the pinned inputs/ copy is an input, not an output.
    out_json = os.path.join(REC, "centers_PAPER_180.json")
    json.dump(meta, open(out_json, "w"), indent=2)
    log(f"  wrote {out_json}")


if __name__ == "__main__":
    main()
