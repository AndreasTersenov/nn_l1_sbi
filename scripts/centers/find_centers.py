#!/usr/bin/env python3
"""Step 1/5 — search the parameter space of the 180-patch centre configuration.

===============================================================================
 DO NOT REGENERATE THE PATCH CENTRES.  USE inputs/centers_PAPER_180.npy.
 `_build_non_overlapping_centers` orders candidates with `np.argsort`, which is
 quicksort and therefore UNSTABLE, and every HEALPix ring is a block of exact
 ties: numpy 1.24.4 and 1.26.4, given identical parameters, produce tilings that
 share only 2 of the 180 centres.  The geometry is not reproducible from the
 parameters alone, which is why it is pinned.  These five drivers are shipped so
 the pinned file can be VERIFIED, not routinely rebuilt — see README.md here and
 scripts/slurm/inputs/README.md.
===============================================================================

The (n_centers, min_separation_deg, center_nside, max_abs_lat) quadruple was never
recorded -- the 10deg cache has no manifest. But `_build_non_overlapping_centers` is
deterministic for a fixed numpy, so the space can be searched.

Reformulation: the number of centers achievable is monotone-decreasing in min_sep, so
for each (center_nside, max_abs_lat) there is a maximal min_sep that still yields 180.
A designer picks the largest separation that still fits the required patch count (that
is what the 20deg config did: 48 patches at 28.5 deg, just above the 28.28 deg diagonal
of a 20deg patch). So we bisect for that maximum per (nside, max_abs_lat) instead of
scanning min_sep blindly, then group by the resulting geometry.

Reference point from the surviving 20deg manifest:
    n_centers=48, min_sep=28.5, center_nside=32, max_abs_lat=None
The natural 10deg analogue is min_sep = 28.5/2 = 14.25 (10deg patch diagonal = 14.14).

The center-building code is AST-extracted from `nn_l1_sbi.data.tf_dataset_nbody_tomo`
so that the original implementation runs verbatim rather than a retyped copy; extracting
rather than importing keeps TensorFlow, TFDS and JAX out of the process.

Outputs (all under the work directory, never inside the repository):
    centers_<geomkey>.npy    one file per distinct geometry found
    centers_search.json      the parameter combos behind each geometry
"""
import ast
import hashlib
import importlib.util
import json
import os
import time
from pathlib import Path

import numpy as np
import healpy as hp

# ---------------------------------------------------------------- paths (ported)
# The original ran from a flat directory with absolute paths baked in.  Here the module
# under test is located through the installed package and the outputs go to a work
# directory: $NN_L1_SBI_CENTERS_OUT, else $NN_L1_SBI_DATA/centers.  Nothing is ever
# written next to the pinned inputs/centers_PAPER_180.npy.
_spec = importlib.util.find_spec("nn_l1_sbi.data.tf_dataset_nbody_tomo")
if _spec is None or not _spec.origin:
    raise SystemExit("cannot locate nn_l1_sbi.data.tf_dataset_nbody_tomo -- "
                     "put <repo>/src on PYTHONPATH (scripts/slurm/env.sh does this)")
SRC = _spec.origin

from nn_l1_sbi.paths import data_root  # noqa: E402  (after the spec lookup on purpose)

WORK = Path(os.environ.get("NN_L1_SBI_CENTERS_OUT") or (data_root() / "centers"))
WORK.mkdir(parents=True, exist_ok=True)
OUT = str(WORK)

print("=" * 79)
print("  These centres are PINNED in scripts/slurm/inputs/centers_PAPER_180.npy.")
print("  Regeneration is not reproducible (unstable argsort over HEALPix tie blocks).")
print("  This search writes candidates for VERIFICATION only, into:")
print(f"    {OUT}")
print("=" * 79, flush=True)

WANT = {"_angular_distance_deg", "_build_non_overlapping_centers"}

tree = ast.parse(open(SRC).read())
ns = {"np": np, "hp": hp}
picked = []
for node in tree.body:
    if isinstance(node, ast.FunctionDef) and node.name in WANT:
        exec(compile(ast.Module(body=[node], type_ignores=[]), SRC, "exec"), ns)
        picked.append(node.name)
assert WANT.issubset(set(picked)), f"missing: {WANT - set(picked)}"
build = ns["_build_non_overlapping_centers"]
angdist = ns["_angular_distance_deg"]
print(f"extracted verbatim from {SRC}: {sorted(picked)}", flush=True)

N = 180
NSIDES = [16, 32, 64]
MAX_LATS = [None, 85.0, 84.0, 80.0, 78.0, 75.0, 70.0, 65.0, 60.0]


def works(msep, nside, mlat):
    try:
        c = build(n_centers=N, min_separation_deg=msep, center_nside=nside, max_abs_lat=mlat)
    except ValueError:
        return None
    return c if c.shape[0] == N else None


def min_pairwise(c):
    lon, lat = c[:, 0].astype(float), c[:, 1].astype(float)
    d = np.inf
    for i in range(1, len(c)):
        d = min(d, angdist(lon[i], lat[i], lon[:i], lat[:i]).min())
    return d


def key_of(c):
    return hashlib.sha256(np.ascontiguousarray(c, dtype=np.float32).tobytes()).hexdigest()[:16]


results = []
t0 = time.time()
for nside in NSIDES:
    for mlat in MAX_LATS:
        # bisect for the largest min_sep that still yields exactly N centers
        lo, hi = 5.0, 40.0                      # lo works, hi fails (assumed)
        if works(lo, nside, mlat) is None:
            print(f"  nside={nside} maxlat={mlat}: cannot reach {N} even at min_sep={lo}", flush=True)
            continue
        if works(hi, nside, mlat) is not None:
            lo = hi
        else:
            for _ in range(24):                 # ~1e-5 deg resolution
                mid = 0.5 * (lo + hi)
                if works(mid, nside, mlat) is not None:
                    lo = mid
                else:
                    hi = mid
        c = works(lo, nside, mlat)
        results.append(dict(nside=nside, max_abs_lat=mlat, max_min_sep=round(lo, 5),
                            key=key_of(c), achieved_min_sep=round(min_pairwise(c), 5),
                            abs_lat_max=round(float(np.abs(c[:, 1]).max()), 5)))
        r = results[-1]
        print(f"  nside={nside:3d} maxlat={str(mlat):>5s} -> max min_sep={r['max_min_sep']:8.4f}"
              f"  achieved={r['achieved_min_sep']:8.4f}  |lat|max={r['abs_lat_max']:7.3f}"
              f"  geom={r['key']}", flush=True)
        np.save(f"{OUT}/centers_{r['key']}.npy", c)

print(f"\nbisection done in {time.time()-t0:.0f}s", flush=True)

# Also evaluate the a-priori natural choice: the 20deg config scaled by 1/2.
print("\n=== a-priori candidates (20deg config scaled) ===", flush=True)
for nside in NSIDES:
    for mlat in MAX_LATS:
        for msep in (14.25, 14.142, 14.14):
            c = works(msep, nside, mlat)
            if c is not None:
                k = key_of(c)
                print(f"  nside={nside:3d} maxlat={str(mlat):>5s} min_sep={msep:7.3f}"
                      f" -> OK geom={k} |lat|max={float(np.abs(c[:,1]).max()):7.3f}", flush=True)
                np.save(f"{OUT}/centers_{k}.npy", c)
                results.append(dict(nside=nside, max_abs_lat=mlat, max_min_sep=msep,
                                    key=k, achieved_min_sep=round(min_pairwise(c), 5),
                                    abs_lat_max=round(float(np.abs(c[:, 1]).max()), 5),
                                    apriori=True))

keys = sorted({r["key"] for r in results})
print(f"\nDISTINCT geometries found: {len(keys)}")
for k in keys:
    who = [r for r in results if r["key"] == k]
    print(f"  {k}: {len(who)} param combos, e.g. nside={who[0]['nside']} "
          f"maxlat={who[0]['max_abs_lat']} min_sep={who[0]['max_min_sep']}")
json.dump(results, open(f"{OUT}/centers_search.json", "w"), indent=2)
print(f"\nwrote {OUT}/centers_search.json")
