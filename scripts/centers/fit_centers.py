#!/usr/bin/env python3
"""Step 2/5 — fit (min_separation_deg, max_abs_lat) to the 22 MEASURED patch centres.

===============================================================================
 DO NOT REGENERATE THE PATCH CENTRES.  USE inputs/centers_PAPER_180.npy.
 The selection inside `_build_non_overlapping_centers` uses an UNSTABLE argsort
 over HEALPix rings, which are blocks of exact ties, so the tiling depends on the
 numpy build: 1.24.4 and 1.26.4 share only 2 of the 180 centres.  This driver is
 shipped so the pinned file can be VERIFIED, not routinely rebuilt.
===============================================================================

measure_centers.py established the true centres[k] for 22 patch indices by bit-exact
reprojection against the TFDS. None of the 69 previously-searched geometries reproduces
all 22, so the generating parameters lie outside that grid -- but 22 exact constraints
are more than enough to fit them.

The candidate ordering inside _build_non_overlapping_centers does not depend on
min_separation_deg, and the resulting geometry is piecewise-constant in it (it only
changes when some candidate's distance-to-prefix crosses the threshold). So a coarse
scan followed by a local refinement finds the plateau containing the true value.

max_abs_lat is likely inert here: the measured centres reach only |lat| = 62.7, and the
equator-first candidate order succeeds without falling back to HEALPix order, so any cut
above the true maximum gives the same geometry. Tried explicitly rather than assumed.

Success criterion: the generated 180 must match ALL 22 measured centres to 1e-4 deg.

Input : measured_centers.json  (the work directory, else the shipped scripts/slurm/inputs/)
Output: centers_PAPER_180.npy + .json in the work directory -- NOT in inputs/.
"""
import ast
import importlib.util
import json
import os
import time
from pathlib import Path

import numpy as np
import healpy as hp

# ---------------------------------------------------------------- paths (ported)
# Outputs go to $NN_L1_SBI_CENTERS_OUT, else $NN_L1_SBI_DATA/centers.  The pinned
# scripts/slurm/inputs/centers_PAPER_180.npy is read-only as far as these tools are
# concerned and is never written to.
_spec = importlib.util.find_spec("nn_l1_sbi.data.tf_dataset_nbody_tomo")
if _spec is None or not _spec.origin:
    raise SystemExit("cannot locate nn_l1_sbi.data.tf_dataset_nbody_tomo -- "
                     "put <repo>/src on PYTHONPATH (scripts/slurm/env.sh does this)")
SRC = _spec.origin

from nn_l1_sbi.paths import data_root  # noqa: E402

WORK = Path(os.environ.get("NN_L1_SBI_CENTERS_OUT") or (data_root() / "centers"))
WORK.mkdir(parents=True, exist_ok=True)
REC = str(WORK)
SHIPPED_INPUTS = Path(__file__).resolve().parent.parent / "slurm" / "inputs"


def pinned_input(name: str) -> Path:
    """Locate a pinned input: work directory first, then the shipped inputs/ directory."""
    cands = [WORK / name]
    env_inputs = os.environ.get("NN_L1_SBI_INPUTS")
    if env_inputs:
        cands.append(Path(env_inputs) / name)
    cands.append(SHIPPED_INPUTS / name)
    for c in cands:
        if c.exists():
            return c
    raise SystemExit(f"cannot find {name}; looked in " + ", ".join(str(c) for c in cands))


N_TARGET = 180
NSIDE = 64                      # every measured centre is an nside-64 pixel centre
TOL = 1e-4

# AST-extract the original center builder (importing the module would pull jax/tf/tfds)
tree = ast.parse(open(SRC).read())
ns = {"np": np, "hp": hp}
for node in tree.body:
    if isinstance(node, ast.FunctionDef) and node.name in {
            "_angular_distance_deg", "_build_non_overlapping_centers"}:
        exec(compile(ast.Module(body=[node], type_ignores=[]), SRC, "exec"), ns)
build = ns["_build_non_overlapping_centers"]
angdist = ns["_angular_distance_deg"]

_measured_path = pinned_input("measured_centers.json")
measured = {int(k): v for k, v in json.load(open(_measured_path)).items()}
print(f"measured centres from {_measured_path}", flush=True)
print(f"fitting to {len(measured)} measured centres: indices {sorted(measured)}", flush=True)
mx = max(abs(v[1]) for v in measured.values())
print(f"max |lat| among measured = {mx:.4f}", flush=True)


def matches(C):
    """How many measured centres does this geometry reproduce, and does it get all?"""
    if C is None or C.shape[0] != N_TARGET:
        return -1
    n = 0
    for k, (lo, la) in measured.items():
        if abs(float(C[k, 0]) - lo) < TOL and abs(float(C[k, 1]) - la) < TOL:
            n += 1
    return n


def attempt(msep, mlat):
    try:
        return build(n_centers=N_TARGET, min_separation_deg=msep,
                     center_nside=NSIDE, max_abs_lat=mlat)
    except ValueError:
        return None


def scan(values, mlat, label):
    best = (-1, None, None)
    for v in values:
        C = attempt(v, mlat)
        n = matches(C)
        if n > best[0]:
            best = (n, v, C)
            print(f"    [{label}] min_sep={v:.5f} -> {n}/{len(measured)} matched", flush=True)
        if n == len(measured):
            return best
    return best


t0 = time.time()
winner = None
for mlat in (None, 90.0, 85.0, 80.0, 75.0, 70.0, 65.0):
    print(f"\n=== max_abs_lat = {mlat} ===", flush=True)
    n, v, C = scan(np.arange(13.80, 15.2001, 0.01), mlat, "coarse")
    if n == len(measured):
        winner = (mlat, v, C); break
    if n <= 0:
        print(f"    best {n}/{len(measured)}; skipping refinement", flush=True)
        continue
    # refine around the best coarse plateau
    n2, v2, C2 = scan(np.arange(max(13.5, v - 0.02), v + 0.02001, 0.0005), mlat, "fine")
    if n2 == len(measured):
        winner = (mlat, v2, C2); break
    print(f"    best for this max_abs_lat: {max(n, n2)}/{len(measured)}", flush=True)

print(f"\nelapsed {time.time()-t0:.0f}s", flush=True)

if winner is None:
    print("NO EXACT FIT. The measured centres are still the ground truth -- fall back to")
    print("assembling the geometry from measurements plus a documented valid tiling.")
    raise SystemExit(1)

mlat, msep, C = winner
lat = C[:, 1].astype(float); lon = C[:, 0].astype(float)
dmin = min(float(angdist(lon[i], lat[i], lon[:i], lat[:i]).min()) for i in range(1, len(C)))
print("=== EXACT FIT ===")
print(f"  center_nside       = {NSIDE}")
print(f"  min_separation_deg = {msep}")
print(f"  max_abs_lat        = {mlat}")
print(f"  achieved min sep   = {dmin:.5f} deg   (10deg patch diagonal = {10*np.sqrt(2):.5f})")
print(f"  |lat| max          = {np.abs(lat).max():.4f}")
print(f"  reproduces all {len(measured)}/{len(measured)} measured centres")

np.save(f"{REC}/centers_PAPER_180.npy", C)
json.dump(dict(center_nside=NSIDE, min_separation_deg=float(msep),
               max_abs_lat=(None if mlat is None else float(mlat)),
               n_centers=N_TARGET, achieved_min_sep=dmin,
               abs_lat_max=float(np.abs(lat).max()),
               measured_indices=sorted(measured), n_measured=len(measured),
               fit_tolerance_deg=TOL,
               provenance="fit to bit-exact reprojection measurements vs the cross TFDS"),
          open(f"{REC}/centers_PAPER_180.json", "w"), indent=2)
print(f"  wrote centers_PAPER_180.npy + .json under {REC}")
print("  COMPARE it against scripts/slurm/inputs/centers_PAPER_180.npy; do not replace that file.")
