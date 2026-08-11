#!/bin/bash
#SBATCH --job-name=obs_cache_10deg
#SBATCH --account=nzu@cpu
#SBATCH --partition=cpu_p1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=40
#SBATCH --hint=nomultithread
#SBATCH --time=10:00:00
#SBATCH --output=logs/obs_cache_%j.out

# Build the 10deg fiducial observation cache -- the observation set the 9000 mocks are
# drawn from. Everything downstream (all 12 Table-1 rows) reads it.
#
# Geometry and flags are NOT reconstructed here: they are copied verbatim from the
# original 10deg build script (Phase 2b), and the 180 centres it implies were confirmed
# independently by bit-exact reprojection against the cross TFDS at 22 patch indices
# (inputs/centers_PAPER_180.json).
#
#   GEO      = --field-size 10 --field-npix 80 --n-centers 180 --center-nside 64 \
#              --min-separation-deg 14.2 --max-abs-lat 75
#   COMMON   = --regime nobnt --num-workers 40
#   FIDPERMS = 0..199
#
# BNT is deliberately absent: the cache is nobnt only, and BNT is applied later
# on-device as a channel mix (--flatsky-channel-mix bnt).
#
# Remaining defaults (sigma_e 0.26, galaxy_density 10.0, nside 512, lmax 1024,
# noise_seed_base 12345, map_label nobaryons) come from the builder itself and were
# each verified against the archived _bnt frozen sigma table -- sigma_pix matches to
# six decimals, and the reprojection match could not have happened at 1e-8 unless
# the noise seed formula (base + 100*cosmo_idx + perm) were exactly right.
#
# Source data: a read-only shared copy of CosmoGridV1 ($NN_L1_SBI_COSMOGRID). The
# builder's _remap_path_par honours CG_PARENT for exactly this reason; all 200 perms of
# cosmo_fiducial must be present. Output goes to $NN_L1_SBI_DATA (bulk data, not an
# inode-capped home).
#
# Centres are PINNED via --centers-npy rather than regenerated. _build_non_overlapping_centers
# orders candidates with np.argsort(|lat|), whose default quicksort is NOT stable, and every
# HEALPix ring is a block of exact ties -- so identical arguments give different tilings under
# different numpy versions. Measured directly: numpy 1.24.4 vs 1.26.4 share only 2 of 180
# centres, and only the 1.24.4 result matches the paper (22/22 measured). Pinning the verified
# file removes that hidden dependence.

set -uo pipefail
NN_L1_SBI_ARCH=cpu
for _d in "${NN_L1_SBI_JOBS:-}" "${SLURM_SUBMIT_DIR:-}" "${SLURM_SUBMIT_DIR:-}/.." \
          "$(dirname "$0")" "$(dirname "$0")/.."; do
  [ -n "$_d" ] && [ -r "$_d/env.sh" ] && { . "$_d/env.sh"; break; }
done
[ -n "${NN_L1_SBI_ENV_SH:-}" ] || { echo "cannot find env.sh: set NN_L1_SBI_JOBS=<repo>/scripts/slurm" >&2; exit 2; }

nn_l1_sbi_need_dir NN_L1_SBI_COSMOGRID "CosmoGridV1 simulation root"

OUT=$NN_L1_SBI_DATA/full_sphere_cache_fiducial_10deg
CG=$NN_L1_SBI_COSMOGRID
CENTERS=$NN_L1_SBI_INPUTS/centers_PAPER_180.npy
nn_l1_sbi_need_path "$CENTERS" "pinned patch centres"

export CG_PARENT=$NN_L1_SBI_COSMOGRID_PARENT
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1

GEO="--field-size 10 --field-npix 80 --n-centers 180 --center-nside 64 --min-separation-deg 14.2 --max-abs-lat 75"
FIDPERMS=$(seq -s, 0 199)

echo "=== 10deg fiducial obs cache build ==="
$PY -V
echo "out : $OUT"
echo "geo : $GEO"
echo "centres: $CENTERS  (pinned -- never regenerate)"
echo "perms: 0..199 (the eval uses perm<50 for its 9000 obs; the full 200 matches the paper)"
mkdir -p "$OUT"

cd "$NN_L1_SBI_DATA"
$PY -u -m nn_l1_sbi.data.build_full_sphere_cross_cache \
    --cosmo-meta $CG/CosmoGridV1_metainfo.h5 \
    --cosmogrid-root $CG \
    --cosmo-subset fiducial --cosmo-id cosmo_fiducial \
    --realizations "$FIDPERMS" \
    $GEO \
    --centers-npy "$CENTERS" \
    --regime nobnt --num-workers 40 \
    --out-dir "$OUT"
rc=$?
echo "builder exit=$rc"
[ $rc -eq 0 ] || { echo "BUILD FAILED"; exit 1; }

echo; echo "=== verify_cache gate (ported from the original 10deg build script) ==="
$PY - "$OUT" 180 FIDUCIAL "$CENTERS" <<'PY'
import sys, glob, numpy as np
d, npat, lab, ref_npy = sys.argv[1], int(sys.argv[2]), sys.argv[3], sys.argv[4]
fs = sorted(glob.glob(d + "/nobnt/**/*.npz", recursive=True))
assert fs, f"[{lab}] no .npz produced in {d}"
z = np.load(fs[0], allow_pickle=True)
p = np.asarray(z["patches"]); c = np.asarray(z["patch_centers"])
assert p.shape == (npat, 80, 80, 10), f"[{lab}] patch shape {p.shape} != ({npat},80,80,10)"
assert np.isfinite(p).all(), f"[{lab}] non-finite patches"
assert np.abs(c[:, 1]).max() < 75, f"[{lab}] polar leak max|lat|={np.abs(c[:,1]).max():.1f}"
auto = np.abs(p[..., :4]).mean(); cross = np.abs(p[..., 4:]).mean()
assert auto > 0 and cross > 0 and cross < auto, \
    f"[{lab}] channel scales off auto={auto:.2e} cross={cross:.2e}"
print(f"[{lab}] OK: {len(fs)} files, patches{p.shape}, "
      f"max|lat|={np.abs(c[:,1]).max():.1f}, auto~{auto:.2e} cross~{cross:.2e}")

# extra gate: the centres written into the cache must equal the pinned geometry
ref = np.load(ref_npy)
assert c.shape == ref.shape, f"centre shape {c.shape} != {ref.shape}"
dmax = float(np.abs(c.astype(np.float64) - ref.astype(np.float64)).max())
print(f"[{lab}] centres vs centers_PAPER_180.npy: max|diff| = {dmax:.3e}")
assert dmax < 1e-4, "cache centres do not match the pinned geometry"
print(f"[{lab}] GEOMETRY CONFIRMED")
PY

echo; echo "=== summary ==="
echo "npz files : $(find $OUT -name '*.npz' | wc -l)  (expect 200)"
echo "size      : $(du -sh $OUT | cut -f1)"
echo "manifest  : $OUT/manifest.json"
