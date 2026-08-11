#!/bin/bash
#SBATCH --job-name=freeze_sigma_nobnt
#SBATCH --account=nzu@cpu
#SBATCH --partition=cpu_p1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --hint=nomultithread
#SBATCH --time=04:00:00
#SBATCH --output=logs/freeze_sigma_%j.out

# Generate flatsky_cross_noise_sigma.npz -- the NO-BNT frozen per-(channel,scale)
# noise sigma table, which is the SNR denominator for the flat-local L1 binning.
#
# The pinned parameters are:
#     n_real 48, n_scales 5, seed_base 987654321, op both (16 channels)
# and they are cross-checked below against the BNT twin ($NN_L1_SBI_SIGMA_BNT): BNT
# only mixes channels, so every metadata field must agree exactly. Only the four
# no-BNT rows depend on this file; the BNT rows use the twin directly.
#
# The script reads its physical constants straight out of the fiducial cache npz
# ("so they cannot drift"), which is why it must run after build_obs_cache.slurm.

set -uo pipefail
NN_L1_SBI_ARCH=cpu
for _d in "${NN_L1_SBI_JOBS:-}" "${SLURM_SUBMIT_DIR:-}" "${SLURM_SUBMIT_DIR:-}/.." \
          "$(dirname "$0")" "$(dirname "$0")/.."; do
  [ -n "$_d" ] && [ -r "$_d/env.sh" ] && { . "$_d/env.sh"; break; }
done
[ -n "${NN_L1_SBI_ENV_SH:-}" ] || { echo "cannot find env.sh: set NN_L1_SBI_JOBS=<repo>/scripts/slurm" >&2; exit 2; }

nn_l1_sbi_need_dir  NN_L1_SBI_COSMOGRID "CosmoGridV1 simulation root"
nn_l1_sbi_need_file NN_L1_SBI_SIGMA_BNT "frozen BNT noise-sigma table (the cross-check twin)"

OUT=$NN_L1_SBI_DATA/sigma_tables
mkdir -p "$OUT"

export CG_PARENT=$NN_L1_SBI_COSMOGRID_PARENT
export FLATSKY_FID_CACHE=$NN_L1_SBI_DATA/full_sphere_cache_fiducial_10deg/nobnt/obs
export FLATSKY_FID_H5=${NN_L1_SBI_FID_H5:-$NN_L1_SBI_COSMOGRID/stage3_forecast/fiducial/cosmo_fiducial/perm_0000/projected_probes_maps_nobaryons512.h5}
export FLATSKY_OUT_DIR=$OUT
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
nn_l1_sbi_need_path "$FLATSKY_FID_H5" "CosmoGridV1 fiducial perm_0000 map (override with NN_L1_SBI_FID_H5)"

echo "=== generate no-BNT frozen sigma table ==="
echo "cache: $FLATSKY_FID_CACHE  ($(ls $FLATSKY_FID_CACHE/*.npz 2>/dev/null | wc -l) npz)"
echo "out  : $OUT"

cd "$NN_L1_SBI_DATA"
$PY -u -m nn_l1_sbi.flatsky.freeze_flatsky_cross_noise \
    --op both --n-real 48 --n-scales 5 --seed-base 987654321 \
    --workers 32 --out-dir "$OUT"
rc=$?
echo "exit=$rc"
[ $rc -eq 0 ] || { echo "FREEZE FAILED"; exit 1; }

echo; echo "=== cross-check the new no-BNT table against the BNT twin ==="
$PY - "$OUT" "$NN_L1_SBI_SIGMA_BNT" <<'PY'
import numpy as np, glob, os, sys
OUT, TWIN = sys.argv[1], sys.argv[2]
NEW = glob.glob(os.path.join(OUT, "flatsky_cross_noise_sigma*.npz"))
assert NEW, "no sigma table produced"
new = [p for p in NEW if "_bnt" not in p and "_whiten" not in p]
assert new, f"no NO-BNT table among {[os.path.basename(p) for p in NEW]}"
a = np.load(new[0], allow_pickle=True); b = np.load(TWIN, allow_pickle=True)
print(f"  new : {os.path.basename(new[0])}")
# metadata must match the twin exactly (BNT only mixes channels; it cannot change these)
same = True
for k in ("n_scales","n_real","nside","lmax","sigma_e","galaxy_density",
          "reso_arcmin","field_npix","seed_base"):
    va, vb = a[k], b[k]
    ok = np.allclose(np.asarray(va, float), np.asarray(vb, float))
    same &= ok
    print(f"  {k:16s} new={va}  twin={vb}   {'OK' if ok else 'MISMATCH'}")
print(f"  channel_names match: {list(a['channel_names'])==list(b['channel_names'])}")
print(f"  sigma shape: new={a['sigma'].shape} twin={b['sigma'].shape}")
print(f"  bnt flag   : new={a['bnt']} (expect False)  twin={b['bnt']} (expect True)")
s = np.asarray(a["sigma"], float)
print(f"  sigma finite/positive: {np.isfinite(s).all()} / {(s>0).all()}")
assert same, "metadata disagrees with the BNT twin"
assert np.isfinite(s).all() and (s > 0).all(), "bad sigma values"
assert not bool(a["bnt"]), "new table is flagged bnt=True"
print("\n  NO-BNT SIGMA TABLE GENERATED AND CONSISTENT")
PY
