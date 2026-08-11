#!/bin/bash
#SBATCH --job-name=l1both_build_bnt
#SBATCH --account=nzu@a100
#SBATCH --constraint=a100
#SBATCH --qos=qos_gpu_a100-dev
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=32
#SBATCH --hint=nomultithread
#SBATCH --time=02:00:00
#SBATCH --output=logs/l1both_bnt_%j.out

# STEP 1 of the L1 campaign, BNT frame: build the 'both' (16ch) training datavector ONCE.
# The BNT twin of train_l1both_build.slurm -- same driver, same flags, three differences:
#   --flatsky-channel-mix bnt        BNT applied on-device (noise -> demean -> BNT ->
#                                    cross-build -> whiten); the observation cache stays
#                                    the no-BNT one, see README.md sec 5
#   --flatsky-cross-sigma            the frozen BNT sigma table ($NN_L1_SBI_SIGMA_BNT)
#   --cache-dir  l1_datavector_cache_bnt
# It writes flat_local_ranges.npy for the BNT frame, which setup/bnt_obs_and_slice.slurm
# column-slices into the product (10ch) and none (4ch) BNT caches.
#
# --fiducial-summaries-out is deliberately NOT used: on the flat_local route that dump
# bins with representative SCALAR SNR ranges while obs_l1 uses the PER-CHANNEL
# flat_local_ranges, so the driver's own G1 gate rejects it (correctly). Observations
# come from setup/bnt_obs_and_slice.slurm instead.
# Every flag below is taken from the archived posterior.meta.json of the original arm,
# NOT from the earlier 20deg campaign's recipe -- that one is wrong for these rows
# (wrong SNR range, wrong batch size, wrong route).
#
# Two inputs are passed explicitly rather than left to the driver:
#   --cosmogrid-meta          (resolved here from $NN_L1_SBI_COSMOGRID)
#   --fiducial-obs-cache-dir  (no default; must be the cache ROOT -- the loader appends
#                              regime/obs itself, so passing .../nobnt/obs would break)
#
# The datavector cache is deliberately SHARED across seeds: the training datavector comes
# from the fixed cross TFDS (noise baked in at TFDS build time), so it does not depend on
# --seed. Seeds 42/43 reuse it and skip the expensive loader pass.

set -uo pipefail
NN_L1_SBI_ARCH=gpu
for _d in "${NN_L1_SBI_JOBS:-}" "${SLURM_SUBMIT_DIR:-}" "${SLURM_SUBMIT_DIR:-}/.." \
          "$(dirname "$0")" "$(dirname "$0")/.."; do
  [ -n "$_d" ] && [ -r "$_d/env.sh" ] && { . "$_d/env.sh"; break; }
done
[ -n "${NN_L1_SBI_ENV_SH:-}" ] || { echo "cannot find env.sh: set NN_L1_SBI_JOBS=<repo>/scripts/slurm" >&2; exit 2; }

nn_l1_sbi_need_dir  NN_L1_SBI_TFDS      "cross TFDS root"
nn_l1_sbi_need_dir  NN_L1_SBI_COSMOGRID "CosmoGridV1 simulation root"
nn_l1_sbi_need_file NN_L1_SBI_SIGMA_BNT "frozen BNT noise-sigma table"

SEED=41
OUT=$NN_L1_SBI_DATA/arms/l1both_bnt_s${SEED}
DVCACHE=$NN_L1_SBI_DATA/l1_datavector_cache_bnt      # shared across seeds
SIGMA=$NN_L1_SBI_SIGMA_BNT
OBSCACHE=$NN_L1_SBI_DATA/full_sphere_cache_fiducial_10deg
CGMETA=$NN_L1_SBI_COSMOGRID/CosmoGridV1_metainfo.h5
TFDS=$NN_L1_SBI_TFDS

mkdir -p "$OUT" "$DVCACHE"
export TFDS_DATA_DIR=$TFDS
export CG_PARENT=$NN_L1_SBI_COSMOGRID_PARENT
export TF_CPP_MIN_LOG_LEVEL=3
export XLA_PYTHON_CLIENT_PREALLOCATE=false

echo "=== BUILD: l1 +conv+product (--cross-op both, 16ch), BNT frame, seed $SEED ==="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null
$PY -c "import jax;print('  jax devices:',jax.devices())" 2>/dev/null
echo "  sigma : $SIGMA"
echo "  obs   : $OBSCACHE   (no-BNT cache: BNT is applied on-device)"
echo "  dvcache: $DVCACHE (shared across seeds)"

cd "$NN_L1_SBI_DATA"
srun $PY -u -m nn_l1_sbi.inference.npe_l1norm_cross_jaxili_nbody_tomo \
    --cross-maps-route flat_local --cross-op both \
    --flatsky-cross-sigma "$SIGMA" \
    --flatsky-snr-percentiles 0.5,99.5 \
    --l1-clamp-overflow --flatsky-channel-mix bnt \
    --fiducial-obs-cache-dir "$OBSCACHE" \
    --cosmogrid-meta "$CGMETA" \
    --harmonic-obs-cosmo-id cosmo_fiducial \
    --cross-tfds-name nbody_cosmogrid_dataset_tomo_cross/grid_10deg_80px_nonoverlap180 \
    --cross-tfds-data-dir "$TFDS" \
    --field-size 10 --field-npix 80 \
    --nbins 4 --tomo-bin-indices 1,2,3,4 \
    --n-scales 5 --l1-nbins 40 \
    --map-kind nbody --zero-mean-maps \
    --seed $SEED \
    --pca-components 0 \
    --summary-transform log1p-zscore --clip-value 5.0 \
    --min-feature-variance 1e-05 \
    --cross-snr-percentile 1.0 \
    --nde-train-split train \
    --epochs 5000 --batch-size 128 --learning-rate 0.0001 \
    --npe-warmup-steps 128 --npe-decay-steps 10000 \
    --npe-samples 100000 \
    --cache-dir "$DVCACHE" \
    --save-dir "$OUT" \
    --cuda-visible-devices 0 \
    --no-wandb
rc=$?
echo "driver exit=$rc"

echo; echo "=== artefacts ==="
ls -la "$OUT" 2>/dev/null | head -20
[ -f "$OUT/fiducial_summaries.npz" ] && $PY -c "
import numpy as np
d=np.load('$OUT/fiducial_summaries.npz')
print('  fiducial_summaries:', {k:d[k].shape for k in d.files})
p=d['perm']; pa=d['patch']
print(f'  perms {p.min()}..{p.max()} ({len(np.unique(p))}) x patches {pa.min()}..{pa.max()} ({len(np.unique(pa))}) = {len(p)}')
"
exit $rc
