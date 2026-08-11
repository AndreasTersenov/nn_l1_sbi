#!/bin/bash
#SBATCH --job-name=vmim_both_bnt
#SBATCH --account=nzu@a100 --constraint=a100 --qos=qos_gpu_a100-dev
#SBATCH --nodes=1 --ntasks=1 --gres=gpu:1 --cpus-per-task=32 --hint=nomultithread
#SBATCH --time=02:00:00
#SBATCH --output=logs/vmim_both_bnt_%j.out
# 3 VMIM compressor seeds for op=both (16ch), BNT frame -- Table-1 arm l1 +conv+product, BNT.
# Inputs come from setup/bnt_obs_and_slice.slurm, which writes both the BNT 'both' cache
# and fid_raw_both_bnt.npz (already under key "S", so no adapter is needed here).
# Defaults reproduce the archived l1_cache_meta.npz (summary_dim 10, hidden 256,256,
# nf 4x128, steps 30000).
set -uo pipefail
NN_L1_SBI_ARCH=gpu
for _d in "${NN_L1_SBI_JOBS:-}" "${SLURM_SUBMIT_DIR:-}" "${SLURM_SUBMIT_DIR:-}/.." \
          "$(dirname "$0")" "$(dirname "$0")/.."; do
  [ -n "$_d" ] && [ -r "$_d/env.sh" ] && { . "$_d/env.sh"; break; }
done
[ -n "${NN_L1_SBI_ENV_SH:-}" ] || { echo "cannot find env.sh: set NN_L1_SBI_JOBS=<repo>/scripts/slurm" >&2; exit 2; }
export TF_CPP_MIN_LOG_LEVEL=3 XLA_PYTHON_CLIENT_PREALLOCATE=false
cd "$NN_L1_SBI_DATA"
for SEED in 41 42 43; do
  echo "######## VMIM both BNT seed $SEED ########"
  $PY -u -m nn_l1_sbi.compression.vmim_from_cache \
    --cache-dir $NN_L1_SBI_DATA/l1_datavector_cache_bnt/flat_local_both_bnt \
    --fid-npz $NN_L1_SBI_DATA/fid_raw_both_bnt.npz \
    --out-cache $NN_L1_SBI_DATA/vmim/both_bnt_s${SEED} \
    --out-fid $NN_L1_SBI_DATA/vmim/both_bnt_s${SEED}/fiducial_summaries.npz \
    --summary-dim 10 --hidden 256,256 --nf-layers 4 --nf-hidden 128 \
    --steps 30000 --seed $SEED \
    --preproc-transform log1p-zscore --clip-value 5 --min-feature-variance 1e-5 \
    --max-minutes 35 --cuda-visible-devices 0 || echo "SEED $SEED FAILED"
done
