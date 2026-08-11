#!/bin/bash
#SBATCH --job-name=vmim_both_s4243
#SBATCH --account=nzu@a100 --constraint=a100 --qos=qos_gpu_a100-dev
#SBATCH --nodes=1 --ntasks=1 --gres=gpu:1 --cpus-per-task=32 --hint=nomultithread
#SBATCH --time=02:00:00
#SBATCH --output=logs/vmim_both_s4243_%j.out
# Compressor seeds 42 and 43 for op=both, no-BNT -- the continuation of vmim_both.slurm
# (seed 41), which must have run first: it writes fid_raw_both.npz.  Same flags, smaller
# time budget.  Three compressor seeds per (operator, frame) is what the +/- of the row is
# the spread over; see CAMPAIGN_REFERENCE.md sec 2.
set -uo pipefail
NN_L1_SBI_ARCH=gpu
for _d in "${NN_L1_SBI_JOBS:-}" "${SLURM_SUBMIT_DIR:-}" "${SLURM_SUBMIT_DIR:-}/.." \
          "$(dirname "$0")" "$(dirname "$0")/.."; do
  [ -n "$_d" ] && [ -r "$_d/env.sh" ] && { . "$_d/env.sh"; break; }
done
[ -n "${NN_L1_SBI_ENV_SH:-}" ] || { echo "cannot find env.sh: set NN_L1_SBI_JOBS=<repo>/scripts/slurm" >&2; exit 2; }
export TF_CPP_MIN_LOG_LEVEL=3 XLA_PYTHON_CLIENT_PREALLOCATE=false
cd "$NN_L1_SBI_DATA"

nn_l1_sbi_need_path "$NN_L1_SBI_DATA/fid_raw_both.npz" \
  "raw 16-channel fiducial datavectors (written by vmim_both.slurm's adapter)"

for SEED in 42 43; do
  echo "######## VMIM compressor seed $SEED (op=both, no-BNT) ########"
  $PY -u -m nn_l1_sbi.compression.vmim_from_cache \
    --cache-dir $NN_L1_SBI_DATA/l1_datavector_cache/flat_local_both \
    --fid-npz $NN_L1_SBI_DATA/fid_raw_both.npz \
    --out-cache $NN_L1_SBI_DATA/vmim/both_nobnt_s${SEED} \
    --out-fid $NN_L1_SBI_DATA/vmim/both_nobnt_s${SEED}/fiducial_summaries.npz \
    --summary-dim 10 --hidden 256,256 --nf-layers 4 --nf-hidden 128 \
    --steps 30000 --seed $SEED \
    --preproc-transform log1p-zscore --clip-value 5 --min-feature-variance 1e-5 \
    --max-minutes 45 --cuda-visible-devices 0 || echo "SEED $SEED FAILED"
done
