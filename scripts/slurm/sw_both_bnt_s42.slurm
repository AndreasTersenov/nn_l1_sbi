#!/bin/bash
#SBATCH --job-name=sw_both_bnt_s42
#SBATCH --account=nzu@a100 --constraint=a100 --qos=qos_gpu_a100-dev
#SBATCH --nodes=1 --ntasks=1 --gres=gpu:1 --cpus-per-task=32 --hint=nomultithread
#SBATCH --time=02:00:00
#SBATCH --output=logs/sw_both_bnt_s42_%j.out
# Population sweep for the l1 +conv+product arm (--cross-op both, 16 ch = 4 autos + 6 conv
# + 6 product), BNT frame, compressor 42.
#
# --arm-label / --out keep the archived names 'joint_bnt_*': this arm was briefly mislabelled
# "joint l1" while the campaign ran, and data/table1/arms.json indexes the published row by
# those exact directory names. Renaming them would orphan the Table-1 bundle. The genuine
# joint-l1 arm is sweeps/jointl1_* (sw_joint_chain.slurm) -- README.md sec 6.
set -uo pipefail
NN_L1_SBI_ARCH=gpu
for _d in "${NN_L1_SBI_JOBS:-}" "${SLURM_SUBMIT_DIR:-}" "${SLURM_SUBMIT_DIR:-}/.." \
          "$(dirname "$0")" "$(dirname "$0")/.."; do
  [ -n "$_d" ] && [ -r "$_d/env.sh" ] && { . "$_d/env.sh"; break; }
done
[ -n "${NN_L1_SBI_ENV_SH:-}" ] || { echo "cannot find env.sh: set NN_L1_SBI_JOBS=<repo>/scripts/slurm" >&2; exit 2; }
export TF_CPP_MIN_LOG_LEVEL=3 XLA_PYTHON_CLIENT_PREALLOCATE=false
cd "$NN_L1_SBI_DATA"
srun $PY -u -m nn_l1_sbi.inference.population_sweep --arm-dir $NN_L1_SBI_DATA/vmim/both_bnt_s42 \
  --arm-label joint_bnt_s42 --out $NN_L1_SBI_DATA/sweeps/joint_bnt_s42 \
  --seeds 41,42,43 --n-obs 9000 --max-perm 50 --m-samples 2000
