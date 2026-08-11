#!/bin/bash
#SBATCH --job-name=precompute_obs
#SBATCH --account=nzu@a100
#SBATCH --constraint=a100
#SBATCH --qos=qos_gpu_a100-dev
#SBATCH --nodes=1 --ntasks=1 --gres=gpu:1 --cpus-per-task=16
#SBATCH --hint=nomultithread
#SBATCH --time=02:00:00
#SBATCH --output=logs/precompute_obs_%j.out
# Observation datavectors for ALL L1 rows in one pass over the small fiducial cache
# (explicitly NOT the 290GB TFDS). 16ch 'both' output is column-sliced per arm:
# none=4ch/800-D, product=10ch/2000-D, both=16ch/3200-D.
set -uo pipefail
NN_L1_SBI_ARCH=gpu
for _d in "${NN_L1_SBI_JOBS:-}" "${SLURM_SUBMIT_DIR:-}" "${SLURM_SUBMIT_DIR:-}/.." \
          "$(dirname "$0")" "$(dirname "$0")/.."; do
  [ -n "$_d" ] && [ -r "$_d/env.sh" ] && { . "$_d/env.sh"; break; }
done
[ -n "${NN_L1_SBI_ENV_SH:-}" ] || { echo "cannot find env.sh: set NN_L1_SBI_JOBS=<repo>/scripts/slurm" >&2; exit 2; }

export FLATSKY_FID_CACHE=$NN_L1_SBI_DATA/full_sphere_cache_fiducial_10deg/nobnt/obs
export TF_CPP_MIN_LOG_LEVEL=3
cd "$NN_L1_SBI_DATA"
srun $PY -u -m nn_l1_sbi.data.precompute_fiducial_both_datavectors \
  --both-cache $NN_L1_SBI_DATA/l1_datavector_cache/flat_local_both \
  --sigma $NN_L1_SBI_DATA/sigma_tables/flatsky_cross_noise_sigma.npz \
  --perms 50 \
  --out $NN_L1_SBI_DATA/fiducial_both_datavectors_nobnt.npz
