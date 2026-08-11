#!/bin/bash
#SBATCH --job-name=prove_centers
#SBATCH --account=nzu@cpu
#SBATCH --partition=cpu_p1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --hint=nomultithread
#SBATCH --time=01:30:00
#SBATCH --output=logs/prove_centers_%j.out
#
# Step 4 of the patch-geometry investigation: bit-exact reprojection proof -- rebuild a
# patch at a candidate centre and compare it against the corresponding TFDS record.
# See find_centers.slurm before running.
set -uo pipefail
NN_L1_SBI_ARCH=cpu
for _d in "${NN_L1_SBI_JOBS:-}" "${SLURM_SUBMIT_DIR:-}" "${SLURM_SUBMIT_DIR:-}/.." \
          "$(dirname "$0")" "$(dirname "$0")/.."; do
  [ -n "$_d" ] && [ -r "$_d/env.sh" ] && { . "$_d/env.sh"; break; }
done
[ -n "${NN_L1_SBI_ENV_SH:-}" ] || { echo "cannot find env.sh: set NN_L1_SBI_JOBS=<repo>/scripts/slurm" >&2; exit 2; }

: "${NN_L1_SBI_CENTERS_SRC:=$NN_L1_SBI_REPO/scripts/centers}"
nn_l1_sbi_need_path "$NN_L1_SBI_CENTERS_SRC/prove_centers.py" \
  "prove_centers.py (set NN_L1_SBI_CENTERS_SRC; not shipped -- see find_centers.slurm)"
nn_l1_sbi_need_dir NN_L1_SBI_TFDS "cross TFDS root"

export TFDS_DATA_DIR=$NN_L1_SBI_TFDS
export TF_CPP_MIN_LOG_LEVEL=3
export OMP_NUM_THREADS=8

cd "$NN_L1_SBI_DATA"
$PY -u "$NN_L1_SBI_CENTERS_SRC/prove_centers.py"
