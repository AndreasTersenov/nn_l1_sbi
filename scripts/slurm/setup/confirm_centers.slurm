#!/bin/bash
#SBATCH --job-name=confirm_centers
#SBATCH --account=nzu@cpu
#SBATCH --partition=cpu_p1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --hint=nomultithread
#SBATCH --time=02:00:00
#SBATCH --output=logs/confirm_centers_%j.out
#
# Step 5, the final one: confirm all 22 measurable patch indices reproject bit-exactly.
# Its output is what inputs/centers_PAPER_180.{npy,json} record (22/22 confirmed).
# That file is the single source of truth -- see find_centers.slurm and inputs/README.md.
set -uo pipefail
NN_L1_SBI_ARCH=cpu
for _d in "${NN_L1_SBI_JOBS:-}" "${SLURM_SUBMIT_DIR:-}" "${SLURM_SUBMIT_DIR:-}/.." \
          "$(dirname "$0")" "$(dirname "$0")/.."; do
  [ -n "$_d" ] && [ -r "$_d/env.sh" ] && { . "$_d/env.sh"; break; }
done
[ -n "${NN_L1_SBI_ENV_SH:-}" ] || { echo "cannot find env.sh: set NN_L1_SBI_JOBS=<repo>/scripts/slurm" >&2; exit 2; }

: "${NN_L1_SBI_CENTERS_SRC:=$NN_L1_SBI_REPO/scripts/centers}"
nn_l1_sbi_need_path "$NN_L1_SBI_CENTERS_SRC/confirm_centers.py" \
  "confirm_centers.py (set NN_L1_SBI_CENTERS_SRC; not shipped -- see find_centers.slurm)"
nn_l1_sbi_need_dir NN_L1_SBI_TFDS "cross TFDS root"

export TFDS_DATA_DIR=$NN_L1_SBI_TFDS
export TF_CPP_MIN_LOG_LEVEL=3
export OMP_NUM_THREADS=8

cd "$NN_L1_SBI_DATA"
$PY -u "$NN_L1_SBI_CENTERS_SRC/confirm_centers.py"
