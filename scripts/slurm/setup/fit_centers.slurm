#!/bin/bash
#SBATCH --job-name=fit_centers
#SBATCH --account=nzu@cpu
#SBATCH --partition=cpu_p1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --hint=nomultithread
#SBATCH --time=03:00:00
#SBATCH --output=logs/fit_centers_%j.out
#
# Step 2 of the patch-geometry investigation: fit candidate centres against the cross
# TFDS. See find_centers.slurm for why you almost certainly should not run this --
# the confirmed result is pinned as inputs/centers_PAPER_180.npy.
set -uo pipefail
NN_L1_SBI_ARCH=cpu
for _d in "${NN_L1_SBI_JOBS:-}" "${SLURM_SUBMIT_DIR:-}" "${SLURM_SUBMIT_DIR:-}/.." \
          "$(dirname "$0")" "$(dirname "$0")/.."; do
  [ -n "$_d" ] && [ -r "$_d/env.sh" ] && { . "$_d/env.sh"; break; }
done
[ -n "${NN_L1_SBI_ENV_SH:-}" ] || { echo "cannot find env.sh: set NN_L1_SBI_JOBS=<repo>/scripts/slurm" >&2; exit 2; }

: "${NN_L1_SBI_CENTERS_SRC:=$NN_L1_SBI_REPO/scripts/centers}"
nn_l1_sbi_need_path "$NN_L1_SBI_CENTERS_SRC/fit_centers.py" \
  "fit_centers.py (set NN_L1_SBI_CENTERS_SRC; not shipped -- see find_centers.slurm)"
nn_l1_sbi_need_dir NN_L1_SBI_TFDS "cross TFDS root"

export TFDS_DATA_DIR=$NN_L1_SBI_TFDS
export TF_CPP_MIN_LOG_LEVEL=3 OMP_NUM_THREADS=10 N_GROUPS=12
cd "$NN_L1_SBI_DATA"
$PY -u "$NN_L1_SBI_CENTERS_SRC/fit_centers.py"
