#!/bin/bash
#SBATCH --job-name=measure_centers
#SBATCH --account=nzu@cpu
#SBATCH --partition=cpu_p1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --hint=nomultithread
#SBATCH --time=05:00:00
#SBATCH --output=logs/measure_centers_%j.out
#
# Step 3 of the patch-geometry investigation: independent measurement of the centres
# straight off the cross TFDS -> measured_centers.json (the shipped copy is in
# inputs/). See find_centers.slurm before running.
set -uo pipefail
NN_L1_SBI_ARCH=cpu
for _d in "${NN_L1_SBI_JOBS:-}" "${SLURM_SUBMIT_DIR:-}" "${SLURM_SUBMIT_DIR:-}/.." \
          "$(dirname "$0")" "$(dirname "$0")/.."; do
  [ -n "$_d" ] && [ -r "$_d/env.sh" ] && { . "$_d/env.sh"; break; }
done
[ -n "${NN_L1_SBI_ENV_SH:-}" ] || { echo "cannot find env.sh: set NN_L1_SBI_JOBS=<repo>/scripts/slurm" >&2; exit 2; }

: "${NN_L1_SBI_CENTERS_SRC:=$NN_L1_SBI_REPO/scripts/centers}"
nn_l1_sbi_need_path "$NN_L1_SBI_CENTERS_SRC/measure_centers.py" \
  "measure_centers.py (set NN_L1_SBI_CENTERS_SRC; not shipped -- see find_centers.slurm)"
nn_l1_sbi_need_dir NN_L1_SBI_TFDS "cross TFDS root"

export TFDS_DATA_DIR=$NN_L1_SBI_TFDS
export TF_CPP_MIN_LOG_LEVEL=3 OMP_NUM_THREADS=10 N_GROUPS=12
cd "$NN_L1_SBI_DATA"
$PY -u "$NN_L1_SBI_CENTERS_SRC/measure_centers.py"
