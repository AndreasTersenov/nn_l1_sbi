#!/bin/bash
#SBATCH --job-name=find_centers
#SBATCH --account=nzu@cpu
#SBATCH --partition=cpu_p1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --hint=nomultithread
#SBATCH --time=02:00:00
#SBATCH --output=logs/find_centers_%j.out
#
# Step 1 of the patch-geometry investigation: search for the 180 non-overlapping
# 10-degree patch centres the paper's TFDS was built with.
#
# YOU ALMOST CERTAINLY DO NOT NEED TO RUN THIS. The confirmed result is pinned as
# inputs/centers_PAPER_180.npy and must be used via --centers-npy, never regenerated:
# _build_non_overlapping_centers orders candidates with an unstable np.argsort over
# HEALPix rings, which are blocks of exact ties, so numpy 1.24.4 and 1.26.4 produce
# tilings sharing only 2 of the 180 centres. See inputs/README.md.
#
# The five {find,fit,measure,prove,confirm}_centers.py drivers are one-off investigative
# tools and are not part of the released package; point NN_L1_SBI_CENTERS_SRC at the
# directory holding them if you have a copy.

set -uo pipefail
NN_L1_SBI_ARCH=cpu
for _d in "${NN_L1_SBI_JOBS:-}" "${SLURM_SUBMIT_DIR:-}" "${SLURM_SUBMIT_DIR:-}/.." \
          "$(dirname "$0")" "$(dirname "$0")/.."; do
  [ -n "$_d" ] && [ -r "$_d/env.sh" ] && { . "$_d/env.sh"; break; }
done
[ -n "${NN_L1_SBI_ENV_SH:-}" ] || { echo "cannot find env.sh: set NN_L1_SBI_JOBS=<repo>/scripts/slurm" >&2; exit 2; }

: "${NN_L1_SBI_CENTERS_SRC:=$NN_L1_SBI_REPO/scripts/centers}"
nn_l1_sbi_need_path "$NN_L1_SBI_CENTERS_SRC/find_centers.py" \
  "find_centers.py (set NN_L1_SBI_CENTERS_SRC; not shipped -- see the header)"

export OMP_NUM_THREADS=4

cd "$NN_L1_SBI_DATA"
$PY -u "$NN_L1_SBI_CENTERS_SRC/find_centers.py"
