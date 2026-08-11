#!/bin/bash
#SBATCH --job-name=sw_cnn
#SBATCH --account=nzu@a100 --constraint=a100 --qos=qos_gpu_a100-dev
#SBATCH --nodes=1 --ntasks=1 --gres=gpu:1 --cpus-per-task=32 --hint=nomultithread
#SBATCH --time=02:00:00
#SBATCH --output=logs/sw_cnn_%j.out
#
# Usage: sbatch sw_cnn.slurm <label> <preproc> <clip> <arm> [<arm> <arm>]
#   one <arm> = single-quoted row; three = ensemble row (9 pooled flows).
#
# PREPROC — this is the one place the CNN sweep differs from the L1 sweeps:
#   the L1 driver has no --standardize-summary at all, so its compressed summaries went
#   into the flow raw ("none"); the CNN driver standardizes by default
#   (--standardize-summary, --summary-clip-value 5.0) and that is what produced the
#   paper's 3326. So CNN rows use "zscore 5.0". population_sweep's preprocess_summaries
#   fits mean/std on the arm's train x and clips — identical to the driver's
#   fit_and_apply_summary_standardization.
set -uo pipefail
NN_L1_SBI_ARCH=gpu
for _d in "${NN_L1_SBI_JOBS:-}" "${SLURM_SUBMIT_DIR:-}" "${SLURM_SUBMIT_DIR:-}/.." \
          "$(dirname "$0")" "$(dirname "$0")/.."; do
  [ -n "$_d" ] && [ -r "$_d/env.sh" ] && { . "$_d/env.sh"; break; }
done
[ -n "${NN_L1_SBI_ENV_SH:-}" ] || { echo "cannot find env.sh: set NN_L1_SBI_JOBS=<repo>/scripts/slurm" >&2; exit 2; }
export TF_CPP_MIN_LOG_LEVEL=3 XLA_PYTHON_CLIENT_PREALLOCATE=false

LABEL=$1; PREPROC=$2; CLIP=$3; shift 3
ARGS=(); for A in "$@"; do ARGS+=(--arm-dir "$NN_L1_SBI_DATA/cnn/$A"); done
echo "=== sweep $LABEL | preproc=$PREPROC clip=$CLIP | arms: $* ==="

cd "$NN_L1_SBI_DATA"
srun $PY -u -m nn_l1_sbi.inference.population_sweep "${ARGS[@]}" \
  --arm-label "$LABEL" --out "$NN_L1_SBI_DATA/sweeps/$LABEL" \
  --seeds 41,42,43 --n-obs 9000 --max-perm 50 --m-samples 2000 \
  --preproc-transform "$PREPROC" --clip-value "$CLIP" --min-feature-variance 1e-12
