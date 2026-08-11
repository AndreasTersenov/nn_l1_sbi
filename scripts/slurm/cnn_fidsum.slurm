#!/bin/bash
#SBATCH --job-name=cnn_fidsum
#SBATCH --account=nzu@a100 --constraint=a100 --qos=qos_gpu_a100-dev
#SBATCH --nodes=1 --ntasks=1 --gres=gpu:1 --cpus-per-task=16 --hint=nomultithread
#SBATCH --time=01:00:00
#SBATCH --output=logs/cnn_fidsum_%j.out
#
# Compress the 9000 fiducial obs through each finished CNN arm.
# Arms are passed as $1.. (arm dir names under $NN_L1_SBI_DATA/cnn); with no args it
# processes every arm whose cache/cnn_obs.npz exists, skipping ones already built.
#
# Each arm runs its own G1 gate: recompute the driver's persisted observed vector
# (cache/cnn_obs.npz, perm 0 / patch 0) and require max|delta| <= 9e-4. A failure is
# fatal for that arm -- the summaries would be wrong and the FoM3 meaningless.
set -uo pipefail
NN_L1_SBI_ARCH=gpu
for _d in "${NN_L1_SBI_JOBS:-}" "${SLURM_SUBMIT_DIR:-}" "${SLURM_SUBMIT_DIR:-}/.." \
          "$(dirname "$0")" "$(dirname "$0")/.."; do
  [ -n "$_d" ] && [ -r "$_d/env.sh" ] && { . "$_d/env.sh"; break; }
done
[ -n "${NN_L1_SBI_ENV_SH:-}" ] || { echo "cannot find env.sh: set NN_L1_SBI_JOBS=<repo>/scripts/slurm" >&2; exit 2; }

export TFDS_DATA_DIR=$NN_L1_SBI_TFDS
export CG_PARENT=$NN_L1_SBI_COSMOGRID_PARENT TF_CPP_MIN_LOG_LEVEL=3
export XLA_PYTHON_CLIENT_PREALLOCATE=false

ARMS=("$@")
if [ ${#ARMS[@]} -eq 0 ]; then
  ARMS=()
  for d in "$NN_L1_SBI_DATA"/cnn/*/; do
    [ -f "$d/cache/cnn_obs.npz" ] && ARMS+=("$(basename "$d")")
  done
fi
echo "arms: ${ARMS[*]:-<none>}"

fail=0
for A in "${ARMS[@]}"; do
  ARM=$NN_L1_SBI_DATA/cnn/$A
  if [ -f "$ARM/fiducial_summaries.npz" ]; then
    echo "### $A: already built, skipping"; continue
  fi
  echo; echo "############################## $A ##############################"
  $PY -u -m nn_l1_sbi.cnn.build_fiducial_summaries_cnn \
      --arm-dir "$ARM" \
      --obs-cache $NN_L1_SBI_DATA/full_sphere_cache_fiducial_10deg \
      --max-perm 50 --g1-tol 9e-4 --cuda-visible-devices 0
  rc=$?
  [ $rc -eq 0 ] || { echo "### $A FAILED (rc=$rc)"; fail=1; }
done
echo; echo "=== summary ==="
for A in "${ARMS[@]}"; do
  f=$NN_L1_SBI_DATA/cnn/$A/fiducial_summaries.npz
  [ -f "$f" ] && echo "  OK   $A  ($(stat -c%s "$f") B)" || echo "  MISS $A"
done
exit $fail
