#!/bin/bash
#SBATCH --job-name=nde_rows
#SBATCH --account=nzu@a100 --constraint=a100 --qos=qos_gpu_a100-dev
#SBATCH --nodes=1 --ntasks=1 --gres=gpu:1 --cpus-per-task=32 --hint=nomultithread
#SBATCH --time=02:00:00
#SBATCH --output=logs/nde_rows_%j.out
#
# Usage: sbatch nde_variance_rows.slurm <seedset> <label>:<armdir> [<label>:<armdir> ...]
#   e.g. sbatch nde_variance_rows.slurm 51,52,53 auto_nobnt:vmim/none_nobnt_s41
#   <armdir> is relative to $NN_L1_SBI_DATA.
#
# Measures the NDE(flow)-seed contribution for a row by re-sweeping the SAME trained
# compressor with a different set of pooled flow seeds. Everything else identical, so the
# paired difference against the s41 baseline isolates the NDE term.
#
# Motivation: the Table-1 +/- varies only the COMPRESSOR seed. For the CNN that term is
# 0.41%/0.59% while swapping the flow-seed set moved the answer 1.19%/2.36% -- i.e. the
# quoted bar was measuring the smaller contribution. NOTE_FOM_ERROR_BARS.md sec 2 bounds
# the NDE term at "<1.2%"; CNN BNT measured 2.36%, so the bound needs checking per row.
set -uo pipefail
NN_L1_SBI_ARCH=gpu
for _d in "${NN_L1_SBI_JOBS:-}" "${SLURM_SUBMIT_DIR:-}" "${SLURM_SUBMIT_DIR:-}/.." \
          "$(dirname "$0")" "$(dirname "$0")/.."; do
  [ -n "$_d" ] && [ -r "$_d/env.sh" ] && { . "$_d/env.sh"; break; }
done
[ -n "${NN_L1_SBI_ENV_SH:-}" ] || { echo "cannot find env.sh: set NN_L1_SBI_JOBS=<repo>/scripts/slurm" >&2; exit 2; }
export TF_CPP_MIN_LOG_LEVEL=3 XLA_PYTHON_CLIENT_PREALLOCATE=false TF_FORCE_GPU_ALLOW_GROWTH=true

SEEDSET=$1; shift
TAG=nde$(echo "$SEEDSET" | tr -d ',')
cd "$NN_L1_SBI_DATA"
for spec in "$@"; do
  LBL=${spec%%:*}; ARM=${spec#*:}
  OUT=$NN_L1_SBI_DATA/sweeps/${LBL}_s41_${TAG}
  [ -f "$OUT/median_summary.json" ] && { echo "### $LBL done"; continue; }
  echo "######## $LBL  arm=$ARM  flow seeds $SEEDSET ########"
  $PY -u -m nn_l1_sbi.inference.population_sweep \
     --arm-dir "$NN_L1_SBI_DATA/$ARM" --arm-label "${LBL}_s41_${TAG}" --out "$OUT" \
     --seeds "$SEEDSET" --n-obs 9000 --max-perm 50 --m-samples 2000 \
     --preproc-transform none --clip-value 0 --min-feature-variance 1e-12 \
     || echo "$LBL FAILED"
done
echo; echo "=== baseline (flow 41,42,43) vs $SEEDSET ==="
for spec in "$@"; do
  LBL=${spec%%:*}
  a=$NN_L1_SBI_DATA/sweeps/${LBL}_s41/median_summary.json
  b=$NN_L1_SBI_DATA/sweeps/${LBL}_s41_${TAG}/median_summary.json
  [ -f "$a" ] && [ -f "$b" ] && $PY -c "
import json
x=json.load(open('$a'))['fom3']; y=json.load(open('$b'))['fom3']
print(f'  {\"$LBL\":22s} {x:8.1f} -> {y:8.1f}   delta {y-x:+8.1f}  ({100*(y-x)/x:+.2f}%)')"
done
