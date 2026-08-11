#!/bin/bash
#SBATCH --job-name=nde_var
#SBATCH --account=nzu@a100 --constraint=a100 --qos=qos_gpu_a100-dev
#SBATCH --nodes=1 --ntasks=1 --gres=gpu:1 --cpus-per-task=32 --hint=nomultithread
#SBATCH --time=02:00:00
#SBATCH --output=logs/nde_var_%j.out
#
# Usage: sbatch nde_variance.slurm <nobnt|bnt>
#
# WHY: the Table-1 +/- varies the COMPRESSOR seed only; the three pooled NDE flow seeds
# (41/42/43) are held fixed for every arm. For rows whose compressor spread is large that
# is fine. For the CNN it is 0.4-0.6%, small enough that the unmeasured NDE contribution
# could be comparable or larger -- NOTE_FOM_ERROR_BARS.md sec 2 bounds NDE variance at
# "<1.2%", and 1.2% of 3326 is +/-40, ~3x the compressor bar.
#
# This re-sweeps the SAME three trained compressors with a DIFFERENT set of flow seeds
# (51/52/53 instead of 41/42/43). Everything else is identical, so the paired difference
# per compressor seed isolates the NDE-seed contribution to the quoted median.
#
# Interpretation:
#   |delta| << compressor spread  -> the quoted +/- is already the whole story
#   |delta| ~  compressor spread  -> the CNN bar should be seed (+) NDE in quadrature
set -uo pipefail
NN_L1_SBI_ARCH=gpu
for _d in "${NN_L1_SBI_JOBS:-}" "${SLURM_SUBMIT_DIR:-}" "${SLURM_SUBMIT_DIR:-}/.." \
          "$(dirname "$0")" "$(dirname "$0")/.."; do
  [ -n "$_d" ] && [ -r "$_d/env.sh" ] && { . "$_d/env.sh"; break; }
done
[ -n "${NN_L1_SBI_ENV_SH:-}" ] || { echo "cannot find env.sh: set NN_L1_SBI_JOBS=<repo>/scripts/slurm" >&2; exit 2; }
export TF_CPP_MIN_LOG_LEVEL=3 XLA_PYTHON_CLIENT_PREALLOCATE=false TF_FORCE_GPU_ALLOW_GROWTH=true

B=$1
cd "$NN_L1_SBI_DATA"
for SEED in s41 s42 s43; do
  OUT=$NN_L1_SBI_DATA/sweeps/cnn_${B}_${SEED}_nde${SEEDS_TAG:-515253}
  [ -f "$OUT/median_summary.json" ] && { echo "### $SEED done"; continue; }
  echo "######## cnn $B $SEED  flow seeds 51,52,53 ########"
  $PY -u -m nn_l1_sbi.inference.population_sweep \
     --arm-dir "$NN_L1_SBI_DATA/cnn/${B}_${SEED}" \
     --arm-label "cnn_${B}_${SEED}_nde${SEEDS_TAG:-515253}" --out "$OUT" \
     --seeds ${SEEDS:-51,52,53} --n-obs 9000 --max-perm 50 --m-samples 2000 \
     --preproc-transform none --clip-value 0 --min-feature-variance 1e-12 \
     || echo "$SEED FAILED"
done
echo; echo "=== cnn $B : flow-seed set 41,42,43  vs  51,52,53 ==="
for SEED in s41 s42 s43; do
  a=$NN_L1_SBI_DATA/sweeps/cnn_${B}_${SEED}_pnone/median_summary.json
  b=$NN_L1_SBI_DATA/sweeps/cnn_${B}_${SEED}_nde${SEEDS_TAG:-515253}/median_summary.json
  [ -f "$a" ] && [ -f "$b" ] && $PY -c "
import json
x=json.load(open('$a'))['fom3']; y=json.load(open('$b'))['fom3']
print(f'  $SEED  {x:8.1f} -> {y:8.1f}   delta {y-x:+7.1f}  ({100*(y-x)/x:+.2f}%)')"
done
