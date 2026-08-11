#!/bin/bash
#SBATCH --job-name=conv_arm
#SBATCH --account=nzu@a100 --constraint=a100 --qos=qos_gpu_a100-dev
#SBATCH --nodes=1 --ntasks=1 --gres=gpu:1 --cpus-per-task=32 --hint=nomultithread
#SBATCH --time=02:00:00
#SBATCH --output=logs/conv_arm_%j.out
#
# Usage: sbatch conv_arm.slurm <nobnt|bnt>
#
# The l1+conv arm (10 ch = 4 autos + 6 conv), end to end: 3 VMIM compressor seeds then
# 3 single sweeps + the 3-pooled ensemble sweep. Raw datavector was column-sliced from the
# existing 16-ch 'both' cache by slice_conv.slurm (auto block verified identical to the
# 'none' arm in both frames), so no TFDS pass was needed.
#
# Recipe identical to every other l1 arm -- only the input cache differs:
#   VMIM  : summary_dim 10, hidden 256,256, nf 4x128, 30000 steps, log1p-zscore/5/1e-5
#   sweep : 3 pooled flow seeds 41,42,43, 9000 mocks, preproc none/0/1e-12
#
# Validation target (no-BNT): operator_realnvp/conv/n9000 = 2671.1, a SINGLE-compressor
# run whose sibling 'product' arm reproduced the committed 3045 to 0.4% (3057.9).
# GATE C on that campaign: PASS-with-caveat. There is NO published BNT counterpart.
set -uo pipefail
NN_L1_SBI_ARCH=gpu
for _d in "${NN_L1_SBI_JOBS:-}" "${SLURM_SUBMIT_DIR:-}" "${SLURM_SUBMIT_DIR:-}/.." \
          "$(dirname "$0")" "$(dirname "$0")/.."; do
  [ -n "$_d" ] && [ -r "$_d/env.sh" ] && { . "$_d/env.sh"; break; }
done
[ -n "${NN_L1_SBI_ENV_SH:-}" ] || { echo "cannot find env.sh: set NN_L1_SBI_JOBS=<repo>/scripts/slurm" >&2; exit 2; }
export TF_CPP_MIN_LOG_LEVEL=3 XLA_PYTHON_CLIENT_PREALLOCATE=false TF_FORCE_GPU_ALLOW_GROWTH=true

B=$1
if [ "$B" = "bnt" ]; then
  RAW=$NN_L1_SBI_DATA/l1_datavector_cache_bnt/flat_local_conv_bnt; FID=$NN_L1_SBI_DATA/fid_raw_conv_bnt.npz
else
  RAW=$NN_L1_SBI_DATA/l1_datavector_cache/flat_local_conv;         FID=$NN_L1_SBI_DATA/fid_raw_conv.npz
fi
[ -s "$RAW/l1_train.npz" ] || { echo "missing $RAW/l1_train.npz"; exit 1; }
[ -s "$FID" ] || { echo "missing $FID"; exit 1; }

cd "$NN_L1_SBI_DATA"
for SEED in 41 42 43; do
  OUT=$NN_L1_SBI_DATA/vmim/conv_${B}_s${SEED}
  [ -s "$OUT/fiducial_summaries.npz" ] && { echo "### vmim s$SEED done"; continue; }
  echo "######## VMIM conv $B seed $SEED ########"
  $PY -u -m nn_l1_sbi.compression.vmim_from_cache \
    --cache-dir "$RAW" --fid-npz "$FID" \
    --out-cache "$OUT" --out-fid "$OUT/fiducial_summaries.npz" \
    --summary-dim 10 --hidden 256,256 --nf-layers 4 --nf-hidden 128 \
    --steps 30000 --seed $SEED \
    --preproc-transform log1p-zscore --clip-value 5 --min-feature-variance 1e-5 \
    --max-minutes 35 --cuda-visible-devices 0 || echo "VMIM s$SEED FAILED"
done

for SEED in 41 42 43; do
  OUT=$NN_L1_SBI_DATA/sweeps/conv_${B}_s${SEED}
  [ -f "$OUT/median_summary.json" ] && { echo "### sweep s$SEED done"; continue; }
  echo "######## sweep conv $B s$SEED ########"
  $PY -u -m nn_l1_sbi.inference.population_sweep \
     --arm-dir "$NN_L1_SBI_DATA/vmim/conv_${B}_s${SEED}" \
     --arm-label "conv_${B}_s${SEED}" --out "$OUT" \
     --seeds 41,42,43 --n-obs 9000 --max-perm 50 --m-samples 2000 \
     --preproc-transform none --clip-value 0 --min-feature-variance 1e-12 || echo "sweep s$SEED FAILED"
done
OUT=$NN_L1_SBI_DATA/sweeps/conv_${B}_ensemble
if [ ! -f "$OUT/median_summary.json" ]; then
  echo "######## sweep conv $B ENSEMBLE (3 pooled) ########"
  $PY -u -m nn_l1_sbi.inference.population_sweep \
     --arm-dir "$NN_L1_SBI_DATA/vmim/conv_${B}_s41" --arm-dir "$NN_L1_SBI_DATA/vmim/conv_${B}_s42" \
     --arm-dir "$NN_L1_SBI_DATA/vmim/conv_${B}_s43" \
     --arm-label "conv_${B}_ensemble" --out "$OUT" \
     --seeds 41,42,43 --n-obs 9000 --max-perm 50 --m-samples 2000 \
     --preproc-transform none --clip-value 0 --min-feature-variance 1e-12 || echo "ENSEMBLE FAILED"
fi
echo; echo "=== conv $B ==="
for t in s41 s42 s43 ensemble; do
  f=$NN_L1_SBI_DATA/sweeps/conv_${B}_${t}/median_summary.json
  [ -f "$f" ] && printf "  %-10s %s\n" "$t" "$($PY -c "import json;print(round(json.load(open('$f'))['fom3'],1))")" \
              || printf "  %-10s MISSING\n" "$t"
done
