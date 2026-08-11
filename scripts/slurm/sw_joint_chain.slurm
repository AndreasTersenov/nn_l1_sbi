#!/bin/bash
#SBATCH --job-name=sw_joint
#SBATCH --account=nzu@a100 --constraint=a100 --qos=qos_gpu_a100-dev
#SBATCH --nodes=1 --ntasks=1 --gres=gpu:1 --cpus-per-task=32 --hint=nomultithread
#SBATCH --time=02:00:00
#SBATCH --output=logs/sw_joint_%j.out
#
# Usage: sbatch sw_joint_chain.slurm <nobnt|bnt>
#
# All four sweeps for one basis in ONE job (3 singles + the 3-pooled ensemble),
# run sequentially. Chained rather than fanned out because qos_gpu_a100-dev caps
# submitted jobs at 10 per user and each sweep is only ~5-13 min.
#
# Preproc = none/0/1e-12: these are VMIM-COMPRESSED arms, and the compressed-arm
# path is train_nde_from_compressed.py whose defaults are exactly that (the same
# settings run_jointl1_seedcheck.py relied on for the published 3754/3761/4034).
# The log1p-zscore/5/1e-5 defaults in population_sweep_flatsky.py apply to RAW
# datavectors, not to 10-D VMIM summaries.
#
# Validation targets (archived campaign records):
#   nobnt singles 3754 / 3761 / 4034   ensemble 3371 (published)
#   bnt   singles 3232 / 3235 / 2708   ensemble 2424.3 (archived per-obs data)
set -uo pipefail
NN_L1_SBI_ARCH=gpu
for _d in "${NN_L1_SBI_JOBS:-}" "${SLURM_SUBMIT_DIR:-}" "${SLURM_SUBMIT_DIR:-}/.." \
          "$(dirname "$0")" "$(dirname "$0")/.."; do
  [ -n "$_d" ] && [ -r "$_d/env.sh" ] && { . "$_d/env.sh"; break; }
done
[ -n "${NN_L1_SBI_ENV_SH:-}" ] || { echo "cannot find env.sh: set NN_L1_SBI_JOBS=<repo>/scripts/slurm" >&2; exit 2; }
export TF_CPP_MIN_LOG_LEVEL=3 XLA_PYTHON_CLIENT_PREALLOCATE=false TF_FORCE_GPU_ALLOW_GROWTH=true

B=$1
J=$NN_L1_SBI_DATA/joint
cd "$NN_L1_SBI_DATA"

for SEED in 41 42 43; do
  OUT=$NN_L1_SBI_DATA/sweeps/jointl1_${B}_s${SEED}
  [ -f "$OUT/median_summary.json" ] && { echo "### s$SEED done, skip"; continue; }
  echo "######## sweep jointl1 $B s$SEED ########"
  $PY -u -m nn_l1_sbi.inference.population_sweep \
     --arm-dir "$J/jointl1_${B}_s${SEED}" \
     --arm-label "jointl1_${B}_s${SEED}" --out "$OUT" \
     --seeds 41,42,43 --n-obs 9000 --max-perm 50 --m-samples 2000 \
     --preproc-transform none --clip-value 0 --min-feature-variance 1e-12 \
     || echo "s$SEED FAILED"
done

OUT=$NN_L1_SBI_DATA/sweeps/jointl1_${B}_ensemble
if [ ! -f "$OUT/median_summary.json" ]; then
  echo "######## sweep jointl1 $B ENSEMBLE (3 pooled) ########"
  $PY -u -m nn_l1_sbi.inference.population_sweep \
     --arm-dir "$J/jointl1_${B}_s41" --arm-dir "$J/jointl1_${B}_s42" --arm-dir "$J/jointl1_${B}_s43" \
     --arm-label "jointl1_${B}_ensemble" --out "$OUT" \
     --seeds 41,42,43 --n-obs 9000 --max-perm 50 --m-samples 2000 \
     --preproc-transform none --clip-value 0 --min-feature-variance 1e-12 \
     || echo "ENSEMBLE FAILED"
fi

echo; echo "=== jointl1 $B summary ==="
for t in s41 s42 s43 ensemble; do
  f=$NN_L1_SBI_DATA/sweeps/jointl1_${B}_${t}/median_summary.json
  [ -f "$f" ] && printf "  %-10s %s\n" "$t" "$($PY -c "import json;print(round(json.load(open('$f'))['fom3'],1))")" \
              || printf "  %-10s MISSING\n" "$t"
done
