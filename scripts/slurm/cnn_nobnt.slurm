#!/bin/bash
#SBATCH --job-name=cnn_nobnt
#SBATCH --account=nzu@a100 --constraint=a100 --qos=qos_gpu_a100-dev
#SBATCH --nodes=1 --ntasks=1 --gres=gpu:1 --cpus-per-task=32 --hint=nomultithread
#SBATCH --time=02:00:00
#SBATCH --output=logs/cnn_nobnt_%j.out
# 3 ResNet-18 VMIM compressors (seeds 41/42/43), nobnt. Recipe verbatim from
# cnn_phase/arch_sweep_2026_06_13/SUMMARY_ARCH.md: auto-only flat-local, dim 10,
# 80k steps, batch 128, lr 5e-4, best_val -> resnet18 = FoM3 3326.
# Measured rate: ~70 steps/s => ~19 min train + ~2 min compress per seed.
set -uo pipefail
NN_L1_SBI_ARCH=gpu
for _d in "${NN_L1_SBI_JOBS:-}" "${SLURM_SUBMIT_DIR:-}" "${SLURM_SUBMIT_DIR:-}/.." \
          "$(dirname "$0")" "$(dirname "$0")/.."; do
  [ -n "$_d" ] && [ -r "$_d/env.sh" ] && { . "$_d/env.sh"; break; }
done
[ -n "${NN_L1_SBI_ENV_SH:-}" ] || { echo "cannot find env.sh: set NN_L1_SBI_JOBS=<repo>/scripts/slurm" >&2; exit 2; }

nn_l1_sbi_need_dir NN_L1_SBI_TFDS      "cross TFDS root"
nn_l1_sbi_need_dir NN_L1_SBI_COSMOGRID "CosmoGridV1 simulation root"
TFDS=$NN_L1_SBI_TFDS
export TFDS_DATA_DIR=$TFDS
export CG_PARENT=$NN_L1_SBI_COSMOGRID_PARENT TF_CPP_MIN_LOG_LEVEL=3 XLA_PYTHON_CLIENT_PREALLOCATE=false
cd "$NN_L1_SBI_DATA"
for SEED in 41 42 43; do
  OUT=$NN_L1_SBI_DATA/cnn/nobnt_s${SEED}
  echo "######## CNN resnet18 nobnt seed $SEED ########"
  $PY -u -m nn_l1_sbi.cnn.npe_cnn_nbody_tomo \
    --train-compressor --exit-after-compress \
    --cnn-map-route flat_local --cross-op none  \
    --compressor-arch resnet18 --compressor-dim 10 \
    --compressor-steps 80000 --compressor-batch-size 128 --compressor-lr 5e-4 \
    --compressor-checkpoint-policy best_val --compressor-save-every 5000 \
    --cross-tfds-name nbody_cosmogrid_dataset_tomo_cross/grid_10deg_80px_nonoverlap180 \
    --cross-tfds-data-dir "$TFDS" \
    --fiducial-obs-cache $NN_L1_SBI_DATA/full_sphere_cache_fiducial_10deg \
    --cosmogrid-meta $NN_L1_SBI_COSMOGRID/CosmoGridV1_metainfo.h5 \
    --field-size 10 --field-npix 80 --nbins 4 --tomo-bin-indices 1,2,3,4 \
    --map-kind nbody --zero-mean-maps --seed $SEED \
    --cache-dir "$OUT/cache" --save-dir "$OUT" --total-steps 2 \
    --cuda-visible-devices 0 --no-wandb || echo "SEED $SEED FAILED"
done
