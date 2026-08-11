#!/usr/bin/env bash
# Generate the l1 auto-only (= "none" operator) mean-observation contour and MERGE it into
# contour_nobnt_samples.npz, so fig_corner.py can draw a 4th (auto) arm.
#
# One GPU step. Usage:  ./gen_auto_contour.sh [GPU]   (default GPU 1)
# Run ONLY when `nvidia-smi` is healthy (it was wedged host-wide on 2026-06-26; see README).
#
# Environment (same variables and defaults as nn_l1_sbi.paths):
#   PYTHON         interpreter with nn_l1_sbi installed          (default `python`)
#   NN_L1_SBI_DATA root of the unpacked data bundle, read-only   (default ~/.cache/nn_l1_sbi/data)
#   NN_L1_SBI_OUT  where everything is written                   (default ./nn_l1_sbi_out)
set -euo pipefail
GPU="${1:-1}"
PY="${PYTHON:-python}"
DATA="${NN_L1_SBI_DATA:-$HOME/.cache/nn_l1_sbi/data}"
OUT="${NN_L1_SBI_OUT:-$PWD/nn_l1_sbi_out}"
FC=$DATA/analysis/flatsky_cross_2026_06
A=$FC/analytical_nde_match

# 1) VMIM-compress the raw 800-D auto datavector -> 10-D cache + fid_summ (same step the
#    operator conv/both arms used; seed 41, matching the single-compressor l1+product arm).
#    This lands in the analysis tree under $DATA, where dump_contour_samples.py reads it, so it
#    needs $NN_L1_SBI_DATA writable; the bundle already ships the cache, in which case it is skipped.
if [ ! -f "$A/operator_realnvp/none/cache/l1_train.npz" ]; then
  XLA_PYTHON_CLIENT_MEM_FRACTION=0.85 CUDA_VISIBLE_DEVICES="$GPU" \
    "$PY" -m nn_l1_sbi.compression.vmim_from_cache \
      --cache-dir "$FC/l1_matrix/l1_none_cache/flat_local_none" \
      --fid-npz   "$FC/gate_c/lc2st/fiducial_summaries_none.npz" \
      --out-cache "$A/operator_realnvp/none/cache" \
      --out-fid   "$A/operator_realnvp/none/fid_summ.npz" \
      --summary-dim 10 --seed 41 --cuda-visible-devices "$GPU"
fi

# 2) train RealNVP on that 10-D cache, sample at the mean observation, merge l1_auto into the dump.
XLA_PYTHON_CLIENT_MEM_FRACTION=0.85 \
  "$PY" -m nn_l1_sbi.figures.dump_contour_samples --basis nobnt --only l1_auto --merge \
    --cuda-visible-devices "$GPU"

echo "done -- l1_auto merged into $OUT/derived/contour_nobnt_samples.npz"
echo "fig_corner reads $DATA/derived/contour_nobnt_samples.npz: copy it there (or point"
echo "NN_L1_SBI_DATA at a root containing it), then: python -m nn_l1_sbi.figures.fig_corner"
