#!/bin/bash
#SBATCH --job-name=vmim_both_nobnt
#SBATCH --account=nzu@a100
#SBATCH --constraint=a100
#SBATCH --qos=qos_gpu_a100-dev
#SBATCH --nodes=1 --ntasks=1 --gres=gpu:1 --cpus-per-task=32
#SBATCH --hint=nomultithread
#SBATCH --time=02:00:00
#SBATCH --output=logs/vmim_both_%j.out
#
# VMIM MLP compressor (256,256 -> 10D) on the raw 'both' (16ch) L1 cache, no-BNT frame,
# RealNVP companion.  Table-1 arm: l1 +conv+product, no-BNT.
#
# SEED 41 ONLY -- seeds 42/43 are vmim_both_s4243.slurm.  The split is how the campaign
# actually ran: s41 first with a 100-minute budget, 42/43 afterwards at 45.  The flags are
# otherwise identical, and the defaults of vmim_from_cache reproduce the archived
# l1_cache_meta.npz exactly: summary_dim 10, hidden 256,256, nf_layers 4, nf_hidden 128,
# steps 30000, preproc log1p-zscore / clip 5 / min-var 1e-5 (the downstream sweep then
# uses none / 0 / 1e-12).
#
# THIS JOB ALSO WRITES fid_raw_both.npz.  precompute_obs.slurm writes the 16-channel
# observation datavectors as fiducial_both_datavectors_nobnt.npz with the array under key
# "x"; vmim_from_cache (and setup/slice_raw.slurm) want key "S".  The adapter below is the
# only producer of $NN_L1_SBI_DATA/fid_raw_both.npz in the no-BNT chain, so run this job
# BEFORE setup/slice_raw.slurm.  (The BNT chain has no such step: setup/bnt_obs_and_slice
# writes fid_raw_both_bnt.npz itself.)
set -uo pipefail
NN_L1_SBI_ARCH=gpu
for _d in "${NN_L1_SBI_JOBS:-}" "${SLURM_SUBMIT_DIR:-}" "${SLURM_SUBMIT_DIR:-}/.." \
          "$(dirname "$0")" "$(dirname "$0")/.."; do
  [ -n "$_d" ] && [ -r "$_d/env.sh" ] && { . "$_d/env.sh"; break; }
done
[ -n "${NN_L1_SBI_ENV_SH:-}" ] || { echo "cannot find env.sh: set NN_L1_SBI_JOBS=<repo>/scripts/slurm" >&2; exit 2; }
export TF_CPP_MIN_LOG_LEVEL=3 XLA_PYTHON_CLIENT_PREALLOCATE=false
cd "$NN_L1_SBI_DATA"

nn_l1_sbi_need_path "$NN_L1_SBI_DATA/fiducial_both_datavectors_nobnt.npz" \
  "16-channel no-BNT observation datavectors (setup/precompute_obs.slurm)"

echo "######## adapter: fiducial_both_datavectors_nobnt.npz -> fid_raw_both.npz ########"
$PY - <<'PY'
import os
import numpy as np
S = os.environ["NN_L1_SBI_DATA"]
d = np.load(f"{S}/fiducial_both_datavectors_nobnt.npz")
np.savez(f"{S}/fid_raw_both.npz", S=d["x"], perm=d["perm"], patch=d["patch"])
print("  adapter: S", d["x"].shape, "perm", d["perm"].shape, flush=True)
PY

SEED=41
echo "######## VMIM both no-BNT seed $SEED ########"
srun $PY -u -m nn_l1_sbi.compression.vmim_from_cache \
  --cache-dir $NN_L1_SBI_DATA/l1_datavector_cache/flat_local_both \
  --fid-npz $NN_L1_SBI_DATA/fid_raw_both.npz \
  --out-cache $NN_L1_SBI_DATA/vmim/both_nobnt_s${SEED} \
  --out-fid $NN_L1_SBI_DATA/vmim/both_nobnt_s${SEED}/fiducial_summaries.npz \
  --summary-dim 10 --hidden 256,256 --nf-layers 4 --nf-hidden 128 \
  --steps 30000 --seed $SEED \
  --preproc-transform log1p-zscore --clip-value 5 --min-feature-variance 1e-5 \
  --max-minutes 100 --cuda-visible-devices 0
echo "exit=$?"
ls -la $NN_L1_SBI_DATA/vmim/both_nobnt_s${SEED} 2>/dev/null
