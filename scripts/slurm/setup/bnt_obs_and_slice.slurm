#!/bin/bash
#SBATCH --job-name=bnt_obs_slice
#SBATCH --account=nzu@a100 --constraint=a100 --qos=qos_gpu_a100-dev
#SBATCH --nodes=1 --ntasks=1 --gres=gpu:1 --cpus-per-task=32 --hint=nomultithread
#SBATCH --time=02:00:00
#SBATCH --output=logs/bnt_obs_slice_%j.out
# 1) BNT observation datavectors (--mode bnt, BNT ranges + the frozen _bnt sigma table)
# 2) slice the BNT raw cache + obs into product(10ch) and none(4ch)
set -uo pipefail
NN_L1_SBI_ARCH=gpu
for _d in "${NN_L1_SBI_JOBS:-}" "${SLURM_SUBMIT_DIR:-}" "${SLURM_SUBMIT_DIR:-}/.." \
          "$(dirname "$0")" "$(dirname "$0")/.."; do
  [ -n "$_d" ] && [ -r "$_d/env.sh" ] && { . "$_d/env.sh"; break; }
done
[ -n "${NN_L1_SBI_ENV_SH:-}" ] || { echo "cannot find env.sh: set NN_L1_SBI_JOBS=<repo>/scripts/slurm" >&2; exit 2; }

nn_l1_sbi_need_file NN_L1_SBI_SIGMA_BNT "frozen BNT noise-sigma table"
export TF_CPP_MIN_LOG_LEVEL=3
export FLATSKY_FID_CACHE=$NN_L1_SBI_DATA/full_sphere_cache_fiducial_10deg/nobnt/obs
export XLA_PYTHON_CLIENT_PREALLOCATE=false
BOTH=$NN_L1_SBI_DATA/l1_datavector_cache_bnt/flat_local_both_bnt
cd "$NN_L1_SBI_DATA"
echo "######## BNT observation datavectors ########"
srun $PY -u -m nn_l1_sbi.data.precompute_fiducial_both_datavectors \
  --both-cache $BOTH --sigma "$NN_L1_SBI_SIGMA_BNT" --mode bnt --perms 50 \
  --out $NN_L1_SBI_DATA/fiducial_both_datavectors_bnt.npz || exit 1
echo "######## slice BNT raw cache + obs ########"
$PY -u - <<'PY'
import numpy as np, os, sys, shutil
from nn_l1_sbi.flatsky import flatsky_cross_l1 as fxl
S=os.environ["NN_L1_SBI_DATA"]
BOTH=f"{S}/l1_datavector_cache_bnt/flat_local_both_bnt"
d=np.load(f"{S}/fiducial_both_datavectors_bnt.npz")
np.savez(f"{S}/fid_raw_both_bnt.npz", S=d["x"], perm=d["perm"], patch=d["patch"])
print("  fid_raw_both_bnt:", d["x"].shape, flush=True)
for op in ("product","none"):
    cols=fxl.op_feature_columns(op,4,200); out=f"{S}/l1_datavector_cache_bnt/flat_local_{op}_bnt"
    os.makedirs(out,exist_ok=True)
    for sp in ("l1_train","l1_val"):
        z=np.load(f"{BOTH}/{sp}.npz")
        np.savez(f"{out}/{sp}.npz", theta=z["theta"], x=z["x"][:,cols].astype(np.float32))
        print(f"  {op}/{sp}: {z['x'].shape} -> {(z['x'].shape[0],len(cols))}",flush=True)
    shutil.copy(f"{BOTH}/flat_local_ranges.npy", f"{out}/flat_local_ranges.npy")
    np.savez(f"{S}/fid_raw_{op}_bnt.npz", S=d["x"][:,cols].astype(np.float32),
             perm=d["perm"], patch=d["patch"])
print("BNT SLICING DONE")
PY
