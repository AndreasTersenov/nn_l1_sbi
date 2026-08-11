#!/bin/bash
#SBATCH --job-name=sweep_joint_nobnt
#SBATCH --account=nzu@a100
#SBATCH --constraint=a100
#SBATCH --qos=qos_gpu_a100-dev
#SBATCH --nodes=1 --ntasks=1 --gres=gpu:1 --cpus-per-task=32
#SBATCH --hint=nomultithread
#SBATCH --time=02:00:00
#SBATCH --output=logs/sweep_joint_%j.out
#
# SUPERSEDED -- KEPT AS A RECORD, NOT AS A RUNNABLE JOB.
#
# The first population sweep attempted on the l1 +conv+product (--cross-op both) arm: it
# swept the FULL-DRIVER arm directory arms/l1both_nobnt_s41 directly, passing the raw
# observation datavectors and the operator on the command line.  Those two flags
# (--fiducial-datavectors, --op) belong to population_sweep_flatsky.py, which was destroyed
# in the storage failure and is NOT part of this release; the packaged replacement
# nn_l1_sbi.inference.population_sweep sweeps VMIM-COMPRESSED arms instead and accepts
# neither flag.  Running the command below against the packaged driver therefore stops at
# argument parsing -- by design, so the recipe can be read without being mistaken for a
# reproduction path.
#
# The jobs that produced the published l1 +conv+product numbers are
#     compressors : vmim_both.slurm, vmim_both_s4243.slurm, vmim_both_bnt.slurm
#     sweeps      : sw_both_{nobnt,bnt}_{s41,s42,s43,ensemble}.slurm
# Use those.  Set NN_L1_SBI_RUN_SUPERSEDED=1 to run this one anyway (e.g. against your own
# restoration of population_sweep_flatsky.py).
set -uo pipefail
NN_L1_SBI_ARCH=gpu
for _d in "${NN_L1_SBI_JOBS:-}" "${SLURM_SUBMIT_DIR:-}" "${SLURM_SUBMIT_DIR:-}/.." \
          "$(dirname "$0")" "$(dirname "$0")/.."; do
  [ -n "$_d" ] && [ -r "$_d/env.sh" ] && { . "$_d/env.sh"; break; }
done
[ -n "${NN_L1_SBI_ENV_SH:-}" ] || { echo "cannot find env.sh: set NN_L1_SBI_JOBS=<repo>/scripts/slurm" >&2; exit 2; }
export TF_CPP_MIN_LOG_LEVEL=3 XLA_PYTHON_CLIENT_PREALLOCATE=false

if [ "${NN_L1_SBI_RUN_SUPERSEDED:-0}" != "1" ]; then
  cat >&2 <<'MSG'
sweep_both.slurm is superseded and does not run against the packaged population_sweep:
its --fiducial-datavectors / --op flags belong to population_sweep_flatsky.py, which is
not part of this release.  The l1 +conv+product row is produced by
    sw_both_{nobnt,bnt}_{s41,s42,s43,ensemble}.slurm
after the vmim_both*.slurm compressors.  See README.md section 6.
Set NN_L1_SBI_RUN_SUPERSEDED=1 to run this job anyway.
MSG
  exit 2
fi

cd "$NN_L1_SBI_DATA"
srun $PY -u -m nn_l1_sbi.inference.population_sweep \
  --arm-dir $NN_L1_SBI_DATA/arms/l1both_nobnt_s41 \
  --fiducial-datavectors $NN_L1_SBI_DATA/fiducial_both_datavectors_nobnt.npz \
  --op both --arm-label joint_l1_nobnt \
  --out $NN_L1_SBI_DATA/sweeps/joint_l1_nobnt \
  --n-obs 9000 --max-perm 50 --m-samples 2000
