#!/bin/bash
#SBATCH --job-name=env_jaxili
#SBATCH --account=nzu@cpu
#SBATCH --partition=prepost
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --hint=nomultithread
#SBATCH --time=03:00:00
#SBATCH --output=logs/env_build_%j.out

# Rebuild the paper's python env (created at $NN_L1_SBI_ENV).
# The version pins below are the ones recorded in inputs/env_freeze.txt and
# inputs/constraints.txt -- the environment the published numbers were produced in.
# JAX + TensorFlow + torch all touch the GPU in this pipeline (TF uses
# set_memory_growth; the L1 driver returns a torch.device), so all three are
# installed with CUDA support -- matching the original setup.
#
# This job needs OUTBOUND INTERNET. On Jean-Zay that means the 'prepost' partition
# (see #SBATCH above) plus the IDRIS proxy; elsewhere, run it wherever pip can reach
# PyPI and set NN_L1_SBI_HTTP_PROXY="" to disable the proxy. If your compute site has
# no internet at all, use install_env.slurm with a pre-downloaded wheelhouse instead.

set -uo pipefail
NN_L1_SBI_ARCH=build
for _d in "${NN_L1_SBI_JOBS:-}" "${SLURM_SUBMIT_DIR:-}" "${SLURM_SUBMIT_DIR:-}/.." \
          "$(dirname "$0")" "$(dirname "$0")/.."; do
  [ -n "$_d" ] && [ -r "$_d/env.sh" ] && { . "$_d/env.sh"; break; }
done
[ -n "${NN_L1_SBI_ENV_SH:-}" ] || { echo "cannot find env.sh: set NN_L1_SBI_JOBS=<repo>/scripts/slurm" >&2; exit 2; }

if [ -n "${NN_L1_SBI_HTTP_PROXY:-}" ]; then
  export http_proxy=$NN_L1_SBI_HTTP_PROXY
  export https_proxy=$NN_L1_SBI_HTTP_PROXY
fi

ENVP=$NN_L1_SBI_ENV
# keep the pip cache off $HOME (inode-limited on many clusters, IDRIS included)
export PIP_CACHE_DIR=${NN_L1_SBI_PIP_CACHE:-$NN_L1_SBI_DATA/.cache/pip}
mkdir -p "$PIP_CACHE_DIR"

echo "=== creating $ENVP (python 3.11) ==="
conda create -p "$ENVP" python=3.11 -y || exit 1
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$ENVP" || exit 1
python -V

python -m pip install -U pip wheel setuptools

step() { echo; echo "=== pip install $* ==="; pip install "$@" || { echo "FAILED: $*"; return 1; }; }

# numpy first: TF 2.18 requires numpy<2, and the original env pinned 1.26.4
step "numpy==1.26.4"
step "jax[cuda12]==0.5.0"
step "tensorflow==2.18.0" "tensorflow-datasets==4.9.7" "tensorflow-probability==0.25.0"
step "dm-haiku==0.0.13" "optax==0.2.4" "distrax==0.1.5"
step "getdist==1.6.1" "healpy==1.18.0" "h5py==3.13.0"
step scikit-learn matplotlib tqdm wandb
step torch                                    # default PyPI wheel bundles CUDA 12
# --- the three non-obvious dependencies -----------------------------------
step "wl_stats_torch==0.1.0"                  # PyPI, authored by A. Tersenov
step "jaxili @ git+https://github.com/sachaguer/jaxili.git"
step "sbi_lens @ git+https://github.com/DifferentiableUniverseInitiative/sbi_lens.git"

echo
echo "=== import check (CPU only; GPU verified separately in a GPU job) ==="
python - <<'PY'
mods = ["numpy","jax","jaxlib","tensorflow","tensorflow_datasets",
        "tensorflow_probability","haiku","optax","distrax","getdist",
        "healpy","h5py","sklearn","matplotlib","tqdm","wandb",
        "torch","wl_stats_torch","jaxili","sbi_lens"]
bad = []
for m in mods:
    try:
        mod = __import__(m)
        print(f"  OK   {m:24s} {getattr(mod,'__version__','?')}")
    except Exception as e:
        print(f"  FAIL {m:24s} {type(e).__name__}: {e}")
        bad.append(m)
print("\nFAILED:", bad if bad else "none")
PY

echo
echo "=== pip freeze -> env_freeze.txt ==="
# written next to the data, NOT over inputs/env_freeze.txt, which is the pinned record
# of the environment the published numbers were produced in.
pip freeze > "$NN_L1_SBI_DATA/env_freeze.txt"
echo "wrote $NN_L1_SBI_DATA/env_freeze.txt   (compare against $NN_L1_SBI_INPUTS/env_freeze.txt)"
echo "done: $ENVP"
