#!/bin/bash
#SBATCH --job-name=install_env
#SBATCH --account=nzu@cpu
#SBATCH --partition=cpu_p1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --hint=nomultithread
#SBATCH --time=02:00:00
#SBATCH --output=logs/install_env_%j.out

# Offline install of the paper env from a pre-downloaded wheelhouse
# ($NN_L1_SBI_WHEELHOUSE, ~12 GB, disposable). Use this when the nodes that will run
# the pipeline have no outbound internet.
#
# Layout: the env (many small files) goes on a filesystem with inode headroom, and the
# wheelhouse (bulk I/O, disposable) on scratch.
#
# Two failures from the previous attempt are fixed here:
#  1. Pins were being overridden. With --find-links, pip picks the NEWEST matching
#     wheel for any unconstrained requirement, so dm-haiku's "numpy>=1.18" pulled
#     numpy 2.4.6 and chex 0.1.92 dragged jax 0.5.0 -> 0.10.2, breaking TF 2.18.
#     Now: one single transaction under a hard constraints file (inputs/constraints.txt).
#  2. "No space left on device" building the sbi_lens sdist -- that was pip's temp
#     dir on the compute node, not the wheelhouse. TMPDIR (set by env.sh) points at
#     $NN_L1_SBI_DATA/tmp instead.
#
# tf_keras is required by tensorflow-probability 0.25 (and hence distrax) on TF>=2.16.
# jaxili / sbi_lens / wl_stats_torch go in with --no-deps per the repo's own
# guidance; their unpinned trees conflict.
# lenstools is deliberately omitted: the code's import closure never uses it.

set -uo pipefail
NN_L1_SBI_ARCH=cpu
for _d in "${NN_L1_SBI_JOBS:-}" "${SLURM_SUBMIT_DIR:-}" "${SLURM_SUBMIT_DIR:-}/.." \
          "$(dirname "$0")" "$(dirname "$0")/.."; do
  [ -n "$_d" ] && [ -r "$_d/env.sh" ] && { . "$_d/env.sh"; break; }
done
[ -n "${NN_L1_SBI_ENV_SH:-}" ] || { echo "cannot find env.sh: set NN_L1_SBI_JOBS=<repo>/scripts/slurm" >&2; exit 2; }

ENVP=$NN_L1_SBI_ENV
WH=$NN_L1_SBI_WHEELHOUSE
CONS=$NN_L1_SBI_INPUTS/constraints.txt
PIP="$ENVP/bin/pip"
PYE="$ENVP/bin/python"

nn_l1_sbi_need_path "$WH"   "wheelhouse (set NN_L1_SBI_WHEELHOUSE)"
nn_l1_sbi_need_path "$CONS" "constraints file"
nn_l1_sbi_need_path "$PIP"  "pip in \$NN_L1_SBI_ENV -- create the env first (conda create -p \"\$NN_L1_SBI_ENV\" python=3.11)"

OFF="--no-index --find-links=$WH -c $CONS"

$PYE -V
echo "wheelhouse: $(ls $WH | wc -l) files, $(du -sh $WH | cut -f1)"
echo "constraints:"; sed 's/^/    /' "$CONS"

echo; echo "=== single constrained transaction ==="
$PIP install $OFF \
  "numpy==1.26.4" "ml-dtypes==0.4.1" \
  "jax[cuda12]==0.5.0" \
  "tensorflow==2.18.0" "tensorflow-datasets==4.9.7" \
  "tensorflow-probability==0.25.0" "tf_keras==2.18.0" \
  "dm-haiku==0.0.13" "optax==0.2.4" "chex==0.1.88" "distrax==0.1.5" \
  "getdist==1.6.1" "healpy==1.18.0" "h5py==3.13.0" \
  scikit-learn matplotlib tqdm wandb torch \
  || echo "!!! main transaction FAILED"

echo; echo "=== custom packages (--no-deps, --no-build-isolation) ==="
for pkg in wl_stats_torch jaxili sbi_lens; do
  echo "--- $pkg"
  $PIP install --no-index --find-links=$WH --no-deps --no-build-isolation "$pkg" \
    || echo "!!! FAILED: $pkg"
done

echo; echo "=== import check ==="
$PYE - <<'PY'
import warnings, os
warnings.filterwarnings("ignore"); os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
mods = ["numpy","jax","jaxlib","tensorflow","tensorflow_datasets","tensorflow_probability",
        "haiku","optax","distrax","getdist","healpy","h5py","sklearn","matplotlib",
        "tqdm","wandb","torch","wl_stats_torch","jaxili","sbi_lens"]
bad=[]
for m in mods:
    try:
        mod=__import__(m); print(f"  OK   {m:24s} {getattr(mod,'__version__','?')}")
    except Exception as e:
        print(f"  FAIL {m:24s} {type(e).__name__}: {str(e)[:80]}"); bad.append(m)
print("\nFAILED:", bad if bad else "none")

# the pins that actually matter for correctness
import numpy, jax, tensorflow as tf
print(f"\npin check: numpy={numpy.__version__} (want 1.26.4)  "
      f"jax={jax.__version__} (want 0.5.0)  tf={tf.__version__} (want 2.18.0)")
PY

echo; echo "=== pip check ==="
$PIP check || true
# written next to the data, NOT over inputs/env_freeze.txt, which is the pinned record
# of the environment the published numbers were produced in.
$PIP freeze > "$NN_L1_SBI_DATA/env_freeze.txt"
echo "wrote $NN_L1_SBI_DATA/env_freeze.txt   (compare against $NN_L1_SBI_INPUTS/env_freeze.txt)"
