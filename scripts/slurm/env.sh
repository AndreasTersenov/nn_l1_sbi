# shellcheck shell=bash
# =============================================================================
#  nn_l1_sbi — shared preamble for every job under scripts/slurm/
# =============================================================================
#
#  Every .slurm file in this tree begins with the same four lines:
#
#      set -uo pipefail
#      NN_L1_SBI_ARCH=gpu          # gpu | cpu | build  (selects the module set)
#      for _d in "${NN_L1_SBI_JOBS:-}" "${SLURM_SUBMIT_DIR:-}" "${SLURM_SUBMIT_DIR:-}/.." \
#                "$(dirname "$0")" "$(dirname "$0")/.."; do
#        [ -n "$_d" ] && [ -r "$_d/env.sh" ] && { . "$_d/env.sh"; break; }
#      done
#
#  Why a search list instead of a plain `source "$(dirname "$0")/env.sh"`:
#  sbatch copies the batch script to the node spool directory before running it, so
#  inside a running job `$0` is typically /var/spool/slurmd/job<N>/slurm_script and
#  `dirname "$0"` is NOT this directory.  The list tries, in order:
#      $NN_L1_SBI_JOBS      explicit override (exported into the job by sbatch)
#      $SLURM_SUBMIT_DIR    the directory you ran `sbatch` from, and its parent
#                           (so `sbatch conv_arms/conv_arm.slurm` works from here,
#                            and `sbatch conv_arm.slurm` works from conv_arms/)
#      dirname $0           and its parent — for `bash scripts/slurm/<job>.slurm`
#  The same block therefore works at all three depths (this directory, conv_arms/,
#  setup/).  If none of them resolve, the job stops with a message telling you to set
#  NN_L1_SBI_JOBS=<repo>/scripts/slurm.
#
# -----------------------------------------------------------------------------
#  THIS FILE HAS TWO PARTS, AND THE DIFFERENCE MATTERS
# -----------------------------------------------------------------------------
#  PART A — PORTABLE DEFAULTS (§2-§7 below).  Every default is derived from this
#      file's own location, from $SCRATCH / $WORK when the site defines them, or
#      from $HOME.  No path here points inside any particular user's directories,
#      and nothing site-specific is silently inherited.  This is what a stranger
#      who clones the repository gets.
#
#  PART B — SITE CONFIGURATION (§1 below).  The concrete paths, module names and
#      proxy the published numbers were produced with are Jean-Zay (IDRIS) values.
#      They are preserved verbatim — they have provenance value — but as an
#      EXAMPLE, not as a default:
#
#          cp site.jean-zay.sh.example site.local.sh     # then edit
#
#      §1 sources scripts/slurm/site.local.sh (and <repo>/site.local.sh) if it
#      exists, before any default is applied, so anything it sets wins.  Write
#      your own site file the same way on another cluster.  site.local.sh and
#      local_paths.sh are both gitignored.
#
# -----------------------------------------------------------------------------
#  THE TWO VARIABLES YOU NORMALLY SET
# -----------------------------------------------------------------------------
#      NN_L1_SBI_SRC    the package source, <repo>/src/nn_l1_sbi.
#                       Derived from this file's own location, so a fresh clone
#                       needs no configuration at all.
#      NN_L1_SBI_DATA   the data root: caches, arms, sweeps, logs, tmp.  The
#                       default is NOT invented here — §3 mirrors, step for
#                       step, the precedence in nn_l1_sbi.paths.data_root(),
#                       which is the single source of truth:
#                           $NN_L1_SBI_DATA  ->  <repo>/data/zenodo (if it
#                           exists)  ->  $HOME/.cache/nn_l1_sbi/data
#                       On an HPC site you almost certainly want this on
#                       scratch instead; set it once in your site file (see §1)
#                       or export it before `sbatch`.  §3 says so out loud when
#                       it falls through to $HOME and a $SCRATCH exists.
#
#  Four more inputs cannot be derived from either and must point at real data:
#      NN_L1_SBI_TFDS        the cross TFDS root (~290 GB); defaults to
#                            $TFDS_DATA_DIR, else ~/tensorflow_datasets — the same
#                            resolution order as nn_l1_sbi.paths.tfds_dir()
#      NN_L1_SBI_COSMOGRID   the CosmoGridV1 simulation root (read-only).  There is
#                            no sensible default, so it is EMPTY unless you set it,
#                            and the jobs that need it stop by name.  (Same rule as
#                            nn_l1_sbi.paths.cosmogrid_root(), which raises.)
#      NN_L1_SBI_SIGMA_BNT   the frozen BNT noise-sigma table (a single .npz);
#                            defaults to the sigma_tables/ slot under
#                            $NN_L1_SBI_DATA, where setup/ writes its twin
#      NN_L1_SBI_PYTHON      the interpreter of the pinned environment; defaults to
#                            $NN_L1_SBI_ENV/bin/python, falling back to python3
#  Jobs that need one of these call nn_l1_sbi_need_dir / nn_l1_sbi_need_file and
#  fail immediately, by name, if it is missing — rather than 40 minutes in.
#
#  Everything can equally be set in the environment before `sbatch`, which exports
#  it into the job.
#
# =============================================================================
#  SITE-SPECIFIC — Jean-Zay (IDRIS).  READ THIS BEFORE RUNNING ELSEWHERE.
# =============================================================================
#  The #SBATCH directives inside the job files are deliberately NOT abstracted
#  away.  They are the recipe that produced the paper's Table 1, and knowing the
#  exact partition/QoS/walltime a number was produced under has provenance value.
#  They are also, without exception, specific to Jean-Zay.  On another cluster,
#  edit these lines in the .slurm files (or override them on the sbatch command
#  line, which wins over the in-file directives):
#
#    #SBATCH --account=nzu@a100 / nzu@cpu
#         IDRIS accounting: <project>@<partition-class>.  Elsewhere this is your
#         own allocation, and the "@a100"/"@cpu" suffix has no meaning.
#
#    #SBATCH --constraint=a100      (GPU jobs)
#    #SBATCH --qos=qos_gpu_a100-dev (GPU jobs)
#         Selects the A100 partition (gpu_p5, 8 GPU/node, 64 cores) and the short
#         development QoS.  qos_gpu_a100-dev gives 2 h of wall time and allows at
#         most 10 submitted jobs per user — that limit is WHY sw_joint_chain.slurm
#         and conv_arm.slurm chain their sweeps sequentially in one job instead of
#         fanning out.  On another cluster: drop --constraint/--qos and use your
#         GPU partition, or keep the chaining, which is harmless either way.
#
#    #SBATCH --partition=cpu_p1     (CPU jobs)
#    #SBATCH --partition=prepost    (setup/build_env.slurm only)
#         prepost is the only IDRIS partition with outbound internet, which is why
#         the environment build runs there.  Compute nodes have none.
#
#    #SBATCH --gres=gpu:1 --cpus-per-task=32 --hint=nomultithread
#         One A100 and one full 32-core quarter-node.  --hint=nomultithread
#         (no hyperthreading) is an IDRIS convention; harmless elsewhere.
#
#    #SBATCH --time=...
#         Walltimes are the measured ones (see README.md), padded a little.  They
#         are capped by the QoS above, not chosen freely.
#
#    #SBATCH --output=logs/<name>_%j.out
#         SLURM parses #SBATCH lines before the shell runs and performs NO
#         variable expansion in them, so this one cannot read $NN_L1_SBI_LOGS.
#         It is relative to the directory you submit from: run `mkdir -p logs`
#         there once, or override per job with
#             sbatch --output="$NN_L1_SBI_LOGS/%x_%j.out" <job>.slurm
#         The literal basenames are kept as they were, so log names in the paper's
#         provenance records still match (note sw_cnn.slurm writes sw_cnn_<jobid>,
#         whatever --job-name says).
#
#    module purge; module load arch/a100; module load python/3.11.5
#         IDRIS module names.  They are NOT the default here — §6 loads no modules
#         unless NN_L1_SBI_MODULES_* say otherwise.  site.jean-zay.sh.example sets
#         them to the strings above.
#
#    http_proxy=http://prodprox.idris.fr:3128
#         IDRIS outbound proxy, used only by setup/build_env.slurm.  Also not a
#         default: NN_L1_SBI_HTTP_PROXY is empty unless a site file sets it.
# =============================================================================

# ---- 0. locate this file, the jobs dir and the repo --------------------------
_nn_self="${BASH_SOURCE[0]:-$0}"
NN_L1_SBI_ENV_SH="$(cd "$(dirname "$_nn_self")" && pwd)/$(basename "$_nn_self")"
NN_L1_SBI_JOBS="$(dirname "$NN_L1_SBI_ENV_SH")"
NN_L1_SBI_REPO="$(cd "$NN_L1_SBI_JOBS/../.." && pwd)"
NN_L1_SBI_INPUTS="$NN_L1_SBI_JOBS/inputs"
unset _nn_self

nn_l1_sbi_die() { echo "[env.sh] ERROR: $*" >&2; exit 2; }

# =============================================================================
#  PART B — 1. SITE CONFIGURATION HOOK (sourced before any default is applied)
# =============================================================================
# Put every machine-specific path in a site file; keep this file generic.
#
#     cp "$NN_L1_SBI_JOBS/site.jean-zay.sh.example" "$NN_L1_SBI_JOBS/site.local.sh"
#
# site.jean-zay.sh.example carries the exact IDRIS values the published numbers
# were produced with (TFDS root, CosmoGridV1 root, frozen BNT sigma table, conda
# env, module loads, proxy).  They are kept as documentation of what ran, not as
# a default anyone inherits by accident.
#
# local_paths.sh is the older name for the same hook and is still honoured, so an
# existing checkout keeps working.  Both names are in .gitignore.
#
# (The search uses $NN_L1_SBI_JOBS rather than "$(dirname "$0")": inside a running
# batch job $0 is the spooled copy under /var/spool/slurmd, whereas $NN_L1_SBI_JOBS
# was resolved from ${BASH_SOURCE[0]} in §0 and is always this directory.)
NN_L1_SBI_SITE="${NN_L1_SBI_SITE:-portable defaults}"
for _nn_ovr in "$NN_L1_SBI_REPO/site.local.sh"  "$NN_L1_SBI_JOBS/site.local.sh" \
               "$NN_L1_SBI_REPO/local_paths.sh" "$NN_L1_SBI_JOBS/local_paths.sh"; do
  if [ -r "$_nn_ovr" ]; then
    . "$_nn_ovr"
    NN_L1_SBI_SITE_FILE="${NN_L1_SBI_SITE_FILE:+$NN_L1_SBI_SITE_FILE }$_nn_ovr"
  fi
done
unset _nn_ovr
: "${NN_L1_SBI_SITE_FILE:=none}"

# =============================================================================
#  PART A — PORTABLE DEFAULTS.  Nothing below names a machine, a project or a
#  user: every path comes from this file's location, from $SCRATCH / $WORK when
#  the site defines them, or from $HOME.
# =============================================================================

# ---- 2. code -----------------------------------------------------------------
# NN_L1_SBI_SRC is the package directory; its PARENT is what goes on PYTHONPATH,
# because the modules import each other as `nn_l1_sbi.<subpackage>.<module>`.
: "${NN_L1_SBI_SRC:=$NN_L1_SBI_REPO/src/nn_l1_sbi}"
NN_L1_SBI_SRC_ROOT="$(dirname "$NN_L1_SBI_SRC")"
[ -d "$NN_L1_SBI_SRC" ] || nn_l1_sbi_die \
  "NN_L1_SBI_SRC=$NN_L1_SBI_SRC does not exist (expected <repo>/src/nn_l1_sbi)"

# ---- 3. data -----------------------------------------------------------------
# SINGLE SOURCE OF TRUTH: nn_l1_sbi.paths.data_root() (src/nn_l1_sbi/paths.py).
# The three steps below are that function transcribed into shell, in the same
# order, and they must be kept in step with it:
#
#     1. $NN_L1_SBI_DATA, if already set (site file §1, or your shell);
#     2. <repo>/data/zenodo, if it exists — the location README.md,
#        data/ZENODO_MANIFEST.md, docs/FIGURE_MAP.md and the notebook tell the
#        reader to unpack the Zenodo bundle into (gitignored);
#     3. $HOME/.cache/nn_l1_sbi/data — the notebook's download target.
#
# This file used to default to ${SCRATCH}/nn_l1_sbi instead, which meant a job
# that sourced env.sh and a driver run by hand in a clean shell silently
# resolved DIFFERENT roots whenever the variable was unset: the job wrote its
# caches and sweeps to scratch, the hand-run driver read $HOME and reported
# inputs "missing" that were merely elsewhere.  Both sides now answer step 1-3.
#
# NOTE ON STEP 2: <repo>/data/zenodo is the read-only figure bundle, but this
# variable is also the WRITABLE compute root (caches, arms, sweeps, logs, tmp)
# for every job in this tree — one name, two jobs.  If you unpack the bundle
# there AND run the compute pipeline, the pipeline writes beside it; keep them
# apart by setting NN_L1_SBI_DATA explicitly for the jobs.  A genuinely
# read-only bundle is caught by the -w test below rather than failing mid-job.
#
# On an HPC site, step 3 is the wrong place for tens of GB of caches: set
# NN_L1_SBI_DATA to a scratch path in your site file (§1) — the note below
# fires when that has not been done.
if [ -z "${NN_L1_SBI_DATA:-}" ]; then
  if [ -d "$NN_L1_SBI_REPO/data/zenodo" ]; then
    NN_L1_SBI_DATA="$NN_L1_SBI_REPO/data/zenodo"
  else
    NN_L1_SBI_DATA="$HOME/.cache/nn_l1_sbi/data"
    if [ -n "${SCRATCH:-}" ] || [ -n "${WORK:-}" ]; then
      echo "[env.sh] note: NN_L1_SBI_DATA is unset; using $NN_L1_SBI_DATA" >&2
      echo "[env.sh]       (nn_l1_sbi.paths.data_root() resolves the same way)." >&2
      echo "[env.sh]       \$SCRATCH/\$WORK exist here — for a full campaign set" >&2
      echo "[env.sh]       NN_L1_SBI_DATA=\${SCRATCH:-\$WORK}/nn_l1_sbi in your site file" >&2
      echo "[env.sh]       ($NN_L1_SBI_JOBS/site.local.sh) and export it for the drivers too." >&2
    fi
  fi
fi
: "${NN_L1_SBI_LOGS:=$NN_L1_SBI_DATA/logs}"
: "${NN_L1_SBI_TMP:=$NN_L1_SBI_DATA/tmp}"
mkdir -p "$NN_L1_SBI_DATA" "$NN_L1_SBI_LOGS" "$NN_L1_SBI_TMP" 2>/dev/null || true
[ -d "$NN_L1_SBI_DATA" ] || nn_l1_sbi_die \
  "NN_L1_SBI_DATA=$NN_L1_SBI_DATA does not exist and could not be created"
[ -w "$NN_L1_SBI_DATA" ] || nn_l1_sbi_die "NN_L1_SBI_DATA=$NN_L1_SBI_DATA is not writable"

# ---- 4. bulk inputs ----------------------------------------------------------
# TFDS: same resolution order as nn_l1_sbi.paths.tfds_dir(), so the shell and the
# python side agree — $NN_L1_SBI_TFDS, else $TFDS_DATA_DIR, else ~/tensorflow_datasets.
: "${NN_L1_SBI_TFDS:=${TFDS_DATA_DIR:-$HOME/tensorflow_datasets}}"
# CosmoGridV1 is a large read-only third-party tree; guessing a location for it
# would only produce a confusing error further in, so the default is EMPTY and the
# jobs that need it stop with "NN_L1_SBI_COSMOGRID is unset".  Set it in your site
# file (on IDRIS it is a shared dataset — see site.jean-zay.sh.example).
: "${NN_L1_SBI_COSMOGRID:=}"
# The frozen BNT noise-sigma table lives beside its no-BNT twin, which
# setup/freeze_sigma.slurm writes under $NN_L1_SBI_DATA/sigma_tables/.
: "${NN_L1_SBI_SIGMA_BNT:=$NN_L1_SBI_DATA/sigma_tables/flatsky_cross_noise_sigma_bnt.npz}"
: "${NN_L1_SBI_WHEELHOUSE:=$NN_L1_SBI_DATA/wheelhouse}"
# No proxy by default; set NN_L1_SBI_HTTP_PROXY in the site file if your login /
# prepost nodes need one to reach PyPI (setup/build_env.slurm is the only user).
: "${NN_L1_SBI_HTTP_PROXY:=}"

# The CosmoGridV1 loader wants the PARENT of the CosmoGridV1 directory, not the
# directory itself (it appends the grid dirname itself).  Keep them consistent —
# and keep the parent empty when the root itself is unset, so the failure is
# reported against NN_L1_SBI_COSMOGRID by name rather than as a stray "/".
if [ -n "${NN_L1_SBI_COSMOGRID_PARENT:-}" ]; then
  :
elif [ -n "$NN_L1_SBI_COSMOGRID" ]; then
  NN_L1_SBI_COSMOGRID_PARENT="$(dirname "$NN_L1_SBI_COSMOGRID")"
else
  NN_L1_SBI_COSMOGRID_PARENT=""
fi

# ---- 5. python environment ---------------------------------------------------
# The pinned env (see inputs/env_freeze.txt, inputs/constraints.txt).  Point
# NN_L1_SBI_PYTHON at any interpreter that satisfies those pins; NN_L1_SBI_ENV is
# only used by setup/build_env.slurm and setup/install_env.slurm to build it.
# Conda environments are hundreds of thousands of small files, so the default puts
# them on $WORK (inode headroom on most HPC sites) rather than on scratch.
: "${NN_L1_SBI_ENV:=${WORK:-$HOME}/envs/nn_l1_sbi}"
: "${NN_L1_SBI_PYTHON:=$NN_L1_SBI_ENV/bin/python}"
if [ ! -x "$NN_L1_SBI_PYTHON" ]; then
  if command -v python3 >/dev/null 2>&1; then
    echo "[env.sh] note: $NN_L1_SBI_PYTHON is not executable; using 'python3' on PATH" >&2
    NN_L1_SBI_PYTHON="$(command -v python3)"
  else
    nn_l1_sbi_die "no usable interpreter: set NN_L1_SBI_PYTHON"
  fi
fi
PY="$NN_L1_SBI_PYTHON"

# ---- 6. module / conda activation (site-specific, overridable) ---------------
# Each job sets NN_L1_SBI_ARCH=gpu|cpu|build before sourcing this file; the
# matching command string is eval'd below.  The portable default is ":" — a no-op,
# because module names are meaningless off the site they were written for, and a
# site with no module system at all must not see `module: command not found`.
# Set these in your site file to your own module loads or `source .../activate`
# line; site.jean-zay.sh.example has the IDRIS ones the paper used.
: "${NN_L1_SBI_MODULES_GPU:=:}"
: "${NN_L1_SBI_MODULES_CPU:=:}"
: "${NN_L1_SBI_MODULES_BUILD:=:}"
case "${NN_L1_SBI_ARCH:-cpu}" in
  gpu)   eval "$NN_L1_SBI_MODULES_GPU"   ;;
  build) eval "$NN_L1_SBI_MODULES_BUILD" ;;
  *)     eval "$NN_L1_SBI_MODULES_CPU"   ;;
esac

# ---- 7. exports common to every job ------------------------------------------
# PYTHONPATH is set AFTER the module loads, because `module purge` can rewrite it.
export PYTHONPATH="$NN_L1_SBI_SRC_ROOT:${PYTHONPATH:-}"
# Legacy shim: population_sweep.py / build_fiducial_summaries_cnn.py still honour
# CNN_SBI_SRC as a sys.path override.  Same directory as PYTHONPATH's first entry.
export CNN_SBI_SRC="$NN_L1_SBI_SRC_ROOT"
# Compute-node /tmp is too small for torch/sdist unpacking on many sites.
export TMPDIR="$NN_L1_SBI_TMP"
export NN_L1_SBI_ENV_SH NN_L1_SBI_JOBS NN_L1_SBI_REPO NN_L1_SBI_INPUTS
export NN_L1_SBI_SRC NN_L1_SBI_SRC_ROOT NN_L1_SBI_DATA NN_L1_SBI_LOGS NN_L1_SBI_TMP
export NN_L1_SBI_TFDS NN_L1_SBI_COSMOGRID NN_L1_SBI_COSMOGRID_PARENT
export NN_L1_SBI_SIGMA_BNT NN_L1_SBI_WHEELHOUSE NN_L1_SBI_ENV NN_L1_SBI_PYTHON
export NN_L1_SBI_SITE NN_L1_SBI_SITE_FILE

# ---- 8. fail-fast helpers ----------------------------------------------------
# Usage:  nn_l1_sbi_need_dir  NN_L1_SBI_TFDS "cross TFDS root"
#         nn_l1_sbi_need_file NN_L1_SBI_SIGMA_BNT "frozen BNT sigma table"
#         nn_l1_sbi_need_path "$SOME_PATH" "the 16-channel 'both' cache"
nn_l1_sbi_need_dir() {
  local var="$1" what="${2:-required directory}" val
  eval "val=\${$var:-}"
  [ -n "$val" ] || nn_l1_sbi_die "$var is unset — it must point at the $what"
  [ -d "$val" ] || nn_l1_sbi_die "$var=$val does not exist — it must point at the $what"
}
nn_l1_sbi_need_file() {
  local var="$1" what="${2:-required file}" val
  eval "val=\${$var:-}"
  [ -n "$val" ] || nn_l1_sbi_die "$var is unset — it must point at the $what"
  [ -f "$val" ] || nn_l1_sbi_die "$var=$val does not exist — it must point at the $what"
}
nn_l1_sbi_need_path() {
  [ -e "$1" ] || nn_l1_sbi_die "missing ${2:-input}: $1"
}

# ---- 9. banner ---------------------------------------------------------------
echo "[env.sh] repo   : $NN_L1_SBI_REPO"
echo "[env.sh] src    : $NN_L1_SBI_SRC   (PYTHONPATH=$NN_L1_SBI_SRC_ROOT)"
echo "[env.sh] data   : $NN_L1_SBI_DATA"
echo "[env.sh] python : $NN_L1_SBI_PYTHON"
echo "[env.sh] arch   : ${NN_L1_SBI_ARCH:-cpu}"
echo "[env.sh] site   : $NN_L1_SBI_SITE  (site file: $NN_L1_SBI_SITE_FILE)"
