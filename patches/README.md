# Post-install patches

This directory holds patches that must be applied to third-party packages **after**
`pip install`. They are not optional: without them the reproduction environment does not
import. Each patch is a unified diff that applies with `patch -p1` from the *package root*
of the package it targets.

| patch | target package | why |
|---|---|---|
| `sbi_lens_tfp.patch` | `sbi_lens` 0.0.1 @ `d239796b30027ccb12b289e4c14fd61fbdec4d61` | upstream reaches the `tensorflow_probability` JAX substrate through attributes that do not exist on tfp 0.25.0; `import sbi_lens.normflow.models` fails outright |

---

## `sbi_lens_tfp.patch`

### Why

`sbi_lens` is not on PyPI, so it is pinned as a direct git requirement at
`d239796b30027ccb12b289e4c14fd61fbdec4d61`. Six modules of that commit reach the JAX
substrate through an attribute of the top-level package rather than by importing it:

```python
import tensorflow_probability as tfp
tfp = tfp.substrates.jax                 # bijectors.py, simulator/utils.py, gen_dataset/*
tfp = tfp.experimental.substrates.jax    # normflow/models.py, normflow/train_model.py
```

Neither attribute is present on **tensorflow-probability 0.25.0**, the version this project
pins (`tensorflow-probability>=0.25,<0.26`, required by TF 2.18 and distrax):

* `tfp.experimental.substrates` was **removed in tfp >= 0.20**.
* `tfp.substrates` is a **lazily loaded submodule**. It only becomes an attribute of
  `tensorflow_probability` once something else has already imported
  `tensorflow_probability.substrates`. On a clean interpreter it is simply not there.

The second point is the one that actually bites, and it is why the two-file version of this
patch was not enough. `normflow/models.py` imports `normflow/bijectors/bijectors.py` on its
own line 6 — *before* reaching its own substrate line — so on a clean install the documented
check dies inside `bijectors.py`:

```
  File ".../sbi_lens/normflow/bijectors/bijectors.py", line 7, in <module>
    tfp = tfp.substrates.jax
          ^^^^^^^^^^^^^^
AttributeError: module 'tensorflow_probability' has no attribute 'substrates'
```

The patch rewrites all six lines to the supported, import-order-independent spelling:

```python
# tfp = tfp.substrates.jax
from tensorflow_probability.substrates import jax as tfp
```

The upstream line is kept as a comment on purpose: it is the record of what upstream said.

### Behaviour is unchanged

Both spellings bind the **identical module object**, `tensorflow_probability.substrates.jax`
— checked with `is` under tfp 0.25.0. Nothing else in any of the six files is touched: the
diff is six one-line replacements and nothing more.

The patched `normflow/models.py` and `normflow/train_model.py` are byte-identical to the
`sbi_lens` copy in the environment the paper's numbers were produced in (verified by `diff`
against that environment). The other four files were left unpatched there; they worked only
because this project's own entry points import `tensorflow_probability.substrates.jax`
before they touch `sbi_lens` — see the deliberate pre-imports in
`src/nn_l1_sbi/cnn/npe_cnn_nbody_tomo.py` and `src/nn_l1_sbi/compression/vmim_from_cache.py`.
This patch removes that hidden ordering dependency; those pre-imports remain correct and
harmless.

### Scope

| file | upstream line | imported by this project? |
|---|---|---|
| `normflow/models.py` | `tfp.experimental.substrates.jax` | yes — `ConditionalRealNVP`, `AffineCoupling` |
| `normflow/train_model.py` | `tfp.experimental.substrates.jax` | yes — `TrainModel` |
| `normflow/bijectors/bijectors.py` | `tfp.substrates.jax` | yes, transitively — this is the line that fails |
| `simulator/utils.py` | `tfp.substrates.jax` | no |
| `gen_dataset/lensing_lognormal_dataset.py` | `tfp.substrates.jax` | no |
| `gen_dataset/lensing_lpt_dataset.py` | `tfp.substrates.jax` | no |

The last three additionally need `lenstools`, `jax-cosmo` and `JaxPM`, which this project
does not install (`--no-deps`), so they are not importable here regardless. They are patched
for completeness and consistency, not because the reproduction touches them.

### How to apply

`sbi_lens` is installed as a normal (non-editable) package, so patch it in site-packages.
The `-d` argument below resolves the installed package root — the directory that contains
`normflow/` — so `-p1` strips the leading `a/` and lands on `normflow/models.py`.

```bash
# in the activated environment, from the repository root
patch -p1 -d "$(python -c 'import sbi_lens, os; print(os.path.dirname(sbi_lens.__file__))')" \
      < patches/sbi_lens_tfp.patch
```

Expected output — six files, no rejects:

```
patching file normflow/models.py
patching file normflow/train_model.py
patching file normflow/bijectors/bijectors.py
patching file simulator/utils.py
patching file gen_dataset/lensing_lognormal_dataset.py
patching file gen_dataset/lensing_lpt_dataset.py
```

Dry-run first if you want to check before touching anything (same command plus
`--dry-run`; it prints `checking file ...` for the same six):

```bash
patch -p1 --dry-run -d "$(python -c 'import sbi_lens, os; print(os.path.dirname(sbi_lens.__file__))')" \
      < patches/sbi_lens_tfp.patch
```

### How to verify

```bash
python -c "import sbi_lens.normflow.models, sbi_lens.normflow.train_model; print('sbi_lens OK')"
```

If `patch` reports *"Reversed (or previously applied) patch detected"*, the patch is already
in place — answer `n` and leave the files alone. Re-running `pip install` on `sbi_lens` (for
example after an environment rebuild) reinstalls the unpatched upstream files, and the patch
must be re-applied.

---

## Where this sits in the install

**The patch is the last step of the install, not part of any route.** Choose exactly one
install route, complete it, then apply the patch once inside the activated environment.
`README.md` (§ Install) is the authority on the routes; `pyproject.toml` and
`constraints.txt` explain every pin. The sequence below is Route B — the deterministic one,
which mirrors how the paper environment was actually built
(`scripts/slurm/setup/install_env.slurm`).

```bash
# 0. environment. Either a bare venv on python 3.10-3.12, or conda for the compiled
#    leaves (healpy/cfitsio, hdf5, BLAS):
#        conda env create -f environment.yml && conda activate nn-l1-sbi
#    Under conda, delete the `- -e .` line from environment.yml first, so that the
#    steps below — not the conda file — decide how this package and its deps go in.

# 1. the three packages whose own metadata fights this project's pins. jaxili wants
#    tensorflow-cpu and jax[cuda12]; sbi_lens wants lenstools, jax-cosmo and JaxPM,
#    none of which this code imports. They go in WITHOUT their dependencies.
pip install --no-deps \
  "jaxili==0.1.3" \
  "wl-stats-torch @ git+https://github.com/AndreasTersenov/wl_stats_torch.git@77eafc846645f86a550dba13b177190003668900" \
  "sbi-lens @ git+https://github.com/DifferentiableUniverseInitiative/sbi_lens.git@d239796b30027ccb12b289e4c14fd61fbdec4d61"

# 2. this package only — --no-deps here too, so pip does not re-resolve step 1
pip install -e . -c constraints.txt --no-deps

# 3. the real dependency set, under the constraints file
pip install -c constraints.txt numpy scipy matplotlib pandas h5py tqdm pyyaml \
  scikit-learn healpy getdist jax ml-dtypes chex flax dm-haiku optax distrax \
  jaxopt numpyro tensorflow tensorflow-datasets tensorflow-probability \
  tf-keras tensorflow-metadata protobuf torch wandb tarp

# 4. THEN, always — sbi_lens does not import until this runs
patch -p1 -d "$(python -c 'import sbi_lens, os; print(os.path.dirname(sbi_lens.__file__))')" \
      < patches/sbi_lens_tfp.patch

# 5. check
python -c "import sbi_lens.normflow.models, sbi_lens.normflow.train_model; print('sbi_lens OK')"
python -c "import nn_l1_sbi; print(nn_l1_sbi.__version__)"
```

If you took Route A instead (`pip install -e . -c constraints.txt`, which may hit a resolver
conflict on `tensorflow-cpu`), steps 4 and 5 are unchanged and still required.

`-c constraints.txt` is load-bearing in every step that installs dependencies: without it pip
is free to pull numpy 2.x via `dm-haiku` or a newer jax via `chex`, either of which breaks
tensorflow 2.18.

## Third-party attribution

`sbi_lens_tfp.patch` is a **derivative work of `sbi_lens`**
(<https://github.com/DifferentiableUniverseInitiative/sbi_lens>), commit
`d239796b30027ccb12b289e4c14fd61fbdec4d61`, which is distributed under the **MIT Licence**,
Copyright (c) 2023 Differentiable Universe Initiative. The context and removed lines in the
diff are copied verbatim from that source. The patch is redistributed here under the same
MIT terms; this repository's own `LICENSE` is also MIT. No `sbi_lens` source file is vendored
in this repository — only the six one-line changes and their surrounding context.

## Related trap that is *not* a patch

`wl-stats-torch` must come from the git commit `77eafc846645f86a550dba13b177190003668900`.
The PyPI release `wl-stats-torch==0.1.0` carries the **same version string** but different,
older code: it is missing `subtract_coarse_mean()` and roughly half of `statistics.py`.
Installing it does not raise — the wavelet transform silently takes a different code path and
the L1 datavectors change. There is no patch for this; the git pin in `pyproject.toml` is the
fix, and it must not be relaxed.

## Upstream fix that would remove this directory

Forking `sbi_lens` under the project account with the six-line change applied, and pointing
the direct-URL dependency at the fork, would make the post-install step unnecessary. Until
then the patch ships here.
