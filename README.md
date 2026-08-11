# nn_l1_sbi — wavelet ℓ1 versus neural compression for tomographic weak-lensing SBI

Companion code for **"Do we need deep learning for tomographic weak lensing? Wavelets versus
neural compression"** (Tersenov, Starck & Kilbinger). The manuscript repository is private until
publication; cite the paper by title until the arXiv identifier exists. This repository is a
curated re-export of a larger working tree: only the code, the batch jobs and the small result
records that produced the paper — nothing else.

The analysis is a *matched* comparison on CosmoGridV1 light-cones: four tomographic κ bins,
Euclid-like shape noise, 180 non-overlapping 10° patches, and two competing summaries — a
hand-crafted starlet **ℓ1-norm** statistic and a **ResNet-18 CNN trained with VMIM** — each
compressed to the **same 10 dimensions**, passed through the **same** conditional RealNVP and
gated by the same TARP + SBC battery. Headline result (median over 9000 mocks,
FoM₃ = 1/√det Cov(Ω_m, σ₈, w₀); ± = spread over three independently trained compressors, see
`docs/NOTE_FOM_ERROR_BARS.md`): the *complete pairwise* ℓ1 statistic reaches **3371 ± 96**,
indistinguishable from the learned compressor's **3326 ± 14**, while the per-channel ℓ1
auto-only baseline reaches only **2448 ± 27**. The gap is inter-bin information, not "deep
learning"; explicit cross-map operators recover it in stages (+conv 2671, +product 3045,
+conv+product 3255). Under BNT nulling the same ordering appears as a retention fraction:
auto-only **0.16**, +product **0.24**, joint ℓ1 **0.72**, CNN **0.96**.

## Quickstart — print Table 1 from the shipped data

No install, no download, no GPU, no network. `data/table1/` is committed; this needs only the
Python standard library and runs in well under a second. From the root of a checkout:

```bash
python - <<'PY'
import json, os
from statistics import mean, stdev
L = lambda p: json.load(open(os.path.join("data/table1", p)))
for arm, a in L("arms.json").items():
    s = [L(p) for p in a["singles"].values()]          # compressor seeds s41, s42, s43
    src = s[0] if a["quoted"] == "single" else L(a["ensemble"])
    rel = stdev(x["fom3"] for x in s) / mean(x["fom3"] for x in s)   # THE error bar, relative
    c = a["published"] if a["published"] is not None else src["fom3"]
    print(f"{arm:26s} {c:7.0f} +- {c*rel:4.0f}   " +
          "  ".join(f"{src[k]:.4f}" for k in ("sigma_Om", "sigma_s8", "sigma_w0")))
PY
```

Output (verified — the 12 rows of Table 1: FoM₃ ± , then σ(Ω_m), σ(σ₈), σ(w₀)):

```
l1 auto, no-BNT               2448 +-   27   0.0503  0.0799  0.2340
l1 auto, BNT                   388 +-   43   0.0861  0.1725  0.3191
l1 +conv, no-BNT              2671 +-  133   0.0518  0.0814  0.2366
l1 +conv, BNT                  458 +-   15   0.0836  0.1643  0.3155
l1 +product, no-BNT           3045 +-  183   0.0462  0.0746  0.2271
l1 +product, BNT               718 +-   29   0.0721  0.1293  0.2994
l1 +conv+product, no-BNT      3255 +-  200   0.0452  0.0714  0.2202
l1 +conv+product, BNT          704 +-   76   0.0755  0.1349  0.3030
joint l1, no-BNT              3371 +-   96   0.0440  0.0716  0.2234
joint l1, BNT                 2424 +-  208   0.0513  0.0856  0.2366
CNN, no-BNT                   3326 +-   14   0.0444  0.0702  0.2288
CNN, BNT                      3186 +-   19   0.0461  0.0730  0.2311
```

`data/table1/README.md` documents every field, the naming trap (`joint_l1_*` ≠ `jointl1_*` —
two different arms), and the three quantities that are easy to confuse with the error bar.

## Install

Python 3.10–3.12 (reference environment: 3.11.5, CUDA 12). Three dependencies — `jaxili`,
`sbi_lens`, `wl-stats-torch` — declare requirement sets that conflict with this project's pins,
so **they must be installed with `--no-deps`**, and `sbi_lens` **must be patched** afterwards or
it does not import at all. A `pyproject.toml` cannot express either step; hence the routes below.
The headers of `pyproject.toml`, `constraints.txt` and `patches/README.md` say why every pin is
there.

**Route A — pip, convenience** (may hit a resolver conflict on `tensorflow-cpu`, pulled in by
`jaxili`'s own metadata):

```bash
pip install -e . -c constraints.txt          # add ".[gpu]" for CUDA 12, ".[notebook]" for Jupyter
```

**Route B — deterministic**, mirrors how the paper environment was built
(`scripts/slurm/setup/install_env.slurm`):

```bash
pip install --no-deps \
  "jaxili==0.1.3" \
  "wl-stats-torch @ git+https://github.com/AndreasTersenov/wl_stats_torch.git@77eafc846645f86a550dba13b177190003668900" \
  "sbi-lens @ git+https://github.com/DifferentiableUniverseInitiative/sbi_lens.git@d239796b30027ccb12b289e4c14fd61fbdec4d61"
pip install -e . -c constraints.txt --no-deps       # this package only
pip install -c constraints.txt numpy scipy matplotlib pandas h5py tqdm pyyaml scikit-learn \
  healpy getdist jax ml-dtypes chex flax dm-haiku optax distrax jaxopt numpyro tensorflow \
  tensorflow-datasets tensorflow-probability tf-keras tensorflow-metadata protobuf torch wandb
```

**Conda** (recommended on a bare machine: conda builds healpy/cfitsio, hdf5 and the BLAS stack
far more reliably than pip). `environment.yml` installs the compiled leaves, then pip takes over:

```bash
conda env create -f environment.yml && conda activate nn-l1-sbi
pip install -e . -c constraints.txt          # REQUIRED, not a repeat
```

The second command is not optional: `environment.yml`'s `pip:` section is a bare `- -e .` with
no `-c constraints.txt`, so creating the env does **not** enforce the seven hard pins — pip is
free to pull numpy 2.x via `dm-haiku` or jax 0.10 via `chex`, either of which breaks
tensorflow 2.18. For Route B under conda, delete the `- -e .` line before `conda env create`
and run the Route B commands inside the env instead.

**Always, after any route** — `import sbi_lens` fails otherwise, because upstream uses a
`tensorflow_probability` API removed in tfp ≥ 0.20 while this project pins tfp 0.25:

```bash
patch -p1 -d "$(python -c 'import sbi_lens, os; print(os.path.dirname(sbi_lens.__file__))')" \
      < patches/sbi_lens_tfp.patch
python -c "import sbi_lens.normflow.models, sbi_lens.normflow.train_model; print('sbi_lens OK')"
python -c "import nn_l1_sbi; print(nn_l1_sbi.__version__)"
```

Re-running `pip install` on `sbi_lens` restores the unpatched files; re-apply the patch. Two
traps that do not raise, they just change the numbers: `wl-stats-torch` **must** come from git
commit `77eafc84` (PyPI 0.1.0 carries the same version string but different code, missing
`subtract_coarse_mean()`), and `numpy` must stay `<2` (TF 2.18 and ml-dtypes 0.4.x need it).

## Environment variables

Every input and output root resolves through `src/nn_l1_sbi/paths.py`; nothing is written inside
the installation directory and no home directory is hard-coded.

| variable | what it points at | default |
|---|---|---|
| `NN_L1_SBI_DATA` | read-only input root: the unpacked Zenodo bundle (maps, derived products, analysis tier) | `~/.cache/nn_l1_sbi/data` |
| `NN_L1_SBI_OUT` | **everything this package writes** — figures (`$NN_L1_SBI_OUT/figures`), sweeps, `save_params/` | `./nn_l1_sbi_out` (relative to the cwd) |
| `NN_L1_SBI_TFDS` | TensorFlow Datasets root holding the cross dataset (~290 GB) | `$TFDS_DATA_DIR`, else `~/tensorflow_datasets` |
| `NN_L1_SBI_COSMOGRID` | raw CosmoGridV1 tree; needed only to rebuild datasets from scratch | none — raises if unset |

The SLURM layer adds `NN_L1_SBI_SIGMA_BNT` (frozen BNT noise-σ table) and `NN_L1_SBI_PYTHON`;
see `scripts/slurm/env.sh`. The notebook reads `NN_L1_SBI_REPO`, `NN_L1_SBI_GPU`,
`NN_L1_SBI_SEED` and `NN_L1_SBI_FOM_TOL`.

## The notebook

`notebooks/reproduce_paper.ipynb`, three sections; each cell states what it does and how long it
takes. Run-all is safe on a laptop: B and C only print a status report unless their inputs (B) or
`NN_L1_SBI_GPU=1` (C) are present. Section A is the one with an acceptance criterion, and it is
the one that needs nothing.

| section | what it reproduces | cost | needs |
|---|---|---|---|
| **A — Table 1** | all 12 arms: FoM₃ ± and the three marginals, plus the assertions against the precomputed `TABLE1_*.json` | **< 1 s, CPU** | nothing — runs from a bare checkout off `data/table1/` |
| **B — Figures** | the paper's 12 figures, **one cell each**, in manuscript order — each invokes its generator in `src/nn_l1_sbi/figures/` and displays the preview inline. The committed outputs show **7 of the 12** (`fig:data`, `fig:l1joint`, `fig:pipeline`, `fig:cp`, `fig:corner`, `fig:bnt`, `fig:fisher`); the rest state which input is missing and what produces it | **seconds to ~10 min per figure, CPU** | the Zenodo bundle at `$NN_L1_SBI_DATA`; **several figures cannot run at all — see below** |
| **C — End-to-end** | retrains one VMIM compressor on the ℓ1 auto/no-BNT arm and re-runs `population_sweep.py` on it | **~1 h, 1 GPU**; skipped unless `NN_L1_SBI_GPU=1` | the raw ℓ1 caches under `$NN_L1_SBI_DATA` (`l1_datavector_cache/flat_local_none`, `fid_raw_none.npz`) |

**What section C actually gates on.** Not the published 2448. The target is the **mean of the
three retrained singles** of `l1 auto, no-BNT` in `data/table1/` — **2772** (s41/s42/s43 =
2777 / 2739 / 2800) — with `|Δ| ≤ NN_L1_SBI_FOM_TOL`, default **5 %**. The notebook says so in
bold: a single retrained compressor must be compared against retrained *singles* (the
3-compressor ensemble sits ~10 % below them, here 2511), and the published numbers predate this
reproduction and other hardware. The published comparison is printed but not gated on. A 3 %
miss is a compressor seed; a 40 % miss is a broken pipeline.

## Repo tour

```
src/nn_l1_sbi/          the package, 7 subpackages, most modules also runnable as scripts
  flatsky/              cross-map operators, starlet ℓ1 and joint-ℓ1 datavectors, the BNT
                        matrix, the frozen per-(channel,scale) noise σ table
  data/                 CosmoGridV1 → TFDS ingestion, full-sphere fiducial observation cache,
                        joint-ℓ1 arm builder, fiducial datavector precomputation
  compression/          VMIM compression of a raw datavector to 10-D, + NDE training
  cnn/                  the ResNet-18 arm: training driver and its fiducial summaries
  inference/            NPE drivers, population_sweep.py (the 9000-mock sweep behind every
                        Table-1 number), ensemble evaluation, TARP/SBC gates
  figures/              15 fig_*.py paper-figure generators + shared style.py / _data.py
  table1/               the error-bar recipe that produced data/table1 (numpy only)
  paths.py              the single source of truth for every input and output root
scripts/slurm/          46 SLURM jobs — the Table-1 campaign; env.sh (paths, modules, fail-fast
                        checks), inputs/, setup/, conv_arms/
data/table1/            61 committed JSON files (48 kB): everything Table 1 needs
data/ZENODO_MANIFEST.md per-file inventory of the Zenodo record and what it still needs
docs/                   PIPELINE.md, FIGURE_MAP.md, NOTE_FOM_ERROR_BARS.md
notebooks/              reproduce_paper.ipynb
patches/                the mandatory sbi_lens tfp-substrate patch, with rationale
pyproject.toml / constraints.txt / environment.yml    install recipe, the 7 hard pins,
                        and the conda spec for the compiled leaves
```

## Data

**In the repository — 118 kB in 69 files, no blobs.**

* `data/table1/` — **61 JSON files, 49,241 B (48 kB)**: the 54 `median_summary.json` sweep
  digests, `arms.json` and its provenance record `rows_table1_v2.json`, and five precomputed
  cross-check tables. Everything Table 1 needs, and **not** on Zenodo — it is committed here, so
  the headline result reproduces from a checkout with no download. (`du -sh` reports 512 K: one
  4 kB filesystem block per sweep directory, not data.)
* `scripts/slurm/inputs/` — 8.4 kB: the pinned 180 patch centres (`centers_PAPER_180.npy`, which
  **must never be regenerated** — the selection uses an unstable `np.argsort` over HEALPix rings,
  and numpy 1.24.4 vs 1.26.4 agree on only 2 of the 180 centres), the environment freeze and the
  fitted-centre records; plus `data/ZENODO_MANIFEST.md`, 40 kB of inventory rather than data.

**On Zenodo** — DOI: `TODO — fill in after upload (10.5281/zenodo.XXXXXXX)`. Two archives, so a
laptop reader never downloads the GB tier:

| archive | tiers | size | enables |
|---|---|--:|---|
| `nn_l1_sbi_data_v1.tar.gz` | **B** 170.6 MB + **C** 121.7 MB + **D** ~10 MB | 302.3 MB | B: the 54 `per_patch_metrics.npz` (block bootstrap, `fig:cp`). C: fiducial patch maps, the full-sphere snapshot, the noise-σ table (`fig:data`, `fig:pipeline`, `fig:leak`). D: the small derived products — **which must be regenerated first, see below** |
| `nn_l1_sbi_analysis_v1.tar.gz` | `analysis/` | 1.99 GB | `fig:fisher`, `fig:l1joint`, every `--recompute` path, and the one-time GPU contour dump that *produces* tier D |

Both unpack into the same root; point `NN_L1_SBI_DATA` at it. Deliberately excluded:
`per_obs_samples.npy` (11.7 GB), the raw ℓ1 caches (35 GB) and the cross TFDS (~290 GB) —
rebuild those with `scripts/slurm/`.

**Not recovered — read this before you run section B.** A disk incident cost several small
derived products; they are not on Zenodo because they no longer exist anywhere. The generators
ship and are correct, but they raise on the missing input, naming the file and its generator.
Per-item recipes in `data/ZENODO_MANIFEST.md` §6.

| figure | missing input | to regenerate |
|---|---|---|
| `fig:corner`, `fig:bnt` | `derived/contour_{nobnt,bnt}_samples.npz` | `figures/dump_contour_samples.py`, then `figures/gen_auto_contour.sh` — GPU, hours |
| `fig:tarp`, `fig:tarp_bnt` | the TARP-DRP gate dumps, then `derived/tarp_proper_bands{,_bnt}.npz` | `inference/run_joint_gate_c.py` (GPU, 16 arms), then `fig_tarp --recompute` (CPU, ~6 min) |
| `fig:sbc` | the same gate dumps — it has no cached form | as above; GPU |
| `fig:l1` (a) | `derived/l1_hist_vs_s8_levels_cache.npz` | no surviving generator: rebuild `(lvl_X, lvl_s8)` from the raw ℓ1 caches — CPU, minutes |
| `fig:ablation` | `operator_realnvp/{conv,both}/n9000/median_summary.json` | rerun the operator ladder with `inference/population_sweep.py` (GPU, ~2 h × 2). The surviving `conv_nobnt_s41` / `joint_l1_nobnt_s41` sweeps are a **different** pipeline run (2720 / 3565 vs the plotted 2671 / 3255) and are not a drop-in substitute |
| `fig:bntwhiten`, `fig:w0` | 6 whiten/BNT campaign `median_summary.json`; `derived/geometry/<arm>/per_patch_grid.csv` | `inference/population_sweep.py` per campaign arm (GPU, ~2 h each); a per-patch geometry run (GPU, no surviving generator). Neither figure is cited by the current manuscript |
| `fig:embed` | `derived/embed_tsne.npz` | `fig_embed --recompute` off the analysis archive — CPU, minutes |
| `fig:l1joint`, `fig:fisher` | inputs survive, but only in the 1.99 GB analysis archive (432 MB / 1.04 GB) | a small derived slice would serve; none is cut yet |

Rendered PDFs of every paper figure exist in the manuscript repository *(private until
publication)*. Do **not** substitute `per_obs_samples.npy` for the missing contour dumps: the
paper's corner is the posterior at the noiseless mean observation with 20 000 draws, so swapping
it silently changes the figure.

## Reproducing the paper

* `docs/PIPELINE.md` — the method document: every stage from CosmoGridV1 light-cones to the
  calibrated figure of merit, with the settings that are load-bearing and why.
* `docs/FIGURE_MAP.md` — figure label → generator script → exact input data, for every figure.
* `docs/NOTE_FOM_ERROR_BARS.md` — what the quoted `±` is (spread over three pre-ensemble
  compressor seeds, carried as a *relative* quantity), what it is not (the single→ensemble shift
  is a bias, reported separately; the leave-one-out jackknife is never quotable), and the
  n=3 → n=6 caveat on the two widest rows.
* `scripts/slurm/` — the campaign: submit order, which job produces which Table-1 row, and the
  gates that stop a job rather than let it emit a plausible wrong number. Read the header of
  `scripts/slurm/env.sh` first — the `#SBATCH` directives are deliberately left site-specific
  (Jean-Zay) and it says exactly what to change elsewhere; jobs fail immediately, by name, on a
  missing path. Site values belong in a `site.local.sh` copied from `site.jean-zay.sh.example`.

## Citation

```bibtex
@article{tersenov_wavelets_vs_neural,
  author  = {Tersenov, Andreas and Starck, Jean-Luc and Kilbinger, Martin},
  title   = {Do we need deep learning for tomographic weak lensing?
             Wavelets versus neural compression},
  journal = {TODO},          % journal, volume, pages: fill in on acceptance
  year    = {TODO},
  eprint  = {TODO},          % arXiv identifier, once posted
  archivePrefix = {arXiv},
  primaryClass  = {astro-ph.CO}
}

@software{nn_l1_sbi,
  author  = {Tersenov, Andreas},
  title   = {nn\_l1\_sbi: wavelet $\ell_1$ versus neural compression for
             tomographic weak-lensing SBI},
  version = {1.0.0},
  url     = {TODO},          % public repository URL, on release
  doi     = {TODO},          % Zenodo DOI, once the bundle is uploaded
  year    = {TODO}
}
```

## License and acknowledgements

MIT — see `LICENSE`.

This work was granted access to the HPC resources of IDRIS under the allocation
`TODO — GENCI/IDRIS allocation number` made by GENCI. All training, sweeps and cache builds
in `scripts/slurm/` ran on the Jean Zay supercomputer (IDRIS, CNRS). The simulations are
CosmoGridV1 (Kacprzak et al. 2023; Fluri et al. 2022).
