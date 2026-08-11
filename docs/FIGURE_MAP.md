# Figure map

Every figure in the paper is produced by exactly one script in `src/nn_l1_sbi/figures/`. Each
script reads a stored dump, applies the shared house style, and writes a vector PDF (plus a
raster preview under `_preview/`) into the package **output root** — never next to itself, and
never inside the installed package. No figure recomputes an analysis result — the only
exceptions are explicitly flagged (`fig_tarp.py --recompute`, `fig_embed.py --recompute`) and
the GPU contour dump, which is a separate one-time step.

Run any generator as a module:

```bash
python -m nn_l1_sbi.figures.fig_corner
python -m nn_l1_sbi.figures.fig_tarp            # both bases
```

---

## 0. Where figures are read from, and where they are written

Four environment variables configure the package. All four resolve in
`src/nn_l1_sbi/paths.py`, which is the single source of truth; nothing in the figure code
contains an absolute path.

| variable | role | default if unset |
|---|---|---|
| `NN_L1_SBI_DATA` | read-only **input** root: the unpacked Zenodo bundle (`maps/`, `derived/`, `sweeps/`, `tables/`, `analysis/`) | `<repo>/data/zenodo/` if that directory exists (source clone only), else `~/.cache/nn_l1_sbi/data` — where the notebook's download cell writes, so a reader who runs `notebooks/reproduce_paper.ipynb` top to bottom sets nothing |
| **`NN_L1_SBI_OUT`** | **write root — every figure PDF and preview lands in `$NN_L1_SBI_OUT/figures/`** | `./nn_l1_sbi_out`, relative to the **current working directory** → `./nn_l1_sbi_out/figures/` |
| `NN_L1_SBI_COSMOGRID` | raw CosmoGridV1 tree; needed only to rebuild datasets from scratch, never by a figure | none — `cosmogrid_root()` raises if unset |
| `NN_L1_SBI_TFDS` | TensorFlow Datasets directory; dataset builds only, no figure reads it | `$TFDS_DATA_DIR` if set, else `~/tensorflow_datasets` |

```bash
export NN_L1_SBI_DATA=/path/to/nn_l1_sbi_data_v1   # the unpacked bundle
export NN_L1_SBI_OUT=$PWD/out                      # figures -> $PWD/out/figures/
python -m nn_l1_sbi.figures.fig_cp                 # -> $PWD/out/figures/fig_cp_fom3_bars.pdf
```

In code: `nn_l1_sbi.figures._data.figures_dir()` returns `nn_l1_sbi.paths.output_root() /
"figures"` and creates it on first use. Every generator opens with `HERE = figures_dir()` and
saves `HERE/"<name>.pdf"` plus `HERE/"_preview"/"<name>.png"`. `_data.data_root()` is
`nn_l1_sbi.paths.data_root()` re-exported unchanged. Filenames in the tables below are relative
to `$NN_L1_SBI_OUT/figures/`.

> **`NN_L1_SBI_DATA` is overloaded.** `scripts/slurm/env.sh` resolves the same variable the same
> three ways, but then requires it to be **writable** (it is the compute root for caches, arms,
> sweeps and logs), and a site file may point it at a scratch tree. If you source `env.sh` and
> then run a figure script in that shell, `require()` will fail naming a file that is not
> missing, just elsewhere. Run figure scripts in a clean shell.

One more precondition: **`paper.mplstyle`** must sit next to `style.py` in
`src/nn_l1_sbi/figures/` — it is present in this tree (4.9 kB). Every generator calls
`style.use_style()` at import time, which does `plt.style.use(<pkg dir>/paper.mplstyle)`;
without it, every figure script raises at import, before it touches any data.

---

## 1. Data-root layout

```
<data_root>/
  maps/       cosmo_fiducial_perm0.npz                    fig:data, fig:pipeline
              fullsphere_nobnt_cosmo_fiducial_perm0.npz   fig:leak (a)+(b)
  derived/    contour_{nobnt,bnt}_samples.npz             fig:corner, fig:bnt
              tarp_proper_bands{,_bnt}.npz                fig:tarp{,_bnt}
              l1_hist_vs_s8_levels_cache.npz              fig:l1 (a)
              embed_tsne.npz                              fig:embed
              geometry/<arm>/per_patch_grid.csv           fig:w0
  sweeps/     <54 dirs>/per_patch_metrics.npz             notebook only; no figure reads it
  tables/     flatsky_cross_noise_sigma.{npz,json}        provenance only; no figure reads it
  analysis/   the heavy tier, mirroring the analysis tree verbatim:
              flatsky_cross_2026_06/{analytical_nde_match, cnn_phase, l1_matrix, gate_c,
                                     population_sweep, whiten_campaign, bnt_campaign, ...}
```

The authoritative per-file inventory — every bundle path, its source, its exact byte count and
its array schema — is `data/ZENODO_MANIFEST.md`. This section is the figure-facing summary.

Bundle tiers referred to in the **Status** column below:

| tier | contents | size | what it unlocks |
|---|---|--:|---|
| **A** | `data/table1/` — **committed to this repository**, no download | 60 JSON files, 42,973 B (+ a 17 kB README) | Table 1; the FoM₃ numbers several figures hard-code |
| **B** | `sweeps/<dir>/per_patch_metrics.npz`, 54 dirs | 153 MB (170.6 MB with the 5 copies the frozen `analysis/` layout needs) | `fig:cp`; block-bootstrap recomputation |
| **C** | `maps/` — the two fiducial map caches — plus `tables/` (noise-σ provenance, read by no figure) | 121.7 MB | `fig:data`, `fig:pipeline`, `fig:leak` |
| **D** | `derived/` — small products that must exist before their figure can be drawn. **Nothing in this tier exists yet**; every figure that needs it is "needs regeneration" below | ~10 MB once rebuilt | `fig:corner`, `fig:bnt`, `fig:tarp{,_bnt}`, `fig:sbc`, `fig:l1` (a), `fig:w0`, `fig:embed` (cache only) |
| **analysis** | 10-D VMIM caches, the raw 800-D ℓ1 cache, the raw joint fiducial. The **gate-dump subtree of this tier does not survive** — see `fig:tarp` / `fig:sbc` | 1.99 GB (excl. the absent gate dumps) | `fig:fisher`, `fig:l1joint`, `fig:embed`, all `--recompute` paths, the GPU contour dump |

`_data.require()` / `require_glob()` wrap every load: a missing input raises an error naming both
the file **and** the generator that produces it, rather than a bare `FileNotFoundError` or — worse
— a silently wrong plot.

---

## 2. The figure map

Rows marked *(retained)* have a working generator whose figure is not cited by the current
manuscript. They are kept because they are cheap to re-instate, not because they are dead code.
Stripping them leaves exactly the **13 figures of the paper**.

**Status vocabulary.** Every row states exactly one of three things, and nothing softer:

* **reproducible from the repo** — clone and run; no download, no GPU.
* **reproducible from the Zenodo bundle (tier X)** — the inputs exist on disk today and were
  verified present (`nn_l1_sbi_build/recon/E_data_products.md` §8, `data/ZENODO_MANIFEST.md`
  §5). Any caveat that changes what the figure *shows* is stated in the same cell.
* **needs regeneration (generator: …, compute: …)** — at least one input is absent with no
  drop-in replacement, so the figure cannot be drawn from the record as it stands.

A "needs regeneration" row is not a claim that the figure is wrong; the published PDFs exist. It
is a claim that *this repository plus the record* cannot currently redraw it. Note also that
every surviving input is from the **retrained** recovery campaign, so anything rebuilt from them
is a re-derivation, not a bit-reproduction (`data/ZENODO_MANIFEST.md` §6).

| Paper figure | What it shows | Generator (`src/nn_l1_sbi/figures/`) | Input data | Status |
|---|---|---|---|---|
| **`fig:data`** → `fig_data_flatlocal.pdf` | 4 tomographic auto-κ + 6 convolution + 6 product patch-local cross-channels, one representative 10° patch (index 90) | `fig_data.py` | `maps/cosmo_fiducial_perm0.npz` (key `patches`, first 4 channels); the 12 cross-channels are rebuilt on the fly via `nn_l1_sbi.flatsky.flatsky_cross.{build_channels_np, cross_pairs}` | **Reproducible from the Zenodo bundle (tier C)** — `maps/cosmo_fiducial_perm0.npz`, 33,852,223 B, verified present. CPU, seconds |
| **`fig:l1`** → `fig_l1_hist_vs_s8.pdf` | per-channel starlet-ℓ1 histogram vs SNR across the 5 wavelet scales, 3 representative channels (auto κ₄, conv 3×4, product 3×4), coloured by σ₈ | `fig_l1.py::plot_hist_grid()` | `derived/l1_hist_vs_s8_levels_cache.npz` (keys `lvl_X`, `lvl_s8`) | **Needs regeneration** (generator: **none survives** — the level cache must be rebuilt by hand from the raw ℓ1 datavector caches, `l1_matrix/l1_none_cache/flat_local_{none,conv,product}/l1_train.npz`, by binning `x` on σ₈ and averaging within bin, to the schema in `data/ZENODO_MANIFEST.md` §5.3; compute: **CPU, minutes**, but on a prepost/compute node — the caches are 1.04 GB each). The flat channel layout `channel*200 + scale*40 + snr` is part of the contract: `fig_l1.py` slices channels 3/9/15 arithmetically |
| **`fig:l1joint`** → `fig_l1_joint_cellmap.pdf` | the joint ℓ1: the 2-D `(uᵢ,uⱼ)` cell map for one representative pair/scale (pair 5, scale 3, K=10), with the 1-D marginals that *are* the per-channel ℓ1 | `fig_l1.py::plot_joint_cellmap()` | `analysis/…/analytical_nde_match/jointl1_nobnt_raw/fid.npz` (key `S`, mean over obs, reshaped `(6,5,10,10)`) | **Reproducible from the Zenodo bundle (analysis tier)** — `jointl1_nobnt_raw/fid.npz`, 432,289,800 B, verified present. `fig_l1.py` only ever uses `S.mean(0)`, so a one-row `S (1,3000) f4` file (~12 kB) gives a bit-identical figure; if the record ships that substitution instead, it is noted here (`data/ZENODO_MANIFEST.md` §5.4). CPU, seconds |
| **`fig:pipeline`** → `fig_pipeline.pdf` | estimator-controlled design schematic: κ maps → {ℓ1-based / CNN-VMIM} compressor → 10-D summary → shared RealNVP → posterior, with the "every arm gated: TARP-DRP + SBC" banner | `fig_pipeline.py` | `maps/cosmo_fiducial_perm0.npz` (patch 90, 4 autos — same source as `fig:data`); everything else is drawn vector, the banana posterior is analytic | **Reproducible from the Zenodo bundle (tier C)** — same file as `fig:data`. CPU, seconds |
| **`fig:corner`** → `fig_corner_cnn_vs_l1.pdf` | mean-observation posterior on (Ωm, σ₈, w₀), 4 arms (ℓ1 auto / +product / joint ℓ1 / CNN), filled getdist triangle + FoM₃ bar inset | `fig_corner.py` | `derived/contour_nobnt_samples.npz` (keys `truth`, `l1_auto`, `l1_product`, `jointl1`, `cnn`, each `(N,3)`) | **Needs regeneration** (generator: `python -m nn_l1_sbi.figures.dump_contour_samples --basis nobnt`, then `gen_auto_contour.sh` for the `l1_auto` arm; compute: **one-time GPU, hours** — 50 000-step RealNVP 4×128 per compressor, 6 fits per basis; then CPU ~10 s). The dump is absent from every recovered tree, and `sweeps/*/per_obs_samples.npy` is **not** a substitute: the paper's corner is the posterior at the noiseless *mean* observation with 20 000 draws |
| **`fig:cp`** → `fig_cp_fom3_bars.pdf` | per-patch distributions (violins) of σ(Ωm), σ(σ₈), σ(w₀) and FoM₃ over 9000 observations, 3 arms | `fig_cp.py` (via `_data.load_per_patch`) | three `per_patch_metrics.npz`: `l1product_rnvp_s41_n9000`, `jointl1_nobnt_ensemble`, `cnn_phase/…/cnn_resnet18_s41/readout_full` | **Reproducible from the Zenodo bundle (tier B)** — the matching sweep dirs are `product_nobnt_s41`, `jointl1_nobnt_ensemble`, `cnn_nobnt_s41_pnone`; all present. CPU, seconds. **Caveat — the annotated medians move.** These are the *retrained* 2026-07/08 sweeps, not the runs behind the published PDF, and `fig_cp.py` prints each arm's median on its violin: 3231.75 / 3379.51 / 3427.64 against the published 3045 / 3371 / 3326, i.e. **+6.1 % / +0.3 % / +3.1 %** (ℓ1+product / joint ℓ1 / CNN). The *shape* of the distributions and the arm ordering are unchanged; the printed numbers are not. See `data/ZENODO_MANIFEST.md` §5.1b |
| **`fig:ablation`** → `fig_ablation.pdf` | completeness ladder auto → +conv → +product → +both → joint ℓ1, all in the matched RealNVP frame | `fig_ablation.py` | `analysis/…/operator_realnvp/{conv,both}/n9000/median_summary.json`; auto 2448, +product 3045, joint ℓ1 3371 are committed in-script | **Needs regeneration** (generator: `nn_l1_sbi.inference.population_sweep` re-run on the `operator_realnvp` **conv** and **both** arms; compute: **GPU, ~2 h per arm**; then CPU, seconds). Both `median_summary.json` files are absent from every recovered tree and there is **no drop-in replacement**: the nearest survivors are the retrained sweeps `conv_nobnt_s41` (FoM₃ 2720.15) and `joint_l1_nobnt_s41` (3564.87), but those come from a *different pipeline run* — `operator_realnvp` reproduced the product arm's 3045 as 3058, whereas `product_nobnt_s41` gives 3231.75 — and 3564.87 exceeds the script's hard-coded `ylim = (2300, 3550)`. Substituting them would silently redraw the ladder. See `data/ZENODO_MANIFEST.md` §6.5 |
| **`fig:tarp`** → `fig_tarp_overlay.pdf` | TARP-DRP expected coverage with a proper 1σ sightline-bootstrap band, 4 arms, net coverage in the legend (auto +0.018, +product +0.004, joint ℓ1 +0.003, CNN +0.032) | `fig_tarp.py` (loops both bases) | `derived/tarp_proper_bands.npz`; `--recompute` reads the no-BNT gate dumps `…/tarp_drp/dumps/*_{LOW,MID,HIGH}/seed_{41,42,43}/n*_m*/posterior_samples.npz` for all 4 arms | **Needs regeneration** (generator: `nn_l1_sbi.inference.run_joint_gate_c`, which drives `nn_l1_sbi.inference.tarp_stratified_val` + `nn_l1_sbi.inference.run_tarp_coverage`, then `python -m nn_l1_sbi.figures.fig_tarp --recompute`; compute: **GPU, hours per arm** — 3 NDE seeds × 600 validation points, one job per compressor dir — then CPU ~5–6 min for B = 200 bootstraps × 4 arms). Neither the band cache nor the gate dumps survive. Third-party: `tarp` |
| **`fig:tarp_bnt`** → `fig_tarp_overlay_bnt.pdf` | the same, BNT basis; all analytic arms are 3-compressor ensembles (nets: auto +0.001, +product +0.008, joint ℓ1 +0.005, CNN +0.034) | `fig_tarp.py` (`basis="bnt"`) | `derived/tarp_proper_bands_bnt.npz`; `--recompute` reads the BNT gate dumps, incl. the `ens_bnt_{auto,product}_s4{2,3}` ensemble members | **Needs regeneration** (generator: `nn_l1_sbi.inference.run_joint_gate_c` over the 8 **BNT** compressor dirs — including the four `ens_bnt_{auto,product}_s4{2,3}` ensemble members — then `python -m nn_l1_sbi.figures.fig_tarp --recompute`; compute: **GPU, hours per arm**, then CPU ~5–6 min). Identical route to `fig:tarp`, BNT basis. Third-party: `tarp` |
| **`fig:sbc`** → `fig_sbc_hist_combined.pdf` | SBC rank histograms, 4 arms (rows) × 3 science parameters (cols), no-BNT and BNT overlaid in each cell (no-BNT filled+solid, BNT dashed outline, same arm colour) | `fig_sbc.py::plot_hist_combined()` | all gate dumps of **both** bases, via `_data.load_gate_per_seed(arm, basis)` — the same globs as `fig:tarp{,_bnt}`, plus `_data.GATE_ENSEMBLE_BNT_GLOBS` | **Needs regeneration** (generator: `nn_l1_sbi.inference.run_joint_gate_c` for **both** bases — 8 no-BNT + 8 BNT compressor dirs; compute: **GPU, hours per arm**, then ~1–2 min CPU for the figure, which loads every dump). `fig:sbc` has **no cached form**: unlike `fig:tarp` there is no band file that could stand in for the dumps, so it cannot be drawn at all without them. Third-party: `scipy.stats.binom` |
| **`fig:bnt` (a)** → `fig_bnt_analytic.pdf` | analytic ℓ1 under BNT: {auto, +product, joint ℓ1}, each no-BNT filled (2 levels) + a single outer-95 % dashed BNT ring, with the FoM₃ retention inset (×0.16 / ×0.24 / ×0.72) | `fig_bnt.py` (the `panel(..., outer_ring=True)` call in `main()`) | `derived/contour_nobnt_samples.npz` + `derived/contour_bnt_samples.npz` | **Needs regeneration** (generator: `dump_contour_samples.py --basis nobnt` *and* `--basis bnt`, plus `gen_auto_contour.sh` once per basis; compute: **one-time GPU, hours**, then CPU ~15 s). `fig_bnt.py` requires the **same arm keys in both** dumps. Third-party: `getdist` |
| **`fig:bnt` (b)** → `fig_bnt_learned.pdf` | learned CNN under BNT: no-BNT filled + BNT dashed, near-coincident; no inset (retention ×0.96) | `fig_bnt.py` (the `panel(["cnn"], getdist_legend=True)` call in `main()`) | the same two dumps | **Needs regeneration** (generator: `dump_contour_samples.py --basis nobnt` *and* `--basis bnt` — the same two dumps as (a), no extra job; compute: **one-time GPU, hours** shared with (a), then CPU ~15 s). Third-party: `getdist` |
| **`fig:fisher`** → `fig_fisher.pdf` | where the cosmology information lives: per-bin diagonal Fisher of the 800-D auto starlet-ℓ1, summed over the 4 tomographic bins, as a (scale × SNR) heatmap per parameter | `fig_fisher.py` | `analysis/…/l1_matrix/l1_none_cache/flat_local_none/l1_train.npz` (keys `x` `(N,800)`, `theta`) | **Reproducible from the Zenodo bundle (analysis tier)** — `l1_train.npz`, 1,043,415,858 B (**1.04 GB**, the largest single figure input), verified present. `fig_fisher.py` is its only consumer and needs `x` and `theta[:, :3]`; there is no smaller form that leaves the script unmodified, so either the record ships it or `fig:fisher` is declared non-reproducible from the record. CPU, tens of seconds |
| **`fig:leak` (a)** → `fig_cross_leakage_scales.pdf` *(retained)* | super-patch variance fraction per multipole: cross-maps carry much more of their variance at ℓ < 18 (larger than a 10° patch) than the autos | `fig_leak.py` | `maps/fullsphere_nobnt_cosmo_fiducial_perm0.npz` (keys `full_auto`, `full_cross`, `lmax`) | **Reproducible from the Zenodo bundle (tier C)** — `fullsphere_nobnt_cosmo_fiducial_perm0.npz`, 87,878,466 B, verified present. CPU, ~30 s. Third-party: `healpy` |
| **`fig:leak` (b)** → `fig_fullsphere_maps.png` *(retained)* | full-sphere mollweide views of the 4 auto + 6 cross channels (raster, 300 dpi — mollweide is inherently raster) | `fig_leak_maps.py` | the same snapshot; channel pairing via `nn_l1_sbi.flatsky.flatsky_cross.cross_pairs` | **Reproducible from the Zenodo bundle (tier C)** — same file as (a). CPU, ~1 min. Third-party: `healpy` |
| **`fig:bntwhiten`** → `fig_bnt_whiten.pdf` *(retained)* | whitening decomposition: no-BNT vs whitened `Q = (BBᵀ)^{-1/2}B` vs BNT FoM₃ bars for the two ℓ1 arms, plus the deep-direction ladder (+1 / +2 deep channels) | `fig_bntwhiten.py` | six `median_summary.json`: `population_sweep/flat_{none,product}`, `whiten_campaign/population_sweep/l1_{none,product}`, `bnt_campaign/population_sweep/l1_{none,product}`, plus optional `bntdeep{,2}_campaign/population_sweep/l1_none` | **Needs regeneration** (generator: `nn_l1_sbi.inference.population_sweep` per campaign arm; compute: **GPU, ~2 h per arm × 6 required arms**; then CPU, seconds). None of the campaign sweeps survive. The two `bntdeep*` files are read through `opt()`, which returns `None` if absent — those bars simply disappear |
| **`fig:w0`** → `fig_w0_offset.pdf` *(retained)* | fiducial w₀ pull vs patch size: ℓ1 auto+cross −0.37σ at 20° shrinks to ≈ −0.10 at 10°, matching the CNN → a flat-sky projection residual, not an ℓ1 property | `fig_w0.py` | `derived/geometry/{l1_auto_cross,cnn_auto_cross}/per_patch_grid.csv` (column `pull_w_0`); the 20° value −0.37 is committed in-script | **Needs regeneration** (generator: **none survives** — the phase-C geometry grids must be rebuilt by a per-patch geometry run at 10°, emitting a `pull_w_0` column for `l1_auto_cross` and `cnn_auto_cross`; compute: **GPU**; then CPU, seconds). The only CSV input in the set |
| **`fig:embed`** → `fig_embed.pdf` *(retained)* | near-sufficiency made visual: t-SNE of the 10-D summary, 3 compressors × 2 colour rows (S₈ top, Ωm bottom); all three fold cosmology onto the same smooth 2-D manifold | `fig_embed.py` | `derived/embed_tsne.npz`; `--recompute` reads three 10-D VMIM caches: `l1product_vmim_s41/cache/l1_train.npz`, `jointl1_nobnt/cache/l1_train.npz`, `cnn_phase/…/cnn_resnet18_s41/cache/cnn_train.npz` (keys `x`, `theta`) | **Reproducible from the Zenodo bundle (analysis tier)** — the tier-D t-SNE cache is absent, but all three 10-D VMIM caches survive (20,713,458 B each; only `l1_train.npz` is read), and `fig_embed.py --recompute` rebuilds the cache itself: 3 t-SNEs on 6000×10, several minutes CPU, no GPU. This is the one absent tier-D product that needs no new science run. Third-party: `scikit-learn` (its only consumer) |

### Roll-up: the 13 paper figures

| status | count | figures |
|---|--:|---|
| reproducible from the repo | 0 | — (Table 1 is the only thing the repo alone produces; it is not a figure) |
| reproducible from the Zenodo bundle | 5 | `fig:data`, `fig:pipeline` (tier C); `fig:cp` (tier B, **median caveat**); `fig:l1joint`, `fig:fisher` (analysis tier) |
| needs regeneration | 8 | `fig:corner`, `fig:bnt` (a)+(b) — contour dumps, GPU; `fig:tarp`, `fig:tarp_bnt`, `fig:sbc` — gate dumps, GPU; `fig:ablation` — operator sweeps, GPU; `fig:l1` — level cache, CPU |

Of the 5 retained figures: `fig:leak` (a)+(b) and `fig:embed` are reproducible from the bundle;
`fig:bntwhiten` and `fig:w0` need regeneration.

So **`fig:l1` is the only "needs regeneration" row in the whole set that costs no GPU time.**

### Alternate outputs from the same generators

| Output | What it shows | Generator | Input data | Status |
|---|---|---|---|---|
| `fig_sbc_hist{,_bnt}.pdf` | faceted 4×3 rank-histogram grid, one basis per file | `fig_sbc.py::plot_hist()` | as `fig:sbc` | **Needs regeneration** (generator and compute: as `fig:sbc`); superseded in the manuscript by the combined grid |
| `fig_sbc_ecdf{,_bnt}.pdf` | ECDF difference `F̂(r) − r` with a **simultaneous** 99 % band | `fig_sbc.py::plot_ecdf()` | as `fig:sbc` | **Needs regeneration** (generator and compute: as `fig:sbc`). Worth keeping: it is the more stringent test and §5.3 below leans on it |
| `fig_sbc_hist_overlay{,_bnt}.pdf` | legacy single-row overlay | `fig_sbc.py::plot_hist_overlay()` | as `fig:sbc` | **Needs regeneration** (generator and compute: as `fig:sbc`); kept for comparison only |
| `fig_bnt_analytic_noprod.pdf` | as `fig:bnt` (a) but only auto + joint ℓ1 contours (+product stays in the inset bar) | `fig_bnt.py`, second `panel()` call in `main()` | as `fig:bnt` | **Needs regeneration** (generator and compute: as `fig:bnt`); less-busy alternative |

### Helpers

| File | Role |
|---|---|
| `style.py` | shared A&A style: widths (single 88 mm, intermediate 120 mm, double 180 mm), the fixed per-arm palette, `use_style()`, `getdist_settings()` |
| `_data.py` | the one place figure data provenance lives: `data_root()`, `require()`/`require_glob()`, the per-patch and gate-dump path tables, and the rank/pooling loaders |
| `dump_contour_samples.py` | one-time **GPU** dump of the mean-observation posteriors → `derived/contour_{nobnt,bnt}_samples.npz`, so `fig_corner.py`/`fig_bnt.py` stay pure-CPU getdist replots. Defaults: `--m-samples 20000`, `--nde-seed 41`, `--max-perm 50` |
| `gen_auto_contour.sh` | shell driver that VMIM-compresses the raw 800-D auto datavector (seed 41, matching the +product arm) and merges the `l1_auto` key into the no-BNT dump. **GPU**, tens of minutes |

---

## 3. Regenerating the missing inputs

Everything under "needs regeneration" reduces to six jobs. Except where noted, sizes are what the
products will occupy once rebuilt (~10 MB in total — they belong in `derived/`, i.e. Tier D).

| Product | Generator | Compute | Unlocks |
|---|---|---|---|
| `contour_{nobnt,bnt}_samples.npz` | `nn_l1_sbi.figures.dump_contour_samples --basis {nobnt,bnt}`, then `gen_auto_contour.sh` for the `l1_auto` arm | **one-time GPU**, hours (50 000-step RealNVP 4×128 per compressor, 6 fits per basis) | `fig:corner`, `fig:bnt` (a)+(b) |
| gate dumps (`tarp_drp/dumps/**/posterior_samples.npz`) — GB-scale, not Tier D | `nn_l1_sbi.inference.run_joint_gate_c` → `tarp_stratified_val` + `run_tarp_coverage` | **GPU**, one job per arm per basis, 16 in total (3 NDE seeds × 600 val points each) | `fig:tarp`, `fig:tarp_bnt`, `fig:sbc` |
| `tarp_proper_bands{,_bnt}.npz` | `nn_l1_sbi.figures.fig_tarp --recompute` (once the gate dumps exist) | CPU, ~5–6 min | `fig:tarp{,_bnt}` from cache thereafter |
| `operator_realnvp/{conv,both}/n9000/median_summary.json` | `nn_l1_sbi.inference.population_sweep` on the conv / both operator arms — **not** the retrained `conv_nobnt_s41` / `joint_l1_nobnt_s41` sweeps, which are a different pipeline run | **GPU**, ~2 h per arm × 2 | `fig:ablation` |
| whitening / BNT campaign `median_summary.json` | `nn_l1_sbi.inference.population_sweep` per campaign arm | **GPU**, ~2 h per arm × 6 | `fig:bntwhiten` |
| `l1_hist_vs_s8_levels_cache.npz`, `geometry/<arm>/per_patch_grid.csv` | no surviving generator; both are derivable from products that do survive (the raw ℓ1 caches; a per-patch geometry run) | CPU, minutes (level cache — on a compute node, the caches are 1.04 GB each) / GPU (geometry grid) | `fig:l1` (a), `fig:w0` |

`fig_embed.py --recompute` and `fig_tarp.py --recompute` are the only figure scripts that will
rebuild their own cache; every other generator reads and never writes to the data root.

---

## 4. Design conventions

These are what make the set read as one figure family. Anything that changes them changes several
figures at once.

* **Widths.** Single column **88 mm** (3.465 in), double-column `figure*` **180 mm** (7.087 in),
  intermediate 120 mm. `figsize` is set explicitly per figure; LaTeX never scales a figure.
* **Type.** Serif, STIX math. Axis labels 9 pt, ticks 8 pt, legend 8 pt — at final printed size.
* **Ticks.** Inward, all four sides, minor ticks on. No in-axes titles: the caption carries the
  description.
* **Output.** Vector PDF with fonts embedded (`pdf.fonttype 42`). Heavy `imshow` panels are
  `rasterized=True` at 300 dpi *inside* the vector PDF.
* **Colourmaps.** Sequential: cividis / viridis (grayscale- and colourblind-safe). Diverging
  (signed κ): `RdBu_r`.
* **Fixed per-arm palette** (Okabe–Ito), imported from `style.ARMS` everywhere so an arm's colour
  is identical across the whole set:

  | arm | key | hex | marker |
  |---|---|---|---|
  | ℓ1 auto-only | `l1_auto` | `#666666` (grey) | ▽ |
  | ℓ1 + product | `l1_product` | `#D55E00` (vermillion) | ● |
  | joint ℓ1 (ensemble) | `jointl1` | `#009E73` (bluish-green) | ■ |
  | CNN (ResNet-18) | `cnn` | `#0072B2` (blue) | ▲ |

* **Linestyle is semantic, not an arm identifier.** Every arm is drawn **solid** in `style.ARMS`;
  arms are told apart by colour, marker and distinct luminance. Dash pattern is then free to carry
  meaning: **solid = analytical / no-BNT, dashed = learned / BNT** (`fig:corner`, `fig:bnt`,
  `fig:sbc`). In `fig:bnt` the BNT ring uses the *same base hue* as its no-BNT arm, dashed rather
  than darkened, so the hue stays legible.
* **Layout.** `fig_l1.py`, `fig_embed.py`, `fig_fisher.py` and `fig_sbc.py::plot_hist_combined`
  turn `figure.constrained_layout.use` off and set explicit GridSpec margins, then save with the
  default `savefig` (no tight bbox), so each PDF lands at exactly its nominal column width. Do not
  re-enable constrained layout globally in `paper.mplstyle`.

---

## 5. How to read the calibration figures

### 5.1 The SBC rank and its band

The SBC rank is `r = (1/M) · #{posterior samples < truth}`, uniform on `[0,1]` under calibration.

The load-bearing methodological point: each gate holds **N = 600** prior-drawn test points per NDE
seed (3 FoM₃ terciles × 200), evaluated under 3 NDE re-trainings (seeds 41/42/43) on *identical*
test θ (verified byte-equal). The independent unit is therefore **one seed, N = 600** — not the
pooled 1800. So:

* bars and curves are the **mean of the three per-seed histograms/ECDFs**;
* bands are sized to **N = 600**. Pooling to 1800 would shrink the band and over-reject.

Two bands are in use:

* **Histogram band** — central 99 % of `Binomial(N=600, 1/20)` per bin, converted to density. It is
  **pointwise**: a single bin grazing the edge is not by itself significant.
* **ECDF band** — a **simultaneous** 99 % band (Säilynoja, Bürkner & Vehtari 2022): constant
  half-width `d₉₉ = 0.066`, the 99th percentile of `sup_r |F̂(r) − r|` for 600 uniforms (matches the
  analytic KS value `1.628/√600`). A curve leaving it *anywhere* is significant.

The uniform rank-std target is `1/√12 ≈ 0.289`.

### 5.2 Shape dictionary

**Rank histogram.** Flat within the band = calibrated. **U-shape** (excess at `r = 0` and `r = 1`)
= posterior too narrow = **over-confident**. **n-shape** (excess in the middle) = too wide =
**under-confident**. **Slope / asymmetry** = biased mean: more mass at low `r` means the truth sits
low in the posterior, i.e. the summary **over-estimates** that parameter; more mass at high `r`
means it **under-estimates** it.

**ECDF difference `F̂(r) − r`.** Pinned to 0 at both ends, so all the information is interior.
**Overall tilt or a one-sided hump = bias** (above 0 → over-estimates, below 0 → under-estimates).
An **antisymmetric S crossing 0 near `r = 0.5` = dispersion error** (positive-then-negative = U
histogram = over-confident; negative-then-positive = under-confident). Integrating the ranks
smooths bin noise and leaves only the coherent, low-order departures — the ones that matter.

Dispersion and bias are different axes: the rank-std sees only magnitude, the ECDF tilt adds the
sign. Read the two together.

### 5.3 What the calibration figures support — and what they do not

**Headline (no-BNT).** Every arm stays inside the 99 % band in all three of Ωm, σ₈ and w₀, in both
representations. There is no significant miscalibration, so the constraining-power differences
between arms are information, not calibration artifacts.

**Sub-threshold structure is not a claim.** Directional reads of curves that stay inside the band
(a hump in σ₈ here, a dip there) are *not* significant — the simultaneous band exists precisely to
stop over-reading smooth wiggles. The defensible statement is "all arms are calibrated; residual
departures are within the expected scatter." A directional claim would need a quantitative test on
the per-seed scatter (a slope test, or a C2ST on the ranks), not eyeballing.

**BNT, read honestly.** `sup_r |F̂(r) − r|` against `d₉₉ = 0.066` (mean over seeds; `>` = exceeds):

| arm | no-BNT (Ωm / σ₈ / w₀) | BNT, single compressor (Ωm / σ₈ / w₀) |
|---|---|---|
| ℓ1 auto | 0.043 / 0.036 / 0.057 | **0.077> / 0.077> / 0.068>** |
| ℓ1 + product | 0.041 / 0.054 / 0.049 | 0.054 / **0.072>** / 0.062 |
| joint ℓ1 | 0.024 / 0.024 / 0.036 | 0.038 / 0.039 / 0.049 |
| CNN | 0.061 / 0.063 / 0.050 | 0.055 / 0.056 / 0.029 |

With a single compressor the per-channel BNT arms are *not* all calibrated: ℓ1 auto-only exceeds in
all three parameters and ℓ1+product is borderline in σ₈. BNT per-channel inflation is not only a
FoM loss — it leaves a calibration footprint.

**The deep ensemble is the fix, and that is the point.** Pooling 3 *compressor* seeds nearly halves
the miscalibration, so the over-confidence is compressor/NDE variance rather than a hard frame
artifact:

| arm (BNT) | single, max sup | 3-compressor ensemble, max sup | rank-std single → ensemble |
|---|---|---|---|
| ℓ1 + product | 0.072 (σ₈, crosses) | **0.055 (inside)** | 0.313 → 0.302 |
| ℓ1 auto | 0.077 (Ωm/σ₈, crosses) | **0.065 (w₀, just inside)** | 0.320 → 0.297 |

The shipped BNT SBC figures use these ensembles (`_data.GATE_ENSEMBLE_BNT_GLOBS`, pooled on the
tercile key), so all four BNT arms read calibrated: max sup auto 0.065, +product 0.055, joint ℓ1
0.049, CNN 0.056. The same check on the no-BNT +product arm: single `0.041/0.054/0.049` → ensemble
`0.021/0.026/0.032` (std `0.295/0.291/0.293`, essentially ideal).

### 5.4 The TARP band

The TARP curve is the expected coverage probability (ECP) vs credibility α, and its band must show
how far that **measured curve** can move given a finite validation set. `tarp.get_tarp_coverage`'s
built-in `bootstrap=True` resamples only the random *reference points* (per-bin std ~10⁻⁴), which
understates the real uncertainty by roughly two orders of magnitude. `fig_tarp.py` therefore builds
the band itself:

1. Form the reported posterior over the **600 un-stratified sightlines**: concatenate the three
   FoM₃ terciles (LOW+MID+HIGH = 600) and pool seeds/compressors along the **sample axis M** — so
   it stays 600 sightlines, not the 1800 you would get by stacking the 3 NDE-seed re-trainings as
   separate rows. ℓ1+product and CNN pool 3 NDE seeds (K = 6000); joint ℓ1 pools the 3-compressor
   ensemble (K = 18000). Cross-compressor pooling uses the tercile key.
2. **Sightline bootstrap:** for B = 200 resamples, draw the 600 sightlines with replacement,
   recompute ECP via `get_tarp_coverage(references="random", norm=True)` (a single curve, *no*
   `bootstrap=True`), interpolate onto a common α grid; the band is the 16–84 percentile (1σ).

This gives 1σ ≈ ±0.019 at α = 0.5, matching the binomial SE `√(0.25/600)`. The net statistic is
`net = 2∫(ECP − α) dα`: positive = conservative / over-covers, negative = over-confident. The CNN's
mildly conservative +0.032 is ≈1.4σ — its 1σ net interval only just excludes 0, so it is not
significant, and it errs on the safe side.

---

## 6. Headline numbers, one place

Several figures hard-code these so that the plotted number and the manuscript number cannot drift.
No-BNT values are the single-compressor headline; joint ℓ1 is its 3-compressor ensemble.

| arm | no-BNT | BNT | retention |
|---|--:|--:|--:|
| ℓ1 auto-only | 2448 | **388** (ensemble; single 425) | ×0.16 |
| ℓ1 + product | 3045 | **718** (ensemble; single 779) | ×0.24 |
| joint ℓ1 | 3371 | 2424 | ×0.72 |
| CNN | 3326 | 3186 | ×0.96 |

Operator ladder in the matched RealNVP frame (`fig:ablation`):
auto 2448 → +conv 2671 → +product 3045 → +both 3255 → joint ℓ1 3371.

Two consistency notes that belong with the figures:

* The BNT per-channel FoM₃ quoted in the `fig:bnt` inset (auto 388, +product 718) are the
  calibrated 3-compressor **ensemble** values, while the plotted BNT *contours* are still
  single-compressor (auto 425, +product 779). The visual difference is ≈4 %; the message is
  unchanged.
* `fig:corner`'s inset uses the committed 9000-observation medians. The mean-observation contour it
  draws for the ℓ1 auto arm has a single-observation FoM₃ of 2273 — a different estimator, not a
  disagreement.
* **Compressor ensembling moves the no-BNT analytical FoM₃ by several per cent, not by ≤ 1.2 %.**
  Measured single → 3-compressor ensemble shift, read from the shipped
  `data/table1/TABLE1_ERRORBARS_V2.json` (field `bias_pct`, on the `per_seed.s41` baseline that
  file uses):

  | arm (no-BNT) | single, s41 | 3-compressor ensemble | shift |
  |---|--:|--:|--:|
  | ℓ1 auto-only | 2776.9 | 2511.1 | **−9.57 %** |
  | ℓ1 + product | 3231.8 | 3053.9 | **−5.50 %** |
  | joint ℓ1 | 3762.3 | 3379.5 | −10.17 % |
  | CNN | 3427.6 | 3324.2 | −3.02 % |

  Two things this is not. It is **not** the ± in Table 1: that is `std(ddof=1)/mean` over the
  three *pre-ensemble* compressor singles (auto 2448 ± 27, +product 3045 ± 183, …). And it is
  **not** the "< 1.2 %" figure quoted elsewhere in the docs, which is a different axis — the
  effect of pooling the three **NDE (flow) seeds**, not the three **compressor** seeds. Keep the
  two apart. The single → ensemble shift is a **bias**, reported separately from the error bar,
  and it is negative because ensembling removes over-confidence (see §5.3). The numbers above
  are from the retrained campaign, so they are the size of the effect, not a correction to the
  published centrals.

`+conv` and `+both` were deliberately not added under BNT: they are still per-channel statistics
(they would collapse like `+product`), and the BNT datavector for them was never built.

---

## 7. Third-party dependencies, by generator

| package | required by |
|---|---|
| `numpy`, `matplotlib` | all |
| `getdist` | `fig_corner.py`, `fig_bnt.py` |
| `scipy` (`stats.binom`) | `fig_sbc.py` |
| `healpy` | `fig_leak.py`, `fig_leak_maps.py` |
| `tarp` | `fig_tarp.py` |
| `scikit-learn` (`manifold.TSNE`) | `fig_embed.py` — its only consumer |
| `jax` + the jaxili / sbi_lens stack | `dump_contour_samples.py`, `gen_auto_contour.sh` (**GPU only**) |
| stdlib `csv` / `json` | `fig_w0.py` / `fig_ablation.py`, `fig_bntwhiten.py` |

`fig_data.py`, `fig_leak.py` and `fig_leak_maps.py` each call
`os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")` before importing anything else — that is what
makes them safe to run on a shared login node. Keep it. (`setdefault`, so an explicit
`CUDA_VISIBLE_DEVICES` in the environment still wins.)

---

## 8. Runtime summary

| tier | generators | cost |
|---|---|---|
| seconds, CPU | `fig_data`, `fig_l1`, `fig_pipeline`, `fig_ablation`, `fig_cp`, `fig_w0`, `fig_bntwhiten` | trivial |
| ~10–60 s, CPU | `fig_corner`, `fig_bnt` (getdist KDE), `fig_fisher` (800-column `lstsq`), `fig_leak`, `fig_leak_maps` (healpy) | fine on a laptop |
| ~1–2 min, CPU | `fig_sbc` (loads the whole gate-dump tree for both bases) | fine |
| ~6 min, CPU | `fig_tarp --recompute` (4 arms × 200 bootstraps × 2 bases) | cached by default |
| minutes, CPU | `fig_embed --recompute` (3 × t-SNE on 6000×10) | cached by default |
| **hours, GPU** | `dump_contour_samples.py`, `gen_auto_contour.sh` | run once; the dumps are then reused |

---

## 9. References

* Talts, Betancourt, Simpson, Vehtari & Gelman 2018, arXiv:1804.06788 — SBC, rank histograms.
* Säilynoja, Bürkner & Vehtari 2022, *Stat. Comput.* **32**, 32, arXiv:2103.10522 — graphical
  uniformity test, ECDF with simultaneous bands.
* Lemos, Coogan, Hezaveh & Perreault-Levasseur 2023, arXiv:2302.03026 — TARP, the sibling
  expected-coverage test.
