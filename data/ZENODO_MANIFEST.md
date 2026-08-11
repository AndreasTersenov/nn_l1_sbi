# Data record specification — `nn_l1_sbi` figure and table inputs

This document specifies the data record that backs the figures and tables of *"Do we need deep
learning for tomographic weak lensing? Wavelets versus neural compression"*: what is in it, how
large each part is, what the arrays and JSON objects inside contain, and how to point this
package at an unpacked copy.

Every byte count below is an exact file size, measured on the files that go into the record.
The MB/GB shorthand is decimal (10⁶ / 10⁹).

> ## Status: the record is not published yet
>
> The archives described here have not been uploaded, and **the DOI in this document is a
> placeholder** (`10.5281/zenodo.XXXXXXX`). Tier D (`derived/`) and the gate-dump subtree of
> the `analysis/` tier still have to be regenerated before the record can claim to reproduce
> the figure section of the paper — see §9. Everything else is measured, schema-checked and
> ready to stage.
>
> Two files in this repository carry the placeholder string and must be updated together once
> the DOI exists: this status note, and the **Data** section of `README.md`.
> `grep -rn 'zenodo.XXXXXXX' .` finds both. (`docs/FIGURE_MAP.md` and
> `notebooks/reproduce_paper.ipynb` describe the record but contain no placeholder DOI.)

Table 1 does **not** depend on any of this: its inputs are committed to the repository under
`data/table1/` (§4).

Documents this specification is kept consistent with:

| | |
|---|---|
| what the figure code resolves | `src/nn_l1_sbi/paths.py` (`data_root`, `output_root`) and `src/nn_l1_sbi/figures/_data.py` (`MAPS`, `DERIVED`, `ANALYSIS`, `ROOT`, `A`, `CNNP`, `PER_PATCH`, the gate globs) |
| figure → input → status | `docs/FIGURE_MAP.md` §1–3 |
| Tier-A contents and the Table-1 recipe | `data/table1/README.md` |

---

## 1. What the record contains

Two archives, so that a laptop user never has to download the GB tier:

| archive | contents | unpacked size | notes |
|---|---|--:|---|
| `nn_l1_sbi_data_v1.tar.gz` | Tiers B + C + D | **302.3 MB** | the portable record; unpacks to `nn_l1_sbi_data_v1/` |
| `nn_l1_sbi_analysis_v1.tar.gz` | the `analysis/` tier | **1.99 GB** (excluding gate dumps, §9.2) | needed only by the `--recompute` paths and the GPU contour dump; unpacks into the *same* root |
| `MANIFEST.sha256` | `sha256sum` of every file in both archives | — | `sha256sum -c MANIFEST.sha256` after unpacking |

Both archives share one top-level directory name, so unpacking either (or both) into the same
parent yields a single valid data root.

### 1.1 Tiers

| tier | where it lives | size | what it enables |
|---|---|--:|---|
| **A** | **committed in this repository**, `data/table1/` — *not* in the data record | **42,973 B** in 60 JSON files (+ a 17,181 B README) | Table 1: all 12 arms, FoM₃ ± and the three marginals. No download, no network, no GPU, < 1 s on a laptop |
| **B** | record, `sweeps/` + 5 duplicates under `analysis/` | **170,599,966 B (170.6 MB)** | `fig:cp` violins; recomputation of the block-bootstrap `median_se`; per-observation covariances |
| **C** | record, `maps/` + `tables/` | **121,740,654 B (121.7 MB)** | `fig:data`, `fig:pipeline`, `fig:leak` (a)+(b); the noise-model provenance table |
| **D** | record, `derived/` | **~10 MB** | `fig:corner`, `fig:bnt`, `fig:tarp{,_bnt}`, `fig:sbc`, `fig:l1` (a), `fig:embed`, `fig:w0` — **not yet regenerated, see §9** |
| **analysis** | record, second archive, `analysis/` | **1,986,076,690 B (1.99 GB)** excluding gate dumps | `fig:fisher`, `fig:l1joint`, `fig_embed --recompute`, `fig_tarp --recompute`, and the one-time GPU contour dump that *produces* Tier D |

`docs/FIGURE_MAP.md` quotes Tier B as 153 MB. That is the 54 `sweeps/` files alone
(153,154,572 B); the extra 17,445,394 B counted here is the five `fig:cp` files duplicated into
the frozen `analysis/` layout that `_data.PER_PATCH` addresses (§5.2).

### 1.2 What you can draw for a given download

| you download | size | you can draw |
|---|--:|---|
| nothing (clone only) | 43 kB | Table 1 |
| + Tier B | 170.6 MB | + `fig:cp`, + the bootstrap term recomputed rather than quoted |
| + Tier C | 292.3 MB | + `fig:data`, `fig:pipeline`, `fig:leak` |
| + Tier D | ~302 MB | + `fig:corner`, `fig:bnt`, `fig:tarp{,_bnt}`, `fig:sbc`, `fig:l1` (a), `fig:embed`, `fig:w0` — **every figure in the paper except `fig:fisher` and `fig:l1joint`** |
| + analysis | ~2.29 GB | + `fig:fisher`, `fig:l1joint`, and every `--recompute` path |

### 1.3 Deliberately not included

* `sweeps/*/per_obs_samples.npy` — 54 × 216,000,128 B = **11.66 GB** of per-observation
  posterior draws. No script in this repository reads them, and they are **not** a substitute
  for the contour dumps of Tier D (§11, trap 6).
* The 800-D validation caches (only the 800-D *training* cache is a figure input).
* The raw ℓ1 datavector caches for the `conv`, `product` and `both` operators (~35 GB).
  Only the `none` (auto) operator cache is a figure input, and it ships (§8).

---

## 2. The unpacked tree

This is exactly what `nn_l1_sbi.figures._data` resolves. `<root>` is `$NN_L1_SBI_DATA`, or
`~/.cache/nn_l1_sbi/data` if that variable is unset (§3).

```
<root>/                                    # = nn_l1_sbi.paths.data_root()
├── maps/                                  # = _data.MAPS                       [tier C]
│   ├── cosmo_fiducial_perm0.npz
│   └── fullsphere_nobnt_cosmo_fiducial_perm0.npz
├── derived/                               # = _data.DERIVED                    [tier D]
│   ├── contour_nobnt_samples.npz
│   ├── contour_bnt_samples.npz
│   ├── tarp_proper_bands.npz
│   ├── tarp_proper_bands_bnt.npz
│   ├── embed_tsne.npz
│   ├── l1_hist_vs_s8_levels_cache.npz
│   └── geometry/
│       ├── l1_auto_cross/per_patch_grid.csv
│       └── cnn_auto_cross/per_patch_grid.csv
├── sweeps/                                # notebook-only; same 54 dir names as data/table1/
│   └── <54 dirs>/per_patch_metrics.npz                                         [tier B]
├── tables/                                # provenance only; no figure script reads it
│   ├── flatsky_cross_noise_sigma.npz                                           [tier C]
│   └── flatsky_cross_noise_sigma.json
└── analysis/                              # = _data.ANALYSIS  ( = _data.CNN_SBI )
    └── flatsky_cross_2026_06/             # = _data.ROOT
        ├── analytical_nde_match/          # = _data.A
        │   ├── l1product_rnvp_s41_n9000/per_patch_metrics.npz                  [tier B copy]
        │   ├── jointl1_nobnt_ensemble/per_patch_metrics.npz                    [tier B copy]
        │   ├── product_bnt_ensemble/per_patch_metrics.npz                      [tier B copy]
        │   ├── jointl1_bnt_ensemble/per_patch_metrics.npz                      [tier B copy]
        │   ├── jointl1_nobnt_raw/fid.npz                                       [analysis]
        │   ├── l1product_vmim_s41/{cache/{l1_train,l1_val}.npz, fiducial_summaries.npz}
        │   ├── l1product_bnt_vmim_s41/{cache/{l1_train,l1_val}.npz, fiducial_summaries.npz}
        │   ├── jointl1_nobnt/     {cache/{l1_train,l1_val}.npz, fiducial_summaries.npz}
        │   ├── jointl1_nobnt_s42/ {cache/{l1_train,l1_val}.npz, fiducial_summaries.npz}
        │   ├── jointl1_nobnt_s43/ {cache/{l1_train,l1_val}.npz, fiducial_summaries.npz}
        │   ├── jointl1_bnt/       {cache/{l1_train,l1_val}.npz, fiducial_summaries.npz}
        │   ├── jointl1_bnt_s42/   {cache/{l1_train,l1_val}.npz, fiducial_summaries.npz}
        │   ├── jointl1_bnt_s43/   {cache/{l1_train,l1_val}.npz, fiducial_summaries.npz}
        │   ├── bnt_auto/          {cache/{l1_train,l1_val}.npz, fiducial_summaries.npz}
        │   ├── operator_realnvp/none/{cache/{l1_train,l1_val}.npz, fid_summ.npz}
        │   ├── operator_realnvp/{conv,both}/n9000/median_summary.json          [absent — §9.5]
        │   ├── gate_l1none_rnvp/tarp_drp/dumps/…                               [absent — §9.2]
        │   ├── gate_l1product_rnvp/tarp_drp/dumps/…                            [absent — §9.2]
        │   ├── gate_l1product_bnt_rnvp/tarp_drp/dumps/…                        [absent — §9.2]
        │   ├── {jointl1_nobnt,jointl1_nobnt_s42,jointl1_nobnt_s43}/gate/tarp_drp/dumps/…  [absent]
        │   ├── {jointl1_bnt,jointl1_bnt_s42,jointl1_bnt_s43}/gate/tarp_drp/dumps/…        [absent]
        │   ├── bnt_auto/gate/tarp_drp/dumps/…                                  [absent — §9.2]
        │   └── {ens_bnt_auto_s42,ens_bnt_auto_s43,ens_bnt_product_s42,ens_bnt_product_s43}/gate/tarp_drp/dumps/…  [absent]
        ├── cnn_phase/                     # = _data.CNNP
        │   ├── arch_sweep_2026_06_13/
        │   │   ├── cnn_resnet18_s41/readout_full/per_patch_metrics.npz         [tier B copy]
        │   │   ├── cnn_resnet18_s41/cache/{cnn_train,cnn_val}.npz
        │   │   ├── fidsumm_resnet18.npz
        │   │   └── gate_c_resnet18/dumps/…                                     [absent — §9.2]
        │   └── bnt_resnet18_2026_06_14/
        │       ├── cnn_resnet18_bnt_s41/cache/{cnn_train,cnn_val}.npz
        │       ├── fidsumm_bnt_resnet18_s41.npz
        │       └── gate_c_bnt/dumps/…                                          [absent — §9.2]
        ├── l1_matrix/l1_none_cache/flat_local_none/{l1_train.npz, flat_local_ranges.npy}
        ├── gate_c/lc2st/fiducial_summaries_none.npz                            [optional, §8.1]
        ├── population_sweep/flat_{none,product}/median_summary.json            [absent — §9.6]
        ├── whiten_campaign/population_sweep/l1_{none,product}/median_summary.json  [absent]
        ├── bnt_campaign/population_sweep/l1_{none,product}/median_summary.json     [absent]
        ├── bntdeep_campaign/population_sweep/l1_none/median_summary.json       [absent, optional]
        └── bntdeep2_campaign/population_sweep/l1_none/median_summary.json      [absent, optional]
```

Two conventions are load-bearing and must not be "tidied":

1. **`analysis/` mirrors the original analysis tree verbatim**, including directory names that
   no longer match the corresponding `sweeps/` names (`l1product_rnvp_s41_n9000`,
   `cnn_resnet18_s41/readout_full`, `operator_realnvp/none`), and two files whose names differ
   from their siblings' (`fid_summ.npz`, `fidsumm_resnet18.npz`). `_data.py` builds those paths
   by string concatenation, so renaming a directory in the bundle breaks a figure silently.
2. `sweeps/<dir>/` uses the **sweep names**, because `data/table1/arms.json` addresses the arms
   by those names. A notebook that reads `arms.json` and swaps `median_summary.json` →
   `per_patch_metrics.npz` under `<root>/sweeps/` works with no name translation. The five
   files `fig:cp` needs are therefore *duplicated* into the `analysis/` layout (17,445,394 B of
   duplication — cheap, and it keeps both consumers honest).

---

## 3. Pointing the code at the record

```bash
# 1. download the archives from the record (see the status note above: not published yet)
tar xzf nn_l1_sbi_data_v1.tar.gz          # -> ./nn_l1_sbi_data_v1/
tar xzf nn_l1_sbi_analysis_v1.tar.gz      # optional heavy tier, into the same parent
sha256sum -c MANIFEST.sha256

# 2. point the code at it
export NN_L1_SBI_DATA=$PWD/nn_l1_sbi_data_v1

# 3. draw
python -m nn_l1_sbi.figures.fig_cp
python -m nn_l1_sbi.figures.fig_corner
```

`nn_l1_sbi.paths.data_root()` resolves **`$NN_L1_SBI_DATA`, else `~/.cache/nn_l1_sbi/data`** —
those are the only two possibilities. Unpacking the archives so that the root *is*
`~/.cache/nn_l1_sbi/data` therefore works with no environment variable at all. Figures and any
regenerated products are written under `nn_l1_sbi.paths.output_root()` (`$NN_L1_SBI_OUT`, else
`./nn_l1_sbi_out`), never into the record.

The notebook `notebooks/reproduce_paper.ipynb` **downloads nothing**: you download and unpack
the archives yourself, and either export `NN_L1_SBI_DATA` before starting the kernel or set
`os.environ["NN_L1_SBI_DATA"]` in the session — the notebook reloads `_data` so a value set in
the session is honoured. Section B then prints a status table and skips whatever is missing.

> ### ⚠️ Set the variable *before* importing the package
> `_data.MAPS`, `_data.DERIVED`, `_data.ANALYSIS`, `_data.ROOT`, `_data.A` and `_data.CNNP` are
> module-level constants computed from `data_root()` at **import time**. Changing
> `os.environ["NN_L1_SBI_DATA"]` after `nn_l1_sbi.figures._data` has been imported has no
> effect unless the module is reloaded.

> ### ⚠️ `$NN_L1_SBI_DATA` is overloaded
> `scripts/slurm/env.sh` sets the **same** variable to a *writable compute* root (for caches,
> arms, sweeps and logs). If you source `env.sh` in a shell and then run a figure script in that
> shell, `data_root()` points at the compute tree and every figure fails with a `require()`
> error naming a file that is not missing, just elsewhere. Run figure scripts in a clean shell,
> or re-export `NN_L1_SBI_DATA` to the record afterwards.

A missing input never produces a wrong plot: `nn_l1_sbi.paths.require()` and
`_data.require_glob()` raise a message naming the file, the generator that produces it, and the
currently resolved data root.

---

## 4. Tier A — in the repository, not in the record

`data/table1/`: 60 JSON files, **42,973 B**, plus a 17,181 B `README.md` that documents them in
full (schemas, the `bias_pct` baseline trap, the `joint_l1_*` vs `jointl1_*` naming trap, and
the verified recipe). Summarised here only so this specification is self-describing:

| file | bytes | schema |
|---|--:|---|
| `arms.json` | 3,869 | 12 keys = canonical arm labels; each `{singles:{s41,s42,s43 → "sweeps/<dir>/median_summary.json"}, ensemble: "…", published: int\|null, quoted: "single"\|"ensemble"}` |
| `sweeps/<54>/median_summary.json` | 15,345 total (282 B singles / 287 B ensembles) | flat dict: `arm` str, `n` int 9000, `n_members` int 1\|3, `nde_seeds` [41,42,43], `fom3` float (median over 9000 mocks), `sigma_Om`, `sigma_s8`, `sigma_w0` float, `m_samples` int 2000, `elapsed_s` float |
| `TABLE1_ERRORBARS_V2.json` | 8,414 | 12 arms → `published`, `quoted`, `per_seed{s41,s42,s43}`, `mean`, `std`, `lo`, `hi`, `pct`, `ensemble`, `bias_pct`, `central`, `median_se`, `median_ci68_half`, `population{n, CV_pop, rho, n_blocks, reps_per_block}` |
| `TABLE1_CONV_ARMS.json` | 5,384 | 4 conv arms → `published_fom`, `source`, `quoted`, `fom_central`, `fom_pm`, `fom_rel_pct`, `fom_per_seed[3]`, `fom_ensemble`, `bias_pct`, `sigma{sigma_Om,sigma_s8,sigma_w0}` each `{central, pm, per_seed[3], ensemble, rel_pct}` |
| `TABLE1_PAPER.json` | 1,921 | 8 rows → `published`, `pm`, `rel_pct`, `n_seeds`, `median_se`, `retrained`, `pm_retrained`, `n6_seeds`, `n6_rel_pct`, `n6_pm` |
| `TABLE1_MARGINALS_NEW.json` | 974 | 2 rows → `published_fom`, `fom_pm`, and `{value, pm, rel_pct}` per σ |
| `final_bars_all_v2.json` | 7,066 | **list** of 8 dicts → `row`, `per_seed_median`, `median_term{se, blocks, p16, p84, ci68_half_width, reps_per_patch}`, `population_stats{…}`, `ensemble_term{K, theta_leave_one_out[3], theta_bar, se}`, `validation{fom_ok}`. `ensemble_term` is a jackknife **diagnostic, never quotable** |

---

## 5. Tier B — `sweeps/` (170,599,966 B)

### 5.1 `sweeps/<dir>/per_patch_metrics.npz` (54 files, 153,154,572 B)

`<dir>` runs over exactly the 54 names present in `data/table1/sweeps/`: 48 Table-1 dirs
(12 arms × {s41, s42, s43, ensemble}) plus the 6 n=6 caveat dirs `auto_bnt_s4{4,5,6}` and
`jointl1_bnt_s4{4,5,6}`. Sizes are near-uniform by class: **2,452,150–2,452,198 B** for every
single (`n_members = 1`, K = 1) and **4,180,282–4,180,378 B** for every ensemble (K = 3).
These are uncompressed `np.savez` zips, so gzipping the tarball recovers ~15–20 %.

Schema (K = 1 for singles, 3 for ensembles) — verified on `product_nobnt_s41`:

```
sigma        (9000, 3)        float64   per-obs marginal sigmas, order = param_order
fom2d        (9000, 3)        float64   the three 2-D FoMs
fom3         (9000,)          float64   FoM3 per mock observation      <-- Table 1 / fig:cp
perm         (9000,)          int32     noise realisation index, 0..49
patch        (9000,)          int32     patch index, 0..179            <-- bootstrap blocks
sel          (9000,)          int64     flat selection index
truth        (0,)             float64   EMPTY in every file inspected
mean         (9000, 3)        float64   pooled posterior mean
cov          (9000, 3, 3)     float64   pooled posterior covariance
n_pooled     (9000,)          int64     6000 (single) / 18000 (ensemble)
arm_mean     (K, 9000, 3)     float64   per-ensemble-member mean
arm_cov      (K, 9000, 3, 3)  float64   per-ensemble-member covariance
arm_dirs     (K,)             <U…       e.g. ['product_nobnt_s41']
param_order  (3,)             <U7       ['Omega_m' 'sigma_8' 'w_0']
arm_label    ()               <U…       e.g. 'product_nobnt_s41'
nde_seeds    (3,)             int64     [41 42 43]
```

`np.unique(patch)` → 180 values, `np.unique(perm)` → 50; 180 × 50 = 9000. Written by
`nn_l1_sbi.inference.population_sweep`. Each file stores its own provenance in `arm_label` and
`arm_dirs`: check those rather than trusting a directory name.

**Optional trim.** A derived `per_patch_slim.npz` keeping only `fom3`, `sigma`, `patch` and
`perm` is ~0.3 MB per dir (**~17 MB total**) and serves `fig:cp` and the bootstrap unchanged;
it drops the covariances. Not recommended — 153 MB is affordable, and the full file is the one
whose contents were verified.

### 5.2 The five Tier-B duplicates under `analysis/` (17,445,394 B)

`_data.PER_PATCH` addresses these by their analysis-tree names. Each is a byte-identical copy of
one `sweeps/` file. Copy, do not rename:

| bundle path (what the code opens) | duplicate of | bytes | consumer |
|---|---|--:|---|
| `analysis/flatsky_cross_2026_06/analytical_nde_match/l1product_rnvp_s41_n9000/per_patch_metrics.npz` | `sweeps/product_nobnt_s41/` | 2,452,174 | `fig:cp` (`l1_product`) |
| `analysis/flatsky_cross_2026_06/analytical_nde_match/jointl1_nobnt_ensemble/per_patch_metrics.npz` | `sweeps/jointl1_nobnt_ensemble/` | 4,180,378 | `fig:cp` (`jointl1`) |
| `analysis/flatsky_cross_2026_06/cnn_phase/arch_sweep_2026_06_13/cnn_resnet18_s41/readout_full/per_patch_metrics.npz` | `sweeps/cnn_nobnt_s41_pnone/` | 2,452,198 | `fig:cp` (`cnn`) |
| `analysis/flatsky_cross_2026_06/analytical_nde_match/product_bnt_ensemble/per_patch_metrics.npz` | `sweeps/product_bnt_ensemble/` | 4,180,298 | declared in `PER_PATCH`, read by no shipping figure |
| `analysis/flatsky_cross_2026_06/analytical_nde_match/jointl1_bnt_ensemble/per_patch_metrics.npz` | `sweeps/jointl1_bnt_ensemble/` | 4,180,346 | idem |

> ### ⚠️ `fig:cp` medians in this record differ from the printed figure
> The `per_patch_metrics.npz` files here come from a **later re-run** of the sweeps, not from
> the run that produced the published PDF. Their FoM₃ medians — which `fig_cp.py` prints on the
> violins — are **3231.75 / 3379.51 / 3427.64** (`l1+product` / joint ℓ1 / CNN) against the
> published **3045 / 3371 / 3326**, i.e. **+6.1 % / +0.3 % / +3.1 %**. Redrawing `fig:cp` from
> this record therefore yields a figure whose annotated medians differ from the printed one.
> This is the expected size of the compressor-training-seed term (`docs/NOTE_FOM_ERROR_BARS.md`
> §3), not a bug — but it must be stated in the figure caption of any redrawn version.

---

## 6. Tier C — `maps/` and `tables/` (121,740,654 B)

**`maps/cosmo_fiducial_perm0.npz`** · **33,852,223 B** · consumers `fig:data` and `fig:pipeline`
(both read `patches`, patch index 90). One noise permutation of the fiducial 10° patch cache;
only `perm0` ships.

```
patches            (180, 80, 80, 10) float32   180 10deg patches x 4 auto + 6 cross channels
patch_centers      (180, 2)          float32
theta              (6,)              float64
cosmo_id           ()  <U14      cosmo_idx ()  int64     perm ()  int64
regime             ()  <U5       split     ()  <U3       noise_seed () int64
sigma_e            ()  float64   galaxy_density () float64
nside              ()  int64     lmax      ()  int64
field_size         ()  float64   field_npix () int64     reso_arcmin () float64
map_label          ()  <U9       bnt_matrix_version () <U4
```

**`maps/fullsphere_nobnt_cosmo_fiducial_perm0.npz`** · **87,878,466 B** · consumers `fig:leak`
(a) via `fig_leak.py` and (b) via `fig_leak_maps.py` (needs `healpy`).

```
full_auto    (4, 3145728) float32    4 tomographic auto maps, HEALPix nside=512
full_cross   (6, 3145728) float32    6 cross-map channels
cosmo_id ()  <U14    perm () int64    regime () <U5
nside    ()  int64   lmax () int64    noise_seed () int64
```

**`tables/flatsky_cross_noise_sigma.npz`** · **6,209 B**, plus its JSON twin
`tables/flatsky_cross_noise_sigma.json` · **3,756 B** (same content plus `constants` and
`gate_a1b`). This is the provenance of the noise model; **no figure script reads it**. It ships
so that the record documents the noise calibration. Written by
`nn_l1_sbi.flatsky.freeze_flatsky_cross_noise`.

```
sigma (16,5) f8 · channel_names (16,) <U9 · n_scales () i8 · white_per_scale (5,) f8
sigma_pix () f8 · n_real () i8 · nside () i8 · lmax () i8 · sigma_e () f8
galaxy_density () f8 · reso_arcmin () f8 · field_npix () i8 · seed_base () i8
provenance () <U151 · bnt () bool · mode () <U4 · op () <U4
```

---

## 7. Tier D — `derived/` (~10 MB, not yet regenerated)

These files do not exist yet; §9 says how to produce them. The sizes below are expected sizes,
and **the schemas are contracts, not observations**: a regeneration that violates one breaks
the figure that reads it.

| bundle path | expected size | required schema | read by |
|---|--:|---|---|
| `derived/contour_nobnt_samples.npz` | ~1.0 MB | one key per arm from `{l1_auto, l1_product, jointl1, cnn}`, each `(≈20000·n_compressors, 3)` float32 in order (Ω_m, σ₈, w₀); plus `truth` `(3,)`. `fig_corner.py` iterates `ORDER_ALL = [l1_auto, l1_product, jointl1, cnn]` filtered by `z.files`, so a 3-arm dump is valid (the auto arm is merged in afterwards) | `fig:corner`, `fig:bnt` |
| `derived/contour_bnt_samples.npz` | ~1.0 MB | same keys, BNT basis. `fig_bnt.py` requires the **same arm keys in both** files | `fig:bnt` |
| `derived/tarp_proper_bands.npz` | ~10 kB | `alpha` `(61,)` float64 = `np.linspace(0,1,61)`; and for each arm in `{l1_auto, l1_product, jointl1, cnn}` the four keys `<arm>_mean`, `<arm>_lo`, `<arm>_hi` each `(61,)` and `<arm>_net` scalar | `fig:tarp` |
| `derived/tarp_proper_bands_bnt.npz` | ~10 kB | same, BNT basis | `fig:tarp_bnt` |
| `derived/embed_tsne.npz` | ~0.4 MB | `S8` `(6000,)`, `Om` `(6000,)`, `emb_l1_product` `(6000,2)`, `emb_jointl1` `(6000,2)`, `emb_cnn` `(6000,2)` | `fig:embed` |
| `derived/l1_hist_vs_s8_levels_cache.npz` | ~1 MB | `lvl_s8` `(B,)` — the σ₈ level values; `lvl_X` `(B, ≥3200)` — mean ℓ1 histogram per level, flat layout `channel*200 + scale*40 + snr_bin` with 16 channels × 5 scales × 40 SNR bins. `fig_l1.py` indexes channels 3 (auto κ₄), 9 (conv 3×4) and 15 (product 3×4) by arithmetic | `fig:l1` (a) |
| `derived/geometry/l1_auto_cross/per_patch_grid.csv` | ~100 kB | CSV with a header; the only column read is `pull_w_0` (float, NaNs tolerated) | `fig:w0` |
| `derived/geometry/cnn_auto_cross/per_patch_grid.csv` | ~100 kB | same | `fig:w0` |

---

## 8. The `analysis/` tier (1,986,076,690 B)

Paths are relative to `<root>/analysis/flatsky_cross_2026_06/` (= `_data.ROOT`).

| bundle path | bytes | schema | consumer |
|---|--:|---|---|
| `l1_matrix/l1_none_cache/flat_local_none/l1_train.npz` | **1,043,415,858** | `theta (323640,6) f4`, `x (323640,800) f4` — the raw per-bin starlet-ℓ1 auto datavector (4 tomo bins × 5 scales × 40 SNR bins) | `fig:fisher`; `gen_auto_contour.sh` step 1 |
| `l1_matrix/l1_none_cache/flat_local_none/flat_local_ranges.npy` | 384 | the adaptive SNR bin ranges | `nn_l1_sbi.compression.vmim_from_cache` |
| `analytical_nde_match/jointl1_nobnt_raw/fid.npz` | **432,289,800** | `S (36000,3000) f4`, `perm (36000,) i4`, `patch (36000,) i4`, `truth (6,) f8`, `stat () <U7`, `basis () <U5`, `k () i8`. `fig_l1.py` uses `S.mean(0).reshape(6, 5, 10, 10)` | `fig:l1joint` |
| `analytical_nde_match/l1product_vmim_s41/cache/l1_train.npz` | 20,713,458 | `theta (323640,6) f4`, `x (323640,10) f4` | `fig:embed`, contour dump |
| `analytical_nde_match/l1product_vmim_s41/cache/l1_val.npz` | 32,256,498 | `theta (504000,6) f4`, `x (504000,10) f4` | contour dump |
| `analytical_nde_match/l1product_vmim_s41/fiducial_summaries.npz` | 432,738 | `S (9000,10) f4`, `perm (9000,) i4`, `patch (9000,) i4` | contour dump |
| `analytical_nde_match/jointl1_nobnt/cache/{l1_train,l1_val}.npz` + `…/fiducial_summaries.npz` | 20,713,458 / 9,216,498 / 1,729,028 | train `theta (323640,6)`, `x (323640,10)`; val `(144000,·)`; fid `S (36000,10) f4`, `perm`, `patch`, `truth (6,) f8` | `fig:embed`, contour dump |
| `analytical_nde_match/jointl1_nobnt_s42/…`, `…_s43/…` | same per dir | same | contour dump (3-compressor pool) |
| `analytical_nde_match/operator_realnvp/none/cache/{l1_train,l1_val}.npz` | 20,713,458 / 32,256,498 | as above | contour dump, `l1_auto` arm |
| `analytical_nde_match/operator_realnvp/none/fid_summ.npz` | 432,738 | `S (9000,10) f4`, `perm`, `patch` — **the code opens `fid_summ.npz`**, not `fiducial_summaries.npz` | contour dump |
| `analytical_nde_match/l1product_bnt_vmim_s41/{cache/…, fiducial_summaries.npz}` | 20,713,458 / 32,256,498 / 432,738 | as no-BNT | contour dump (BNT) |
| `analytical_nde_match/bnt_auto/{cache/…, fiducial_summaries.npz}` | 20,713,458 / 32,256,498 / 432,738 | as above; `cache/` is already the VMIM 10-D cache | contour dump (BNT `l1_auto`) |
| `analytical_nde_match/jointl1_bnt{,_s42,_s43}/{cache/…, fiducial_summaries.npz}` | 20,713,458 / 9,216,498 / 1,729,028 per dir | as no-BNT joint | contour dump (BNT) |
| `cnn_phase/arch_sweep_2026_06_13/cnn_resnet18_s41/cache/cnn_train.npz` | 20,713,458 | `theta (323640,6) f4`, `x (323640,10) f4` | `fig:embed`, contour dump |
| `cnn_phase/arch_sweep_2026_06_13/cnn_resnet18_s41/cache/cnn_val.npz` | 32,256,498 | `theta (504000,6) f4`, `x (504000,10) f4` | contour dump |
| `cnn_phase/arch_sweep_2026_06_13/fidsumm_resnet18.npz` | 433,028 | `S (9000,10) f4`, `perm`, `patch`, `truth (6,) f8` — **note the file name** | contour dump |
| `cnn_phase/bnt_resnet18_2026_06_14/cnn_resnet18_bnt_s41/cache/{cnn_train,cnn_val}.npz` | 20,713,458 / 32,256,498 | as above | contour dump (BNT) |
| `cnn_phase/bnt_resnet18_2026_06_14/fidsumm_bnt_resnet18_s41.npz` | 433,028 | as above | contour dump (BNT) |

Sub-totals: raw 800-D cache **1,043,416,242 B**; joint raw fiducial **432,289,800 B**;
contour-dump inputs (4 arms × 2 bases, joint ℓ1 = 3 compressors each) **510,370,648 B**. The
three `fig:embed` caches are inside the contour-dump set — do not count them twice.

**`fig:fisher` size relief (optional).** If 1.04 GB is unacceptable, `fig_fisher.py` is the only
consumer and it needs `x` and `theta[:, :3]` only. A float32 `(323640, 800)` slice *is* the
file; there is no smaller form that leaves the script unmodified. Either ship it, or declare
`fig:fisher` non-reproducible from the record.

**`fig:l1joint` size relief (optional, safe).** `fig_l1.py` computes `S.mean(0)`. A one-row file
`S (1,3000) f4` holding that mean gives a bit-identical figure for **~12 kB** instead of 432 MB,
with no code change — provided the mean is computed exactly as the script would:

```python
np.savez(out, S=np.load(src)["S"].mean(0)[None].astype(np.float32),
         perm=…, patch=…, truth=…, stat=…, basis=…, k=…)
```

Note the substitution in `docs/FIGURE_MAP.md` if you take it.

### 8.1 Optional extra

`analysis/flatsky_cross_2026_06/gate_c/lc2st/fiducial_summaries_none.npz` · **28,872,738 B** ·
`S (9000,800) f4`, `perm (9000,) i4`, `patch (9000,) i4`. Used only by `gen_auto_contour.sh`
**step 1**, which is skipped when `operator_realnvp/none/cache/l1_train.npz` already exists (it
does, per §8). Include it only if you want the auto-arm VMIM compression to be re-runnable from
the raw 800-D datavector.

---

## 9. Products not yet in the record

`derived/` (Tier D) is empty, and so is the gate-dump subtree of `analysis/`. This section is
the specification of what has to be produced and by which generator; every generator named here
ships in this package.

> ### ⚠️ Regenerated products are re-derivations, not reproductions
> The inputs that survive are from a later re-run of the campaign, not from the runs that
> produced the published PDFs. Expect shifts of the same order as the retrained column of
> Table 1 (0.3 %–6 % on FoM₃, larger on the BNT arms; see `docs/NOTE_FOM_ERROR_BARS.md` §3).
> Record the shift for each regenerated product in `docs/FIGURE_MAP.md` and in the record's
> description. Do not overwrite a published figure with a re-derived one without saying so.

### 9.1 Contour dumps (`fig:corner`, `fig:bnt`)

| | |
|---|---|
| products | `derived/contour_nobnt_samples.npz`, `derived/contour_bnt_samples.npz` (~1 MB each) |
| generator | `python -m nn_l1_sbi.figures.dump_contour_samples --basis nobnt` and `--basis bnt`. All four arms — `l1_auto`, `l1_product`, `jointl1`, `cnn` — are native entries in `dump_contour_samples.ARMS` **for both bases**, so the two commands above produce complete dumps on their own. The helper `src/nn_l1_sbi/figures/gen_auto_contour.sh` is a **no-BNT-only** convenience wrapper: it hardcodes `--basis nobnt`, first VMIM-compresses the raw 800-D auto datavector if `operator_realnvp/none/cache/l1_train.npz` is absent, then runs `--basis nobnt --only l1_auto --merge`. It has no BNT counterpart and needs none — for the BNT auto arm run `--basis bnt --only l1_auto --merge` directly |
| inputs | the 10-D caches of §8 — **510,370,648 B**; they must be in the `analysis/` layout first, because the script resolves them through `_data.ROOT` |
| compute | **GPU, hours.** Per arm: RealNVP 4×128, 50,000 steps, `--m-samples 20000`, `--nde-seed 41`, `--max-perm 50`; 6 fits per basis (joint ℓ1 = 3 compressors), × 2 bases, + 2 auto-arm fits. One arm per job; each arm writes its own key and `--merge` accumulates, so the work is checkpointed per arm |
| output location | `output_root()/derived/`, **not** the data root — see §10 |
| verify | the script prints `[<basis>/<arm>] FoM3=… N=…` per arm. Expect, within a few percent: no-BNT 2448 / 3045 / 3371 / 3326 (auto / +product / joint ℓ1 / CNN); BNT 425 / 779 / 2424 / 3186. The BNT arms here use **single** compressors except joint ℓ1, so the ℓ1 targets are the single-compressor values (425, 779), not the calibrated ensemble values quoted in Table 1 (388, 718) |

### 9.2 Gate dumps, then the TARP bands (`fig:tarp`, `fig:tarp_bnt`, `fig:sbc`)

| | |
|---|---|
| products | `analysis/**/tarp_drp/dumps/<name>_{LOW,MID,HIGH}/seed_{41,42,43}/n*_m*/posterior_samples.npz` — every path marked *absent* in §2; then `derived/tarp_proper_bands{,_bnt}.npz` |
| generator | `python -m nn_l1_sbi.inference.run_joint_gate_c` (which drives `nn_l1_sbi.inference.tarp_stratified_val` → `nn_l1_sbi.inference.run_tarp_coverage`), one job per arm per basis: 8 no-BNT + 8 BNT compressor directories |
| dump schema | each `posterior_samples.npz` must carry `samples (N, M, ≥3)` and `theta (N, ≥3)` |
| **naming contract** | `_data._key()` parses the tercile and NDE seed out of the path with `dumps/(.+?)_(LOW\|MID\|HIGH)/seed_(\d+)/`, and the single-arm globs additionally pin the dump-directory *prefix*: `l1none_rnvp_*`, `l1product_rnvp_*`, `l1product_bnt_rnvp_*`, `bnt_auto_*`, `resnet18_rnvp_*`, `bnt_resnet18_rnvp_*`. The directory naming is therefore part of the contract, not cosmetic |
| compute | **GPU**, 3 NDE seeds × 600 validation points per arm; budget as for a full Gate-C campaign (hours per arm) |
| then | `python -m nn_l1_sbi.figures.fig_tarp --recompute` (CPU, ~5–6 min: 4 arms × 200 bootstraps × 2 bases) writes both band caches. `fig_sbc.py` reads the dumps directly and needs no cache |
| size decision | the dumps are GB-scale. Either include them in the analysis archive, so that `--recompute` works for a reader, or ship only `derived/tarp_proper_bands*.npz` and mark `fig:sbc` as requiring the analysis archive. **`fig:sbc` has no cached form** — without the dumps it cannot be drawn at all |

### 9.3 `derived/l1_hist_vs_s8_levels_cache.npz` (`fig:l1` panel a)

No generator survives (`fig_l1.GEN_CACHE`). Rebuild the (`lvl_X`, `lvl_s8`) level cache from the
raw ℓ1 datavector caches for the `none`, `conv` and `product` operators by binning `x` on σ₈ and
averaging within each bin. **CPU, minutes**, but the caches are ~1 GB each, so run it somewhere
with memory, not on a login node. It must satisfy the schema in §7 — in particular the flat
channel layout `channel*200 + scale*40 + snr`, since `fig_l1.py` slices channels 3/9/15 by
arithmetic.

### 9.4 `derived/embed_tsne.npz` (`fig:embed`)

`python -m nn_l1_sbi.figures.fig_embed --recompute` — **CPU, minutes** (3 × t-SNE on 6000×10,
`perplexity=40`, `init="pca"`, `random_state=0`). Requires the three 10-D caches of §8 in place.
Shipping the ~0.4 MB cache lets a reader skip 62 MB of caches entirely, so ship both.

### 9.5 `fig:ablation` — `operator_realnvp/{conv,both}/n9000/median_summary.json`

`fig_ablation.py` reads `fom3` from these two files, and neither survives. The nearest survivors
are the later-run sweeps `conv_nobnt_s41` (`fom3 = 2720.15`) and `joint_l1_nobnt_s41`
(`fom3 = 3564.87`) — but those come from a *different pipeline run* than `operator_realnvp`
(whose product arm reproduced 3045 as 3058, whereas `product_nobnt_s41` gives 3231.75).
`fig_ablation.py` also has a hard-coded `ylim = (2300, 3550)`, which 3564.87 exceeds.
Two honest options: (a) rerun the operator ladder
(`nn_l1_sbi.inference.population_sweep` on the conv/both operator arms, GPU ~2 h per arm), or
(b) ship the two later values as a **documented** substitution. Do not substitute silently.

### 9.6 `fig:bntwhiten` — six campaign `median_summary.json`

`population_sweep/flat_{none,product}`, `whiten_campaign/population_sweep/l1_{none,product}` and
`bnt_campaign/population_sweep/l1_{none,product}` (required), plus
`bntdeep{,2}_campaign/population_sweep/l1_none` (optional — `fig_bntwhiten.opt()` tolerates their
absence). None survive. Generator: `nn_l1_sbi.inference.population_sweep` per campaign arm,
**GPU ~2 h each** — about 12 GPU-hours for ~700 B of JSON. `fig:bntwhiten` is not cited by the
current manuscript, so this is the lowest-value item in this section.

### 9.7 `fig:w0` — `derived/geometry/<arm>/per_patch_grid.csv`

No generator survives; the files need a per-patch geometry run (phase C, 10°, columns including
`pull_w_0`) for `l1_auto_cross` and `cnn_auto_cross`. **GPU.** Same disposition as §9.6.

---

## 10. What writes into the data root

The data root is meant to be read-only, and almost everything treats it that way. Three
exceptions must be known before a reader mounts the record read-only:

| what | writes where | note |
|---|---|---|
| `fig_tarp.py` recompute path | `<root>/derived/tarp_proper_bands{,_bnt}.npz` | the band cache; `derived/` is created if missing |
| `fig_embed.py` recompute path | `<root>/derived/embed_tsne.npz` | the t-SNE cache; idem |
| `gen_auto_contour.sh` step 1 | `<root>/analysis/flatsky_cross_2026_06/analytical_nde_match/operator_realnvp/none/{cache/, fid_summ.npz}` | skipped entirely when `cache/l1_train.npz` already exists, which it does in a complete record |

`dump_contour_samples.py` is deliberately *not* in that list: it writes
`output_root()/derived/contour_{nobnt,bnt}_samples.npz`, i.e. under `$NN_L1_SBI_OUT` (default
`./nn_l1_sbi_out`), never into the record. **A manual step is therefore required** before the
figure scripts see a fresh dump — `fig_corner.py` and `fig_bnt.py` read
`<root>/derived/contour_*_samples.npz`:

```bash
cp "$NN_L1_SBI_OUT/derived/contour_nobnt_samples.npz" "$NN_L1_SBI_DATA/derived/"
# ...or point NN_L1_SBI_DATA at a root that already contains the new dump
```

`--merge` handles the asymmetry sensibly: it merges into the dump already under
`output_root()` if there is one, and otherwise into the copy shipped in the record, so merging a
single arm on top of the record keeps `truth` and the other arms instead of silently writing a
one-arm file. Everything else in the package reads the data root and writes only under
`output_root()`.

---

## 11. Traps, collected

1. **`$NN_L1_SBI_DATA` means two different things** — the read-only record root for
   `figures/_data.py`, and a writable compute root for `scripts/slurm/env.sh`. §3.
2. **The root is snapshotted at import time.** Set the variable before importing
   `nn_l1_sbi.figures._data`, or reload the module. §3.
3. **`joint_l1_*` ≠ `jointl1_*`.** The `ℓ1 + conv + product` arm lives in `joint_l1_*` /
   `joint_bnt_*` (with underscore); the separate joint-ℓ1 arm is `jointl1_*` (no underscore).
   Never derive a directory name from an arm label — go through `arms.json`
   (`data/table1/README.md` §2).
4. **`analysis/` directory and file names are frozen** — `l1product_rnvp_s41_n9000`,
   `cnn_resnet18_s41/readout_full`, `operator_realnvp/none`, `fid_summ.npz`,
   `fidsumm_resnet18.npz`. `_data.py` concatenates them as strings; a rename is a silent break.
5. **`fig:cp` medians in this record shift** by +6.1 % / +0.3 % / +3.1 % against the published
   values, because the per-patch files are from a later re-run. §5.2.
6. **`per_obs_samples.npy` is not a substitute for the contour dumps.** The paper's corner is
   the posterior at the *noiseless mean observation* with 20,000 draws; `per_obs_samples.npy`
   holds 2,000 draws at one noisy mock. Different object; swapping them changes the figure.
7. **`fig:sbc` has no cached form** — it needs the gate dumps themselves. §9.2.
8. `_data.py`'s module docstring lists `maps/`, `derived/` and `analysis/` but not `sweeps/` or
   `tables/`; those two are the notebook and provenance tiers that no figure script reads.
   Additive rather than contradictory, but worth a one-line docstring update when someone next
   touches that file.
