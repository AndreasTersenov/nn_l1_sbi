# `data/table1/` — Tier-A bundle: everything needed to print Table 1

61 files, **49 kB** (49,241 B). This directory is committed to the repository. Printing
Table 1 from it requires **no download, no network, no GPU, and no Zenodo access** — only
`json` and `numpy`, and it runs in well under a second. It is the input to section A of
`notebooks/reproduce_paper.ipynb`.

The heavy per-observation products (`per_patch_metrics.npz`, 153 MB;
`per_obs_samples.npy`, 216 MB) are deliberately **not** here — they live on Zenodo and are
only needed to *recompute* the block bootstrap or to draw `fig:cp`. Every number in Table 1
is already a faithful digest inside the tiny JSONs below: `fom3` in `median_summary.json`
equals `np.median(per_patch_metrics['fom3'])` to full float precision, and likewise for the
three `sigma_*` — verified for all 54 sweep dirs, exact form in §3.1.

---

## 1. Layout

```
data/table1/
  arms.json                          3.9 kB   12 arms -> relative paths + published centrals
  rows_table1_v2.json                6.3 kB   the row spec that was actually run (provenance)
  sweeps/<54 dirs>/median_summary.json
                                    15.3 kB   the actual measurements (282-287 B each)
  TABLE1_ERRORBARS_V2.json           8.4 kB   precomputed cross-check, all 12 arms
  TABLE1_CONV_ARMS.json              5.4 kB   sigma triplets + per-seed sigma, 4 conv arms
  TABLE1_PAPER.json                  1.9 kB   paper-facing digest, 8 main rows
  TABLE1_MARGINALS_NEW.json          1.0 kB   sigma digest, 2 new no-BNT conv rows
  final_bars_all_v2.json             7.1 kB   block-bootstrap + jackknife detail, 8 rows
```

Two files differ from their sources: `arms.json` (paths rewritten — §3.1 gives the exact
command that regenerates it) and `rows_table1_v2.json` (the machine-specific directory
prefix replaced by `<data_root>/` — §3.1). The five `TABLE1_*` / `final_bars_*` files and
all 54 `median_summary.json` are **byte-for-byte copies** of the originals.

The code that produced these files ships too, as `src/nn_l1_sbi/table1/` — see §8.

---

## 2. The 12 arms

Table 1 = 6 summaries x 2 frames. The canonical arm label is the JSON key everywhere:

| # | arm key | # | arm key |
|---|---|---|---|
| 1 | `l1 auto, no-BNT`        | 7  | `l1 +conv+product, no-BNT` |
| 2 | `l1 auto, BNT`          | 8  | `l1 +conv+product, BNT` |
| 3 | `l1 +conv, no-BNT`      | 9  | `joint l1, no-BNT` |
| 4 | `l1 +conv, BNT`         | 10 | `joint l1, BNT` |
| 5 | `l1 +product, no-BNT`   | 11 | `CNN, no-BNT` |
| 6 | `l1 +product, BNT`      | 12 | `CNN, BNT` |

> ### :warning: Naming trap — never derive a directory name from an arm label
>
> `l1 +conv+product` maps to sweep dirs named **`joint_l1_*` / `joint_bnt_*`** (with
> underscore), while the *separate* `joint l1` arm maps to **`jointl1_*`** (no underscore).
> `joint_l1_nobnt_s41` and `jointl1_nobnt_s41` are two different runs with different numbers.
> Always go through `arms.json`; never string-munge the label.

---

## 3. `arms.json` — arm -> path map + published centrals

Derived from `rows_table1_v2.json` (also shipped, §3.1) with every absolute path rewritten
to be **relative to this directory** and repointed at `median_summary.json` instead of
`per_patch_metrics.npz`. The `published` and `quoted` fields are unchanged and are
load-bearing.

```json
"l1 auto, no-BNT": {
  "singles": {"s41": "sweeps/auto_nobnt_s41/median_summary.json",
              "s42": "sweeps/auto_nobnt_s42/median_summary.json",
              "s43": "sweeps/auto_nobnt_s43/median_summary.json"},
  "ensemble": "sweeps/auto_nobnt_ensemble/median_summary.json",
  "published": 2448,        // int | null  — the value printed in the paper
  "quoted": "single"        // "single" | "ensemble" — WHICH estimator the paper quotes
}
```

* `singles` always has exactly the keys `s41`, `s42`, `s43`, **in that order**. `s41` is
  index 0 and is the one used as the central when `quoted == "single"`.
* `quoted == "single"` for `l1 auto/no-BNT`, `l1 +conv/no-BNT`, `l1 +product/no-BNT`,
  `l1 +conv+product/no-BNT`, `CNN/no-BNT`, `CNN/BNT`.
  `quoted == "ensemble"` for all four BNT l1 rows plus **both** `joint l1` rows.
* `published` is `null` for `l1 +conv, BNT` and `l1 +conv+product, BNT` — these are new
  results with no previously published central, so the retrained value is quoted instead.

### 3.1 `rows_table1_v2.json` — the upstream, and how to regenerate `arms.json`

`rows_table1_v2.json` is the **row spec that was actually handed to the generator**: same 12
keys, same `published` / `quoted`, and one entry per seed pointing at the 153 MB
`per_patch_metrics.npz` that row was measured from. It is the provenance record — the
evidence of which run produced which row — so the sweep directory names and file names are
preserved exactly.

The only edit is the leading directory: the absolute prefix of the machine the sweeps ran on
has been replaced by the placeholder `<data_root>/`, which is also where those files live in
the Zenodo bundle (see `data/ZENODO_MANIFEST.md`). Nothing is lost — the prefix identified a
filesystem, not a measurement — and the paths now describe the published bundle rather than a
cluster nobody else can reach. Nothing reads this file at analysis time; `arms.json` is what
the notebook loads.

`arms.json` is its path-relativised derivative, and the derivation is a script, not a hand
edit. To regenerate and byte-compare against the committed copy:

```bash
python -m nn_l1_sbi.table1.derive_arms
# wrote <output_root>/table1/arms.json  (3869 B, 12 arms)
# byte-identical to <repo>/data/table1/arms.json
```

It exits 0 on a byte-identical result and 1 with a per-arm diff otherwise. Defaults: `--rows`
= this directory's `rows_table1_v2.json`, `--compare` = this directory's `arms.json`, `--out`
= `<output_root>/table1/arms.json` (never into the repo; pass `--out` to override).

Two substitutions, nothing else:

| | `rows_table1_v2.json` | `arms.json` |
|---|---|---|
| path form | absolute, machine-specific | relative to `data/table1/` |
| target file | `sweeps/<dir>/per_patch_metrics.npz` (153 MB total, Zenodo Tier B) | `sweeps/<dir>/median_summary.json` (15.3 kB total, committed) |
| `published`, `quoted`, key order | — | unchanged |

> **Why swapping the target file is safe.** `median_summary.json` is an *exact digest* of the
> `.npz`, not a re-derivation — both are written in the same pass by
> `src/nn_l1_sbi/inference/population_sweep.py`. With `param_order = (Omega_m, sigma_8, w_0)`
> and the finite-FoM₃ mask `g = np.isfinite(npz["fom3"])`:
>
> ```python
> median_summary["fom3"]     == float(np.median(npz["fom3"][g]))
> median_summary["sigma_Om"] == float(np.median(npz["sigma"][g], axis=0)[0])
> median_summary["sigma_s8"] == float(np.median(npz["sigma"][g], axis=0)[1])
> median_summary["sigma_w0"] == float(np.median(npz["sigma"][g], axis=0)[2])
> median_summary["n"]        == int(g.sum())
> ```
>
> Verified for **all 54** sweep dirs: exact float equality (`==`, not `allclose`) on `fom3`
> and all three `sigma_*`, and `n` exact. `g` is all-True in every one of the 54 (9000 finite
> of 9000), so the mask is a no-op here and plain `np.median(npz["sigma"], axis=0)` gives the
> identical result — but the mask is what the generator applies, so keep it if you re-verify
> on new sweeps.
>
> This is why Table 1 needs no download: the 153 MB array is required only to *re*-run the
> block bootstrap (`median_se`), never to read off a central or a marginal.

---

## 4. `sweeps/<dir>/median_summary.json` — the measurements

Flat dict, no nesting. 282 B for singles, 287 B for ensembles.

```json
{
  "arm": "auto_nobnt_s41",           // str, == basename of the sweep dir
  "n": 9000,                          // int, number of mock observations (180 patches x 50 noise reps)
  "n_members": 1,                     // int, 1 = single compressor, 3 = deep ensemble
  "nde_seeds": [41, 42, 43],          // list[int], pooled RealNVP seeds (always 41,42,43)
  "fom3": 2776.934432287815,          // float, MEDIAN FoM3 over the 9000 mocks
  "sigma_Om": 0.05033291746411782,    // float, median of per-obs sigma(Omega_m)
  "sigma_s8": 0.07992637881069996,    // float, median of per-obs sigma(sigma_8)
  "sigma_w0": 0.23398674217632248,    // float, median of per-obs sigma(w_0)
  "m_samples": 2000,                  // int, posterior draws per mock
  "elapsed_s": 240.7                  // float, wall time of the sweep
}
```

Note the two independent meanings of "seed": `nde_seeds` are the three RealNVP seeds
**pooled inside every run** (identical for all 54 dirs), whereas the `s41/s42/s43` in the
*directory name* are the three independently trained **compressors** — those are what the
error bar is taken over.

### 4.1 Which 54 directories

12 arms x (3 compressor singles `s41,s42,s43` + 1 `*_ensemble`) = **48**, plus the **6**
extra-seed dirs backing the n=6 caveat: `auto_bnt_s44/45/46` and `jointl1_bnt_s44/45/46`.
The extra six are not referenced by `arms.json`; they are consumed only by the `n6_*`
fields of `TABLE1_PAPER.json` (see §5.3).

---

## 5. The cross-check JSONs

These are precomputed. The notebook does **not** need them to build Table 1 — it needs them
to *prove* the recipe in §6 is right. Verified: the recipe reproduces all of them. Their
generators are `src/nn_l1_sbi/table1/` (§8), which need the Tier-B `.npz` files; the recipe
in §6 needs only this directory.

### 5.1 `TABLE1_ERRORBARS_V2.json` — canonical, 12 arms, FoM3 only

The only file with all 12 arms in one place. Dict keyed by arm label. Written by
`python -m nn_l1_sbi.table1.make_table --rows data/table1/rows_table1_v2.json` — the field
list below is exactly that script's `row_stats()` output, in order.

```json
"l1 +conv, BNT": {
  "published": null,             // int|null   (mirrors arms.json)
  "quoted": "ensemble",          // "single"|"ensemble"
  "per_seed": {"s41": 496.39291345037316,
               "s42": 495.38610157522567,
               "s43": 524.7288270333654},   // pre-ensemble compressor singles
  "mean": 505.5026140196548,     // mean over per_seed
  "std": 16.65799710872881,      // np.std(per_seed, ddof=1)
  "lo": 495.38610157522567,      // min(per_seed)
  "hi": 524.7288270333654,       // max(per_seed)
  "pct": 3.295333524839324,      // 100*std/mean   <-- THE relative error bar
  "ensemble": 457.5656092414556, // fom3 of the *_ensemble dir
  "bias_pct": -7.821889305194384,// see the warning below; NOT an error bar
  "central": 457.5656092414556,  // = per_seed.s41 if quoted=="single" else ensemble
  "median_se": 1.2651348484480738,       // block bootstrap over the 180 patches
  "median_ci68_half": 1.319996528484893, // ditto, 68% percentile half-width
  "population": {"n": 9000, "CV_pop": 0.22699997332205601,
                 "rho": 0.00347577297857245,   // intra-patch correlation (ANOVA)
                 "n_blocks": 180, "reps_per_block": 50.0}
}
```

> :warning: **`bias_pct` in this file uses `per_seed.s41` as its baseline**, not `mean`:
> stored `bias_pct == 100*(ensemble - per_seed.s41)/per_seed.s41`. Verified for all 12 arms
> to 1e-9. If you instead want the single->ensemble shift relative to the *mean* of the three
> singles, compute `100*(ensemble - mean)/mean` yourself — it differs by several percent
> (e.g. `l1 +conv, no-BNT`: -9.53 stored vs -5.43 mean-relative). Either way it is a
> **bias**, reported separately, and is never summed into the `+-`.

### 5.2 `TABLE1_CONV_ARMS.json` — 4 conv arms, the sigma-carrying one

The only JSON that carries full sigma triplets *with* per-seed spread.

```json
"l1 +conv, no-BNT": {
  "published_fom": 2671,                 // int|null
  "source": "published",                 // "published" | "new (retrained ensemble)"
  "quoted": "single",                    // "single"|"ensemble"
  "fom_central": 2671,                   // published if non-null, else retrained
  "fom_pm": 133.3458623112472,           // == fom_central * fom_rel_pct/100
  "fom_rel_pct": 4.992357256130558,      // == 100*std(per_seed,ddof=1)/mean(per_seed)
  "fom_per_seed": [2720.15, 2623.54, 2462.97],   // list, order s41,s42,s43
  "fom_ensemble": 2461.0241378554833,
  "bias_pct": -9.526260073217934,        // s41-baseline, as in 5.1
  "sigma": {
    "sigma_Om": {"central": 0.0517794..., "pm": 0.0010693...,
                 "per_seed": [ ...3 floats... ],
                 "ensemble": 0.0531662..., "rel_pct": 2.0652...},
    "sigma_s8": { ...same 5 keys... },
    "sigma_w0": { ...same 5 keys... }
  }
}
```

### 5.3 `TABLE1_PAPER.json` — 8 main rows, paper-facing digest

```json
"l1 auto, no-BNT": {
  "published": 2448,        // int
  "pm": 27.0,               // == published * rel        <-- the number after the "+-"
  "rel_pct": 1.1,           // == 100*std(per_seed,ddof=1)/mean(per_seed)
  "n_seeds": 3,
  "median_se": 4.95,        // caption-only subdominant term
  "retrained": 2776.9,      // the retrained central (s41 or ensemble per `quoted`)
  "pm_retrained": 30.6,     // == retrained * rel  (same relative bar, other central)
  "n6_seeds": 3,            // 6 only for the two arms with s44/45/46 dirs
  "n6_rel_pct": 1.1,        // rel recomputed over 6 compressor seeds
  "n6_pm": 27.0             // == published * n6_rel
}
```

For `l1 auto, BNT` and `joint l1, BNT` the `n6_*` fields are computed over the six dirs
`<prefix>_s41..s46` and differ from the 3-seed values (verified: `l1 auto, BNT`
`n6_rel_pct` 8.79 % vs 11.07 % at n=3). This is the *only* consumer of the 6 extra dirs.

### 5.4 `TABLE1_MARGINALS_NEW.json` — 2 rows

sigma digest for the two new no-BNT conv rows (`l1 +conv, no-BNT`,
`l1 +conv+product, no-BNT`): `published_fom`, `fom_pm`, and for each of
`sigma_Om`/`sigma_s8`/`sigma_w0` a `{value, pm, rel_pct}` triple. Redundant with §5.2 but
kept because it is what the paper text quotes.

### 5.5 `final_bars_all_v2.json` — **a LIST of 8 dicts**, not a dict

Written by `python -m nn_l1_sbi.table1.final_bars --row-json <spec>`; one dict per call to
that module's `do_row()`, appended in spec order. The 8 `row` labels are its own free-text
row names (`"auto no-BNT (ensemble)"`, `"auto no-BNT (singles)"`, …), *not* the canonical
12 arm keys of §2 — do not join this file to `arms.json` on the label.

```json
{"row": "auto no-BNT (ensemble)",                 // str label, NOT the canonical arm key
 "per_seed_median": {"ens": 2511.147...},
 "median_term": {"se": 4.142..., "blocks": "180 patches", "p16": ..., "p84": ...,
                 "ci68_half_width": 4.0229..., "reps_per_patch": 50},
 "population_stats": {"n": 9000, "CV_pop": 0.14299..., "rho": 0.0,
                      "n_blocks": 180, "reps_per_block": 50.0},
 "ensemble_term": {"K": 3, "theta_leave_one_out": [3 floats],
                   "theta_bar": ..., "se": 13.376...},
 "validation": {"fom_ok": false}}
```

> :warning: `ensemble_term` is a **leave-one-out jackknife diagnostic and is never
> quotable** (`docs/NOTE_FOM_ERROR_BARS.md`). Do not surface it in the notebook or the
> paper. The quotable bar is the §6 recipe; `median_term.se` is a caption-only
> subdominant term.

---

## 6. The Table-1 recipe

Verified end-to-end against every file in this directory (see §7).

```python
import json, os
import numpy as np

ROOT = "data/table1"
load = lambda rel: json.load(open(os.path.join(ROOT, rel)))
arms = load("arms.json")

for label, a in arms.items():
    singles = [load(p) for p in a["singles"].values()]        # order: s41, s42, s43
    ens     =  load(a["ensemble"])
    src     = singles[0] if a["quoted"] == "single" else ens  # s41 is index 0
    fom_rt  = src["fom3"]                                     # retrained central
    per_seed = [s["fom3"] for s in singles]
    rel     = np.std(per_seed, ddof=1) / np.mean(per_seed)    # THE error bar, relative
    central = a["published"] if a["published"] is not None else fom_rt
    pm      = central * rel
    sig     = (src["sigma_Om"], src["sigma_s8"], src["sigma_w0"])
    sig_rel = [np.std([s[k] for s in singles], ddof=1) / np.mean([s[k] for s in singles])
               for k in ("sigma_Om", "sigma_s8", "sigma_w0")]
    sig_pm  = [v * r for v, r in zip(sig, sig_rel)]
    print(f"{label:26s} {central:7.0f} +- {pm:5.0f}")
```

### Rules (`docs/NOTE_FOM_ERROR_BARS.md` §5-6)

1. `+-` is **always** the std over the **three pre-ensemble compressor singles**, for every
   row — including rows whose central is an ensemble. It is carried as a *relative* quantity
   `std/mean` and multiplied by whichever central the row quotes, so the bar rescales
   correctly onto the published central.
2. The single -> ensemble shift is a **bias**, reported separately, never summed into `+-`.
3. `median_se` (block bootstrap over the 180 patches, 50 reps kept, 1e4 replicates, 68 %
   percentile) is a **caption-only** subdominant term. Precomputed in
   `TABLE1_ERRORBARS_V2.json[arm]["median_se"]`. Recomputing it requires
   `per_patch_metrics.npz` (Zenodo, Tier B) — it is not recomputable from this directory.
4. The leave-one-out jackknife in `final_bars_all_v2.json` is **not** the bar.
5. sigma centrals track the row's quoted estimator (s41 when `quoted=="single"`, the pooled
   ensemble otherwise). There are no published sigma triplets to defer to for any row.

### Three quantities that are easy to confuse

| quantity | formula | where |
|---|---|---|
| quotable bar | `central * rel` | Table 1, `TABLE1_PAPER.pm`, `TABLE1_CONV_ARMS.fom_pm` |
| same bar on the retrained central | `retrained * rel` | `TABLE1_PAPER.pm_retrained` |
| raw spread | `np.std(per_seed, ddof=1)` | diagnostic only — unrescaled, do not quote |

---

## 7. Verified output

Running §6 against this directory reproduces, exactly:

* all 12 rows of the retrained table (FoM3, spread, and the three marginals);
* `TABLE1_ERRORBARS_V2.json` for all 12 arms on `per_seed`, `mean`, `std`, `lo`, `hi`,
  `pct`, `ensemble`, `central`, `bias_pct`, `published`, `quoted` (atol 1e-9);
* `TABLE1_CONV_ARMS.json` (4 arms, incl. all sigma centrals and pm);
* `TABLE1_PAPER.json` (8 rows, at stored precision), including the `n6_*` fields;
* `TABLE1_MARGINALS_NEW.json` (2 rows).

Retrained centrals:

| arm | quoted | FoM3 (retrained) | std over 3 | sigma(Om) | sigma(s8) | sigma(w0) |
|---|---|--:|--:|--:|--:|--:|
| l1 auto, no-BNT | single | 2776.9 | 30.5 | 0.0503 | 0.0799 | 0.2340 |
| l1 auto, BNT | ensemble | 390.7 | 49.3 | 0.0861 | 0.1725 | 0.3191 |
| l1 +conv, no-BNT | single | 2720.2 | 129.9 | 0.0518 | 0.0814 | 0.2366 |
| l1 +conv, BNT | ensemble | 457.6 | 16.7 | 0.0836 | 0.1643 | 0.3155 |
| l1 +product, no-BNT | single | 3231.8 | 199.3 | 0.0462 | 0.0746 | 0.2271 |
| l1 +product, BNT | ensemble | 758.3 | 37.7 | 0.0721 | 0.1293 | 0.2994 |
| l1 +conv+product, no-BNT | single | 3564.9 | 207.4 | 0.0452 | 0.0714 | 0.2202 |
| l1 +conv+product, BNT | ensemble | 703.9 | 84.9 | 0.0755 | 0.1349 | 0.3030 |
| joint l1, no-BNT | ensemble | 3379.5 | 108.0 | 0.0440 | 0.0716 | 0.2234 |
| joint l1, BNT | ensemble | 2405.2 | 259.6 | 0.0513 | 0.0856 | 0.2366 |
| CNN, no-BNT | single | 3427.6 | 14.1 | 0.0444 | 0.0702 | 0.2288 |
| CNN, BNT | single | 3147.1 | 18.5 | 0.0461 | 0.0730 | 0.2311 |

Rescaling the relative bar onto the published centrals gives the paper values:

```
l1 auto, no-BNT            2448 +- 27      l1 +conv+product, no-BNT   3255 +- 200
l1 auto, BNT                388 +- 43      l1 +conv+product, BNT       704 +- 76   (new)
l1 +conv, no-BNT           2671 +- 133     joint l1, no-BNT           3371 +- 96
l1 +conv, BNT               458 +- 15      joint l1, BNT              2424 +- 208  (new)
l1 +product, no-BNT        3045 +- 183     CNN, no-BNT                3326 +- 14
l1 +product, BNT            718 +- 29      CNN, BNT                   3186 +- 19
```

(The two rows marked *new* have `published == null`; their central is the retrained
ensemble.)

---

## 8. Provenance

Every file here has a named generator **that ships in this repository**, so the chain
`.npz` → digest → error bar → printed row can be re-run rather than taken on trust.

| file(s) | written by | from |
|---|---|---|
| `sweeps/*/median_summary.json` | `nn_l1_sbi.inference.population_sweep`, invoked with `--seeds 41,42,43 --n-obs 9000 --max-perm 50 --m-samples 2000` | the trained NDE posteriors of each arm; written in the same pass as `per_patch_metrics.npz` and `per_obs_samples.npy` |
| `rows_table1_v2.json` | hand-written row spec; shipped verbatim as the provenance record of which run produced which row (§3.1) | -- |
| `arms.json` | `nn_l1_sbi.table1.derive_arms` — run `python -m nn_l1_sbi.table1.derive_arms` to regenerate and byte-compare (§3.1) | `rows_table1_v2.json` |
| `TABLE1_ERRORBARS_V2.json` | `nn_l1_sbi.table1.make_table --rows rows_table1_v2.json` | `per_patch_metrics.npz` of the 48 referenced sweep dirs (`fom3` + `patch`) |
| `final_bars_all_v2.json` | `nn_l1_sbi.table1.final_bars --row-json <spec>` | `per_patch_metrics.npz` (`fom3`, `patch`, and `arm_mean`/`arm_cov` for the jackknife) |
| `TABLE1_CONV_ARMS.json`, `TABLE1_PAPER.json`, `TABLE1_MARGINALS_NEW.json` | one-off presentation digests; **their original script was not retained**. Each is fully reproduced by the §6 recipe from `arms.json` + the 54 `median_summary.json` (plus `TABLE1_ERRORBARS_V2.json["median_se"]` for the `median_se` field), verified per §7 | -- |
| error-bar definition | `docs/NOTE_FOM_ERROR_BARS.md` §5–6 | -- |

`make_table` and `final_bars` read Tier-B `per_patch_metrics.npz`, which is not in this
repository. They take those paths from the row spec you pass (`--rows` / `--row-json`), so
point the spec at your unpacked Zenodo bundle; only `final_bars --validate` resolves its own
paths, under `NN_L1_SBI_DATA`. Both write under `NN_L1_SBI_OUT` (default `./nn_l1_sbi_out`),
never back into `data/`. Nothing in §6 needs any of this.

Re-verified with the shipped code, from `rows_table1_v2.json` and the Tier-B `.npz` files:

* `python -m nn_l1_sbi.table1.make_table --rows data/table1/rows_table1_v2.json` reproduces
  `TABLE1_ERRORBARS_V2.json` **byte-for-byte** — all 12 arms, all 14 fields, including the
  10⁴-replicate `median_se` and `median_ci68_half` (the block bootstrap is seeded, `seed=0`);
* `python -m nn_l1_sbi.table1.final_bars --row-json <the 8-row spec>` reproduces every
  `per_seed_median`, `seed_term`, `median_term`, `population_stats` and `ensemble_term` block
  of `final_bars_all_v2.json` to exact float equality (its `validation` blocks compare against
  `--expect` values that were not retained, so those are not reproducible);
* `python -m nn_l1_sbi.table1.derive_arms` reproduces `arms.json` byte-for-byte.

---

## 9. What is deliberately absent

| product | size | why not here |
|---|--:|---|
| `sweeps/*/per_patch_metrics.npz` | 153 MB | Zenodo Tier B. Needed only to recompute `median_se` (already precomputed here) and to draw `fig:cp`. |
| `sweeps/*/per_obs_samples.npy` | 216 MB | Zenodo. Per-obs posterior draws; not used by Table 1. |

Table 1 is complete without them.
