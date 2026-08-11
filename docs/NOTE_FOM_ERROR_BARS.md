# Error bars on the figure of merit

This is the methodology note referenced by Table 1 of the paper. It defines
precisely what the quoted `±` means, what it does *not* mean, and how it is
computed — so that the numbers in the table can be reproduced and interpreted
without guesswork.

**Every number in this note is recomputed from the JSON bundle shipped in
`data/table1/`** — no download, no network and no GPU required. The recipe in §6
reproduces the stored cross-check fields `pct`, `central` and `bias_pct` for all
12 arms to 1e-9, and reproduces the printed Table 1 exactly. `median_se` is the
one quantity that cannot be recomputed from the bundle (it needs the
per-observation arrays); it is precomputed in the shipped JSONs and cross-checked
between them. `data/table1/README.md` documents the file layout.

---

## 1 · What the quoted number is

Each Table-1 entry is a two-stage estimate.

FoM₃ is a deterministic functional of the **learned** posterior at a mock
observation *x*:

```
FoM₃(x) = 1 / sqrt( det Cov(Ω_m, σ₈, w₀) )
```

The quoted value is the robust **median** of that functional over
*n* = 9000 mock observations. The target of the estimate is the FoM the exact
posterior would deliver under the quoted pipeline recipe. The estimate deviates
from that target in two distinct ways:

- **bias** — the learned posterior is mis-calibrated (over- or under-confident);
- **variance** — re-running the whole pipeline gives a different number.

**These are reported separately, and deliberately so.** Bias is controlled by
the calibration battery (TARP + SBC, see `PIPELINE.md`) and, in the rows where
individually trained compressors failed it, corrected by using a deep ensemble
as the quoted estimator. The measured de-inflation from single to ensemble
**is** the bias term, and it is reported as a bias — never folded into the `±`.

The `±` quotes the remaining **variance**: run-to-run reproducibility of the
quoted pipeline.

## 2 · What the ± is not

Three things it is explicitly not, each of which would be a natural misreading:

| It is not | That quantity instead lives in | It answers |
|---|---|---|
| the population scatter of FoM₃ across the 9000 mock observations | the coverage-probability figure (`fig_cp`); its size is `population.CV_pop` = 17–26 % | "how much does the FoM vary from sky patch to sky patch?" |
| a posterior width | — the FoM already summarises one | "how tight is one posterior?" |
| the single→ensemble shift | `bias_pct`, reported separately (§3.2) | "how mis-calibrated was a single compressor?" |

The third confusion is the expensive one: the single→ensemble shift is **larger
than the `±` in 9 of the 12 arms**. Adding it to the bar, or quoting it as the
bar, would misrepresent the table in opposite directions.

## 3 · The error budget

| source | independent across arms? | measured size | note |
|---|---|---|---|
| **compressor training seed** | **yes** (different networks) | **0.4 – 11.1 %**, per arm — dominant | this is what the `±` quotes (§3.1) |
| median sampling error (finite mocks) | no (same mocks for all arms) | **0.18 – 0.33 %** | block bootstrap over patches, §5 |
| NDE training seed / family | yes | **not measurable from the shipped bundle** | all 54 runs pool the same three RealNVP seeds (41, 42, 43); no per-NDE-seed breakdown is shipped, so no figure is quoted here |
| training-set realisation (899 cosmologies) | no (shared across all arms) | not quantifiable without new simulations | largely cancels in comparisons between summaries |

Listed separately because it is **not** an error term:

| not a source of error | independent across arms? | measured size | note |
|---|---|---|---|
| single → ensemble shift (`bias_pct`) | yes | **+0.4 % to −21.8 %** | a **bias**, reported separately, never summed into the `±` (§3.2) |

The dominant term is the compressor seed. The `±` in Table 1 is the **standard
deviation over three independently trained, pre-ensemble VMIM compressor seeds**
(s41, s42, s43), carried as a *relative* quantity and rescaled onto the quoted
central (§6).

### 3.1 · The seed term, per arm

There is no single "typical" value: the spread is a property of the arm, and it
ranges over more than a factor of twenty across the table.

| arm | quoted estimator | seed spread `pct` | Table 1 |
|---|---|--:|---|
| CNN, no-BNT | single | 0.41 % | 3326 ± 14 |
| CNN, BNT | single | 0.59 % | 3186 ± 19 |
| ℓ1 auto, no-BNT | single | 1.10 % | 2448 ± 27 |
| joint ℓ1, no-BNT | ensemble | 2.84 % | 3371 ± 96 |
| ℓ1 +conv, BNT | ensemble | 3.30 % | 458 ± 15 |
| ℓ1 +product, BNT | ensemble | 4.09 % | 718 ± 29 |
| ℓ1 +conv, no-BNT | single | 4.99 % | 2671 ± 133 |
| ℓ1 +product, no-BNT | single | 6.01 % | 3045 ± 183 |
| ℓ1 +conv+product, no-BNT | single | 6.14 % | 3255 ± 200 |
| joint ℓ1, BNT | ensemble | 8.59 % | 2424 ± 208 |
| ℓ1 +conv+product, BNT | ensemble | 10.81 % | 704 ± 76 |
| ℓ1 auto, BNT | ensemble | 11.07 % | 388 ± 43 |

Median across arms 4.54 %, mean 4.99 %. Two arms are below 1 %, and both are CNN
rows; three are above 8 %, and all three are BNT ℓ1 rows. **Quote the per-arm
value, never a table-wide average** — a statement like "the compressor-seed
spread is ~5 %" is true of the mean and false of most individual rows.

The two ends of the table are the part of the pattern that is robust: the CNN
compressors are by a wide margin the most reproducible across training seeds,
and the widest three rows are all BNT ℓ1 arms. The BNT ℓ1 rows are *not*
uniformly wide, though — `ℓ1 +conv, BNT` (3.30 %) and `ℓ1 +product, BNT`
(4.09 %) sit below three of the no-BNT ℓ1 rows — so this is an observation about
the extremes, not a rule that predicts a row's spread from its frame.

**A three-draw standard deviation is itself uncertain.** For Gaussian draws,
`SD(s)/σ = sqrt(1 − c₄²) = 0.46` at *n* = 3 (0.31 at *n* = 6), so the `±` should
be read as *indicative* reproducibility rather than a precise interval. §6.2
gives the measured *n* = 3 → *n* = 6 check on the two rows where six compressor
seeds exist.

### 3.2 · The single → ensemble shift is a bias, not a bar

`bias_pct` in `TABLE1_ERRORBARS_V2.json` is `100·(ensemble − s41)/s41`: what
happens to the median when three compressors are pooled instead of one.

| arm | `bias_pct` | arm | `bias_pct` |
|---|--:|---|--:|
| ℓ1 auto, BNT | +0.44 % | ℓ1 +conv, no-BNT | −9.53 % |
| CNN, BNT | −1.19 % | ℓ1 auto, no-BNT | −9.57 % |
| CNN, no-BNT | −3.02 % | joint ℓ1, no-BNT | −10.17 % |
| ℓ1 +product, no-BNT | −5.50 % | ℓ1 +conv+product, no-BNT | −12.19 % |
| ℓ1 +conv+product, BNT | −7.60 % | ℓ1 +product, BNT | −16.86 % |
| ℓ1 +conv, BNT | −7.82 % | joint ℓ1, BNT | −21.79 % |

Eleven of twelve are negative — ensembling **de-inflates** the FoM, which is the
signature of single-compressor over-confidence. Median magnitude 8.67 %: roughly
twice the median seed spread (4.54 %) and about forty times the median sampling
error (0.219 %). The extreme is `joint ℓ1, BNT` at −21.8 %. The shift is larger
than the `±` of its own row in 9 of the 12 arms.

Two cautions:

* The stored baseline is `s41`, **not** the mean of the three singles. Against
  the mean the shift is −1.7 % to −20.4 % and the ordering changes (most visibly
  `ℓ1 auto, BNT`: +0.44 % against s41, −12.25 % against the mean). Whichever
  baseline is used, it is a bias.
* It is the reason six rows quote an ensemble central. Where singles failed
  calibration, the ensemble is the *estimator*, and the shift is the correction —
  not an uncertainty about it.

## 4 · The paired design

All arms are evaluated on the **same 9000 mock observations**. Population
scatter and training-set noise therefore cancel in *differences* between arms.
What a difference does inherit is the independent per-arm seed noise.

Because the per-arm spreads differ by more than a factor of twenty (§3.1), the naive
equal-precision rule `σ(difference) ≈ sqrt(2) × pct` is only a sanity check —
and even as a sanity check it spans **0.6 % to 15.7 %** across the table, not a
single number. **Use the quadrature sum of the two arms' own spreads:**

```
σ(difference) ≈ sqrt( pct_A² + pct_B² )
```

Over all 66 arm pairs this runs from **0.7 % to 15.5 %** (median 7.6 %). Worked
examples:

| comparison | pct_A | pct_B | σ(difference) |
|---|--:|--:|--:|
| CNN, no-BNT vs joint ℓ1, no-BNT | 0.41 % | 2.84 % | **2.87 %** |
| CNN, no-BNT vs ℓ1 +conv+product, no-BNT | 0.41 % | 6.14 % | **6.15 %** |
| CNN, BNT vs joint ℓ1, BNT | 0.59 % | 8.59 % | **8.61 %** |
| ℓ1 auto, no-BNT vs ℓ1 +product, no-BNT | 1.10 % | 6.01 % | **6.11 %** |

Note what the quadrature sum does in the lopsided cases: a comparison against a
CNN row is governed almost entirely by the ℓ1 row's spread. This is the correct
yardstick for comparing two summaries, and it is what makes statements like
"these two are statistically indistinguishable" meaningful rather than
rhetorical.

## 5 · The median's sampling error, and why √9000 is the wrong denominator

The 9000 mocks are **180 patches × 50 noise realisations**. They are *not* 9000
independent draws: FoM values that share a patch are correlated through that
patch's signal realisation. Model, per arm:

```
FoM_ij = m + a_i + e_ij ,     i = 1..180 patches,  j = 1..50 noise reps
Var(a) = τ²   (patch effect)
Var(e) = σ_w² (noise-rep effect)
ρ      = τ² / (τ² + σ_w²)     (intra-patch correlation)
```

ρ is measurable by a one-way ANOVA on the per-observation FoM array. The
Gaussian-asymptotic approximation for the median of clustered data (the π/2
inflation of the mean's variance) gives:

```
SE(median) ≈ 1.253 · CV_pop · sqrt( ρ/180 + (1−ρ)/9000 )
```

with `CV_pop` the population coefficient of variation of FoM₃ across mocks.
Worked table, SE as a percentage of the median:

| CV_pop → / ρ ↓ | 10 % | 20 % | 30 % |
|---|---|---|---|
| ρ = 0 (no patch effect) | 0.13 % | 0.26 % | 0.40 % |
| **ρ = 0.005 (the measured regime)** | **0.15 %** | **0.29 %** | **0.44 %** |
| ρ = 0.05 | 0.25 % | 0.49 % | 0.74 % |
| ρ = 0.3 | 0.52 % | 1.05 % | 1.57 % |
| ρ = 1 (pure patch effect) | 0.93 % | 1.87 % | 2.80 % |

**Reading**: the naive √9000 argument ("negligible") fails only when the patch
effect is strong. In this analysis it is not. The measured values, stored in
`TABLE1_ERRORBARS_V2.json[arm]["population"]`, are

* `CV_pop` = **0.169 – 0.258** across the 12 arms,
* `ρ` = **0.0002 – 0.0077** — i.e. at most 0.8 % intra-patch correlation,

which put every arm in the top rows of the table. Plugging each arm's own
(`CV_pop`, ρ) into the formula gives **0.23 – 0.36 %**, indistinguishable from
the rows computed with ρ forced to zero. The patch effect is real but tiny; the
√9000 intuition happens to survive here, and the honest statement is that it was
*checked*, not assumed.

### Block bootstrap (the authoritative computation)

Per arm, per seed:

1. Resample the 180 patches **with replacement**.
2. Keep all 50 noise realisations of each sampled patch.
3. Recompute the robust median FoM₃.
4. 10⁴ replicates; report the percentile 68 % interval.

Results, stored as `median_se` (and `median_ci68_half`) in
`TABLE1_ERRORBARS_V2.json`. These are computed on the **s41 single**, not on the
row's quoted estimator — confirmed by the exact agreement with the `(singles)`
rows of `final_bars_all_v2.json` for the four arms that file covers, where the
corresponding `(ensemble)` rows carry different values. Express them relative to
s41:

| arm | `median_se` | as % of s41 | arm | `median_se` | as % of s41 |
|---|--:|--:|---|--:|--:|
| ℓ1 auto, no-BNT | 4.95 | 0.178 % | ℓ1 +conv+product, no-BNT | 7.94 | 0.223 % |
| ℓ1 auto, BNT | 1.12 | 0.288 % | ℓ1 +conv+product, BNT | 2.21 | 0.290 % |
| ℓ1 +conv, no-BNT | 5.85 | 0.215 % | joint ℓ1, no-BNT | 7.60 | 0.202 % |
| ℓ1 +conv, BNT | 1.27 | 0.255 % | joint ℓ1, BNT | 6.52 | 0.212 % |
| ℓ1 +product, no-BNT | 6.09 | 0.188 % | CNN, no-BNT | 8.37 | 0.244 % |
| ℓ1 +product, BNT | 2.98 | 0.327 % | CNN, BNT | 6.59 | 0.209 % |

**Range 0.18 – 0.33 %** (median 0.219 %), and the 68 % percentile half-widths
agree with the standard errors to within a few per cent — the bootstrap
distribution is close to Gaussian. The analytic estimate above (0.23–0.36 %) is
~20 % conservative relative to the bootstrap, as expected from an asymptotic
approximation.

**Where it is not negligible.** Rescaled onto the same printed central, the seed
term exceeds the median's sampling error by a factor of 1.7 to 40 (median 19).
The two extremes matter:

| arm | central | `±` (seed) | SE(median) | ratio |
|---|--:|--:|--:|--:|
| CNN, no-BNT | 3326 | 13.7 | 8.1 | **1.7×** |
| CNN, BNT | 3186 | 18.7 | 6.7 | **2.8×** |
| joint ℓ1, BNT | 2424 | 208.2 | 5.1 | 40.5× |

For the two CNN rows — the most reproducible in the table — the median's
sampling error is within a factor of three of the quoted bar and should not be
described as negligible. For every other row it is.

One residual caveat, stated rather than fixed: the 180 patches tile the *same*
fiducial sky realisations, so even patch-level resampling is mildly optimistic.
This is irrelevant at the level of a reproducibility statement.

## 6 · Evaluation protocol and the exact recipe

### 6.1 · The recipe

This is the normative definition. `data/table1/README.md` §6 and
`docs/PIPELINE.md` cite this section; the code below is the same recipe and runs
against the shipped bundle in well under a second.

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
    per_seed = [s["fom3"] for s in singles]
    rel     = np.std(per_seed, ddof=1) / np.mean(per_seed)    # THE bar, relative
    central = a["published"] if a["published"] is not None else src["fom3"]
    pm      = central * rel
    sig     = (src["sigma_Om"], src["sigma_s8"], src["sigma_w0"])
    sig_rel = [np.std([s[k] for s in singles], ddof=1) / np.mean([s[k] for s in singles])
               for k in ("sigma_Om", "sigma_s8", "sigma_w0")]
    sig_pm  = [v * r for v, r in zip(sig, sig_rel)]
    print(f"{label:26s} {central:7.0f} +- {pm:5.0f}")
```

Five rules make this the right recipe rather than one of several plausible ones:

1. **The `±` is always the standard deviation over the three pre-ensemble
   compressor singles**, `np.std(per_seed, ddof=1)`, for **every** row —
   including the six rows whose central is an ensemble. It is carried as a
   *relative* quantity `std/mean` and multiplied by whichever central the row
   quotes, so the bar rescales correctly onto the published central. This is what
   makes the `±` comparable across the table and interpretable as training
   stochasticity.
2. **The central is `published` when `arms.json` gives one, and the retrained
   value otherwise.** `published` is `null` only for `ℓ1 +conv, BNT` and
   `ℓ1 +conv+product, BNT` — new results with no previously published number, for
   which the retrained ensemble is quoted. The retrained central itself is `s41`
   when `quoted == "single"` and the pooled ensemble when `quoted == "ensemble"`.
3. **The single → ensemble shift is a bias**, reported separately (§3.2), never
   summed into the `±`. (`bias_pct` uses `s41` as its baseline, not the mean of
   the three.)
4. **σ centrals track the row's quoted estimator** — `s41` when
   `quoted == "single"`, the pooled ensemble otherwise — and their `±` is the
   same relative-spread construction applied per parameter. There are no
   published σ triplets to defer to for any row, so the σ column is retrained
   throughout.
5. **`median_se` is a caption-only subdominant term** (§5), and the leave-one-out
   ensemble jackknife in `final_bars_all_v2.json` (`ensemble_term`) is a
   diagnostic that is **never** the bar. The jackknife returns 0.5 – 5.1 % on the
   four rows where it exists, values that are neither the seed spread nor the
   bias; quoting it would invent a third number with no defined meaning.

Rule 1 is the one that is easy to get wrong: do **not** mix "spread of singles"
with "shift under ensembling". The first is the error bar; the second is the
bias.

### 6.2 · Evaluation protocol

For each Table-1 row, for each compressor seed *c* ∈ {41, 42, 43}:

1. Build (or reuse) the compressed summary cache for that seed.
2. Compute the robust median FoM₃ and the marginals σ(Ω_m), σ(σ₈), σ(w₀) over
   the same *n* = 9000 mocks, holding the NDE recipe fixed exactly
   (`ConditionalRealNVP` 4×128, NDE seeds 41/42/43 pooled inside every run).
3. Also evaluate the three-compressor deep ensemble for the same arm.
4. Apply §6.1 to get the central and the `±`.
5. Quote the **pre-ensemble per-seed spread for every row**, including rows whose
   central value is ensemble-estimated.
6. Run the §5 block bootstrap once per arm (seed 41); report the median's SE in
   the caption as the subdominant term.

**The *n* = 3 → *n* = 6 caveat.** Three extra compressor seeds (s44, s45, s46)
were trained for two of the three widest arms — `ℓ1 auto, BNT` and
`joint ℓ1, BNT`. The bar moves substantially, and in both directions:

| arm | `±` at *n* = 3 | `±` at *n* = 6 | per-seed FoM₃, s41…s46 |
|---|--:|--:|---|
| ℓ1 auto, BNT | 11.07 % (388 ± 43) | 8.79 % (388 ± 34) | 389, 481, 466, 425, 401, 468 |
| joint ℓ1, BNT | 8.59 % (2424 ± 208) | 13.42 % (2424 ± 325) | 3075, 3251, 2740, 2551, 3296, 3712 |

This is the 46 % uncertainty of a three-draw standard deviation (§3.1) made
concrete. Table 1 quotes the *n* = 3 values for uniformity across the table; the
*n* = 6 values are carried in `TABLE1_PAPER.json` as `n6_rel_pct` / `n6_pm` and
are the honest bound on how well the bar itself is known.

The SLURM recipes that execute this protocol are in `scripts/slurm/` — see
`docs/PIPELINE.md` §"Reproducing Table 1".

## 7 · Presentation convention

- **Table 1** carries `±` on the FoM₃ column only. The same construction applied
  per parameter gives σ-column spreads of 0.2 – 4.7 %, smaller than the row's
  FoM₃ spread in 10 of the 12 arms; carrying them in the table would make it
  noisy, so quote them once in the caption if needed. (The two exceptions are the
  CNN rows, whose FoM₃ spread is so small — 0.41 % and 0.59 % — that the σ
  spreads, 0.3 – 1.0 %, exceed it. FoM₃ is not a monotone function of the three
  σ's, so this ordering is not guaranteed and should not be assumed.)
- The caption defines the `±` as: *"± is the spread (standard deviation) over
  three independently trained compressors, the dominant run-to-run variability;
  it ranges from 0.4 % of the central for the CNN summaries to 11 % for the
  BNT ℓ1 summaries. The sampling error of the median over the 9000 mock
  observations is 0.2–0.3 % (block bootstrap over the 180 patches) and is smaller
  than the compressor-seed spread in every row. Pooling three compressors into a
  deep ensemble shifts the medians by up to 22 %; that shift is a calibration
  bias, is quoted separately, and is not included in the ±."*
- Give the per-arm value wherever the bar is discussed in the text. There is no
  table-wide "typical" spread (§3.1).
- For retention ratios (Table 2), compute the retention **per seed** first and
  take the spread. Numerator and denominator share compressor seeds, so naive
  error propagation overstates the uncertainty.
- Do **not** put the population IQR in the FoM column. It answers a different
  question and will be misread.
- Do **not** quote the leave-one-out ensemble jackknife (§6.1, rule 5).

---

## 8 · Reproducing this note

Every quantitative claim above comes from `data/table1/`:

| claim | source field |
|---|---|
| per-arm seed spread, §3.1 | `TABLE1_ERRORBARS_V2.json[arm]["pct"]`, = `100·std(per_seed, ddof=1)/mean` recomputed from `sweeps/*/median_summary.json` |
| single→ensemble bias, §3.2 | `TABLE1_ERRORBARS_V2.json[arm]["bias_pct"]` |
| difference noise, §4 | quadrature sums of the `pct` column |
| CV_pop, ρ, §5 | `TABLE1_ERRORBARS_V2.json[arm]["population"]` |
| median SE, §5 | `TABLE1_ERRORBARS_V2.json[arm]["median_se"]`, `["median_ci68_half"]`; cross-checked against `final_bars_all_v2.json` `(singles)` rows |
| jackknife range, §6.1 rule 5 | `final_bars_all_v2.json[*]["ensemble_term"]["se"]` |
| *n* = 6 values, §6.2 | `sweeps/{auto_bnt,jointl1_bnt}_s4{4,5,6}/median_summary.json`, cross-checked against `TABLE1_PAPER.json` `n6_rel_pct` |
| printed Table 1 | §6.1 recipe applied to `arms.json` |

Section A of `notebooks/reproduce_paper.ipynb` runs the §6.1 recipe and prints
the table. Recomputing `median_se` from scratch additionally requires
`per_patch_metrics.npz` (Zenodo); the values are precomputed in the shipped
JSONs.
