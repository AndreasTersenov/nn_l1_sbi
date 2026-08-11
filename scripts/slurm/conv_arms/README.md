# ℓ1+conv and ℓ1+conv+product — the two middle rungs of the operator ladder

These jobs add two arms to the Table-1 sweep, filling in the rungs of the operator ladder
(2448 → 2671 → 3045 → 3255 → 3371) that had neither error bars nor BNT counterparts. The
recipe is identical to every other ℓ1 arm — only the input cache differs.

| step | job | notes |
|---|---|---|
| slice conv | `slice_conv.slurm` (CPU, ~5 min) | Column-slices ℓ1+conv (10 ch = 4 autos + 6 conv) out of the existing 16-ch `both` cache, in both frames. The column selection comes from `flatsky_cross_l1.op_feature_columns` so it cannot drift from the order `build_channels_*` actually emits, and the sliced auto block is asserted byte-identical to the `none` arm. |
| VMIM + sweep | `conv_arm.slurm <nobnt\|bnt>` (~70 min each) | 3 compressor seeds + 3 single sweeps + the ensemble sweep, chained into one job and idempotent (each stage skips if its output exists). |
| bars & marginals | the table builder, `--rows` pointing at the row spec | 12-row table; results in `data/table1/TABLE1_ERRORBARS_V2.json` and `data/table1/TABLE1_CONV_ARMS.json`. |

**Validation.** conv no-BNT seed 41 reproduces the published `operator_realnvp/conv/n9000`
= 2671 to +1.8 %, the closest agreement of any ℓ1 arm — an independent check that the
column-slicing approach is sound.

**BNT rows are new.** The published operator ladder (`operator_realnvp/`) is no-BNT only.
The `conv` and `conv+product` BNT centrals therefore have no published counterpart; flag
those two rows as new in the paper caption.

## ℓ1+conv+product and the `joint_*` directory names

The ℓ1+conv+product arm is the ordinary ℓ1 driver at `--cross-op both` (16 ch = 4 autos +
6 conv + 6 product). Its outputs live in `sweeps/joint_l1_nobnt_*`, `sweeps/joint_bnt_*`
and `vmim/both_*`. Those names are a historical mislabelling: the arm was briefly called
"joint ℓ1". **The statistic was always correct** — `--cross-op both` really is
autos + conv + product — only the label was wrong, and that data is the real
ℓ1+conv+product arm.

The genuine joint-ℓ1 arm is a different statistic entirely (the 2-D pairwise ℓ1-weighted
histogram, `--stat jointl1`, 3000-D), built by `../build_joint_arm.slurm` and written to
`joint/jointl1_*`. Keep the two directory families apart.

## NDE-variance investigation (`nde_variance*.slurm`)

Not part of the Table-1 sweep — a separate check that the ± convention actually captures
pipeline reproducibility. The quoted ± varies the **compressor** seed while the three
pooled NDE flow seeds (41/42/43) are held fixed for every arm. These jobs re-sweep a fixed
compressor with a different flow-seed set (51/52/53 or 61/62/63) so the paired difference
isolates the NDE term.

Finding: for the CNN the flow-set term is 0.74 % (no-BNT) and 1.42 % (BNT) — 2–3× the
compressor term, so the quoted CNN bar under-represents reproducibility. For the ℓ1 arms
the flow-set term is 0.3–1.5 % and never dominates the compressor term.

`nde_variance.slurm <nobnt|bnt>` does the CNN pair; `nde_variance_rows.slurm <seedset>
<label>:<armdir> …` is the generic form for any arm, with `<armdir>` relative to
`$NN_L1_SBI_DATA`.
