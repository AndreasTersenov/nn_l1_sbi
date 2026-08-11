"""Shared data-loading for the paper figures.

All paths resolve under ONE configurable data root, `nn_l1_sbi.paths.data_root()`, which this
module re-exports (there is exactly one implementation, in nn_l1_sbi.paths). Every figure
script imports the loader (or the root constant) it needs from here, so the data provenance
lives in one place. Figures are WRITTEN under `nn_l1_sbi.paths.output_root()` -- see
figures_dir() -- never next to the installed package. Nothing is recomputed except where a
script says so (the GPU contour dump).

Committed headline numbers (no-BNT = single-compressor; joint l1 = its 3-compressor ensemble):
  l1 auto-only 2448 | l1+product 3045 | joint l1 3371 | CNN 3326
  no-BNT analytical FoM3 are robust to compressor ensembling to <=1.2% (auto->2429, product->3009;
  RESULT_NOBNT_ENSEMBLE_ROBUSTNESS.md) -- reported as a robustness check, not the headline.
  BNT: l1 auto 388 (x0.16) | l1+product 718 (x0.24) | joint l1 2424 (x0.72) | CNN 3186 (x0.96)
  BNT per-channel l1 (auto, +product) = 3-compressor ENSEMBLE (the single BNT is over-confident:
  single product 779/auto 425); the ensemble is the calibrated value.

Data bundle layout
------------------
Everything below data_root(), the root of the unpacked data bundle:

    <data_root>/
      maps/       cosmo_fiducial_perm0.npz              fig:data, fig:pipeline
                  fullsphere_nobnt_cosmo_fiducial_perm0.npz    fig:leak (a)+(b)
      derived/    contour_{nobnt,bnt}_samples.npz       fig:corner, fig:bnt
                  tarp_proper_bands{,_bnt}.npz          fig:tarp{,_bnt}
                  embed_tsne.npz                        fig:embed
                  l1_hist_vs_s8_levels_cache.npz        fig:l1 (a)
                  geometry/<arm>/per_patch_grid.csv     fig:w0
      analysis/   the heavy tier, mirroring the original analysis tree verbatim below
                  <cnn_sbi>/scripts/sbi/results/exploratory/ :
                  flatsky_cross_2026_06/{analytical_nde_match,cnn_phase,l1_matrix,gate_c,
                                         population_sweep,whiten_campaign,bnt_campaign,
                                         bntdeep_campaign,bntdeep2_campaign}/...
                  cross_maps_campaign/..., definitive_comparison_10deg/...

`maps/` and `derived/` are the small, portable tiers (the Zenodo record); `analysis/` is the
gate-dump / 10-D-cache / 800-D-cache tier that only the --recompute paths and the GPU contour
dump need. The `derived/` files are NOT in the analysis tree at all: they are produced by the
generators named in the require() messages below.
"""
import glob
import os
import re
import numpy as np
from pathlib import Path

# Single source of truth for every root in the package. data_root() and output_root() are
# re-exported here unchanged, so the figure scripts keep importing them from _data.
from nn_l1_sbi.paths import data_root, ensure_output_dir, output_root
from nn_l1_sbi.paths import require as _require

DATA_ENV = "NN_L1_SBI_DATA"


def figures_dir() -> Path:
    """Where every figure PDF (and its raster preview) is written: output_root()/"figures".

    Never the package directory: an installed package is read-only, and a figure written next
    to the code is a figure nobody finds. Created on first use.
    """
    return ensure_output_dir("figures")


def require(path, produced_by, what=None):
    """Return `path`, or raise naming the file AND the generator that produces it.

    A thin wrapper over nn_l1_sbi.paths.require that keeps the `what` (human description)
    argument the figure scripts pass. Several figure inputs (the contour dumps, the TARP bands,
    the SBC gate dumps, the fig:l1 panel-a cache, the fig:bntwhiten sweeps, the fig:w0 grids)
    are derived products that are not in the analysis tree. A missing one must say so, not
    surface as a bare FileNotFoundError or -- worse -- a silently wrong plot.
    """
    p = Path(path)
    return _require(p, produced_by=(
        f"{produced_by}\n"
        f"    (it provides {what or p.name}; the figure data root is currently {data_root()})"))


def require_glob(pattern, produced_by, what=None):
    """sorted(glob(pattern)), or raise naming the pattern AND its generator (see require)."""
    hits = sorted(glob.glob(str(pattern)))
    if hits:
        return hits
    raise FileNotFoundError(
        f"missing figure input: no file matches {pattern}\n"
        f"  ({what or 'these dumps'}) are produced by: {produced_by}\n"
        f"  data root is {data_root()} -- set ${DATA_ENV} to the unpacked data bundle, "
        f"or regenerate the dumps with the generator above.")


MAPS = data_root() / "maps"
DERIVED = data_root() / "derived"
ANALYSIS = data_root() / "analysis"
# generators quoted by require(): keep these in sync with docs/FIGURE_MAP.md
GEN_GATE = ("nn_l1_sbi.inference.run_joint_gate_c (TARP-DRP gate; the dumps themselves are "
            "written by nn_l1_sbi.inference.tarp_stratified_val)")
GEN_SWEEP = "nn_l1_sbi.inference.population_sweep / nn_l1_sbi.inference.ensemble_eval"

# CNN_SBI is the bundle stand-in for the old $SBI/results/exploratory (the analysis tree);
# ROOT/A/CNNP keep the sub-structure of that tree verbatim.
CNN_SBI = str(ANALYSIS)
ROOT = f"{CNN_SBI}/flatsky_cross_2026_06"
A = f"{ROOT}/analytical_nde_match"
CNNP = f"{ROOT}/cnn_phase"

# ---- per-patch metric files (9000-obs medians; the violins) ----------
# no-BNT product = single-compressor headline; joint l1 = ensemble. BNT per-channel l1 = ensemble
# (the calibrated value, single BNT over-confident). CNN single throughout.
PER_PATCH = {
    "l1_product": f"{A}/l1product_rnvp_s41_n9000/per_patch_metrics.npz",
    "jointl1":    f"{A}/jointl1_nobnt_ensemble/per_patch_metrics.npz",
    "cnn":        f"{CNNP}/arch_sweep_2026_06_13/cnn_resnet18_s41/readout_full/per_patch_metrics.npz",
    "l1_product_bnt": f"{A}/product_bnt_ensemble/per_patch_metrics.npz",
    "jointl1_bnt":    f"{A}/jointl1_bnt_ensemble/per_patch_metrics.npz",
}

# ---- gate dumps for TARP / SBC ----------------------------------------
GATE_SINGLE = {
    "l1_auto": f"{A}/gate_l1none_rnvp/tarp_drp/dumps/l1none_rnvp_*/seed_*/n*_m*/posterior_samples.npz",
    "l1_product": f"{A}/gate_l1product_rnvp/tarp_drp/dumps/l1product_rnvp_*/seed_*/n*_m*/posterior_samples.npz",
    "cnn": f"{CNNP}/arch_sweep_2026_06_13/gate_c_resnet18/dumps/resnet18_rnvp_*/seed_*/n*_m*/posterior_samples.npz",
}
GATE_ENSEMBLE = ["jointl1_nobnt", "jointl1_nobnt_s42", "jointl1_nobnt_s43"]

# ---- BNT counterparts (same four arms, BNT basis) ---------------------
# joint l1 single-compressor BNT FAILS SBC (std ~0.333); the 3-compressor ensemble is the
# headline arm and recovers calibration (~0.304), exactly as in the no-BNT case.
GATE_SINGLE_BNT = {
    "l1_auto": f"{A}/bnt_auto/gate/tarp_drp/dumps/bnt_auto_*/seed_*/n*_m*/posterior_samples.npz",
    "l1_product": f"{A}/gate_l1product_bnt_rnvp/tarp_drp/dumps/l1product_bnt_rnvp_*/seed_*/n*_m*/posterior_samples.npz",
    "cnn": f"{CNNP}/bnt_resnet18_2026_06_14/gate_c_bnt/dumps/bnt_resnet18_rnvp_*/seed_*/n*_m*/posterior_samples.npz",
}
GATE_ENSEMBLE_BNT = ["jointl1_bnt", "jointl1_bnt_s42", "jointl1_bnt_s43"]
# BNT auto/+product 3-compressor ensembles (calibration fix 2026-06-26). The s41 single arm's gate
# lives in a different layout from the new s42/s43, so these are explicit globs (one per compressor).
GATE_ENSEMBLE_BNT_GLOBS = {
    "l1_product": [
        f"{A}/gate_l1product_bnt_rnvp/tarp_drp/dumps/*/seed_*/n*_m*/posterior_samples.npz",
        f"{A}/ens_bnt_product_s42/gate/tarp_drp/dumps/*/seed_*/n*_m*/posterior_samples.npz",
        f"{A}/ens_bnt_product_s43/gate/tarp_drp/dumps/*/seed_*/n*_m*/posterior_samples.npz",
    ],
    "l1_auto": [
        f"{A}/bnt_auto/gate/tarp_drp/dumps/*/seed_*/n*_m*/posterior_samples.npz",
        f"{A}/ens_bnt_auto_s42/gate/tarp_drp/dumps/*/seed_*/n*_m*/posterior_samples.npz",
        f"{A}/ens_bnt_auto_s43/gate/tarp_drp/dumps/*/seed_*/n*_m*/posterior_samples.npz",
    ],
}


def load_per_patch(key):
    """Return dict with sigma (N,3), fom3 (N,), finite mask, medians."""
    d = np.load(require(PER_PATCH[key], GEN_SWEEP, what=f"per-patch metrics for {key}"))
    f = d["fom3"]; g = np.isfinite(f)
    s = d["sigma"]
    return dict(sigma=s[g], fom3=f[g], n=int(g.sum()),
                med_sigma=np.median(s[g], 0), med_fom3=float(np.median(f[g])))


def _pool_single(pattern):
    S, T = [], []
    for f in require_glob(pattern, GEN_GATE, what="the TARP/SBC gate dumps"):
        z = np.load(f)
        S.append(np.asarray(z["samples"], np.float32)[:, :, :3])
        T.append(np.asarray(z["theta"], np.float32)[:, :3])
    return (np.concatenate(S, 0), np.concatenate(T, 0)) if S else (None, None)


def _key(f):
    m = re.search(r"dumps/(.+?)_(LOW|MID|HIGH)/seed_(\d+)/", f)
    return (m.group(2), m.group(3)) if m else None


def _pool_ensemble(arm_dirs):
    per_arm = []
    for adir in arm_dirs:
        d = {}
        for f in require_glob(f"{A}/{adir}/gate/tarp_drp/dumps/*/seed_*/n*_m*/posterior_samples.npz",
                              GEN_GATE, what=f"the {adir} gate dumps"):
            k = _key(f)
            if k:
                z = np.load(f)
                d[k] = (np.asarray(z["samples"], np.float32)[:, :, :3],
                        np.asarray(z["theta"], np.float32)[:, :3])
        per_arm.append(d)
    keys = sorted(set.intersection(*[set(d) for d in per_arm]))
    S, T = [], []
    for k in keys:
        S.append(np.concatenate([d[k][0] for d in per_arm], axis=1))
        T.append(per_arm[0][k][1])
    return np.concatenate(S, 0), np.concatenate(T, 0)


def load_gate(arm):
    """Return (samples (N,M,3), theta (N,3)) pooled gate dumps for an arm."""
    if arm == "jointl1":
        return _pool_ensemble(GATE_ENSEMBLE)
    return _pool_single(GATE_SINGLE[arm])


# ---- per-NDE-seed ranks (for the SBC band) ----------------------------
# The 3 NDE seeds (41/42/43) are re-trainings on the SAME test theta (verified
# byte-identical), so the independent SBC unit is ONE seed's test set (~600), not
# the 3x-correlated pool of 1800. SBC bars/curves = mean over seeds; the uniform
# band is sized to the per-seed count. This mirrors compute_sbc_from_tarp_dumps.py.
def _ranks(samples, theta):
    return (np.asarray(samples) < np.asarray(theta)[:, None, :]).mean(1)[:, :3]


def _single_by_seed(pattern):
    """{seed:int -> ranks (N_seed,3)} pooling the 3 terciles within each NDE seed."""
    bucket = {}
    for f in require_glob(pattern, GEN_GATE, what="the TARP/SBC gate dumps"):
        m = re.search(r"seed_(\d+)/", f)
        if not m:
            continue
        seed = int(m.group(1))
        z = np.load(f)
        bucket.setdefault(seed, []).append(
            (np.asarray(z["samples"], np.float32)[:, :, :3],
             np.asarray(z["theta"], np.float32)[:, :3]))
    return {seed: _ranks(np.concatenate([a for a, _ in lst], 0),
                         np.concatenate([b for _, b in lst], 0))
            for seed, lst in bucket.items()}


def _ensemble_by_seed(arm_dirs):
    """{seed:int -> ranks (N_seed,3)} for the deep ensemble: the 3 compressors pooled
    along M, the 3 terciles pooled along N, kept separate per NDE seed."""
    per_arm = []
    for adir in arm_dirs:
        d = {}
        for f in require_glob(f"{A}/{adir}/gate/tarp_drp/dumps/*/seed_*/n*_m*/posterior_samples.npz",
                              GEN_GATE, what=f"the {adir} gate dumps"):
            k = _key(f)
            if k:
                z = np.load(f)
                d[k] = (np.asarray(z["samples"], np.float32)[:, :, :3],
                        np.asarray(z["theta"], np.float32)[:, :3])
        per_arm.append(d)
    keys = sorted(set.intersection(*[set(d) for d in per_arm]))
    by_seed = {}
    for terc, seed in keys:
        by_seed.setdefault(int(seed), []).append((terc, seed))
    out = {}
    for seed, klist in by_seed.items():
        S = [np.concatenate([d[k][0] for d in per_arm], axis=1) for k in klist]
        T = [per_arm[0][k][1] for k in klist]
        out[seed] = _ranks(np.concatenate(S, 0), np.concatenate(T, 0))
    return out


def _load_dumps(globpat):
    """{(tercile, seed_str) -> (samples (N,M,3), theta (N,3))} for one compressor's gate dumps."""
    d = {}
    for f in require_glob(globpat, GEN_GATE, what="the TARP/SBC gate dumps"):
        k = _key(f)
        if k:
            z = np.load(f)
            d[k] = (np.asarray(z["samples"], np.float32)[:, :, :3],
                    np.asarray(z["theta"], np.float32)[:, :3])
    return d


def _ensemble_by_seed_globs(globs):
    """{NDE-seed -> ranks} pooling the compressors (explicit globs) along M, by (tercile, seed)
    key. For the BNT auto/+product ensembles whose s41 dir differs in layout from s42/s43."""
    per = [_load_dumps(g) for g in globs]
    keys_all = sorted(set.intersection(*[set(d) for d in per]))
    out = {}
    for s in sorted({int(k[1]) for k in keys_all}):
        ks = [k for k in keys_all if int(k[1]) == s]
        R = [_ranks(np.concatenate([d[k][0] for d in per], axis=1), per[0][k][1]) for k in ks]
        if R:
            out[s] = np.concatenate(R, 0)
    return out


def load_gate_per_seed(arm, basis="nobnt"):
    """{NDE-seed -> ranks (N_seed,3)} for an arm (terciles pooled within seed).
    basis in {"nobnt","bnt"}; joint l1 uses the 3-compressor ensemble in both, and under BNT so do
    auto/+product (the 2026-06-26 calibration fix)."""
    if basis == "bnt":
        if arm == "jointl1":
            return _ensemble_by_seed(GATE_ENSEMBLE_BNT)
        if arm in GATE_ENSEMBLE_BNT_GLOBS:
            return _ensemble_by_seed_globs(GATE_ENSEMBLE_BNT_GLOBS[arm])
        return _single_by_seed(GATE_SINGLE_BNT[arm])
    if arm == "jointl1":
        return _ensemble_by_seed(GATE_ENSEMBLE)
    return _single_by_seed(GATE_SINGLE[arm])
