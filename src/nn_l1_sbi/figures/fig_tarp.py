"""fig:tarp -- expected-coverage (TARP-DRP) test, 3 arms, with the PROPER 1-sigma band.

Bug fixed (matches commit adef767): the earlier version used tarp.get_tarp_coverage(bootstrap=True),
whose built-in bootstrap resamples only the random REFERENCE points (per-bin std ~1e-4), so the band
was ~200x too small. The proper band bootstraps the 600 validation SIGHTLINES themselves
(1-sigma ~= +-0.02 = binomial SE sqrt(0.25/N)). The reported posterior pools the seeds/compressors
along the SAMPLE axis M, so it stays 600 sightlines (not the 1800 of stacking seeds as rows).
Validated: this reconstruct-from-stratified reproduces the committed CNN net +0.033, SE ~0.02, and
matches the un-stratified dumps_all theta exactly.

Method (= recompute_tarp_compare.py / recompute_unstrat_tarp_band.py): B=200 sightline-bootstrap
resamples; each ECP via get_tarp_coverage(references=random, norm=True) interpolated onto a common
alpha grid; report mean + 16-84 percentile (1-sigma). net = 2*int(ecp - alpha) d-alpha
(+ = conservative/over-covers, - = over-confident). Cached to derived/tarp_proper_bands.npz; pass
--recompute to rebuild (~5 min, joint l1 pools 9 sampler-sets). CPU.
"""
import matplotlib
matplotlib.use("Agg")
import numpy as np
import matplotlib.pyplot as plt
import tarp
import sys

from nn_l1_sbi.figures.style import use_style, ARMS, WIDTH_SINGLE
from nn_l1_sbi.figures._data import (A, CNNP, DERIVED, GEN_GATE, ensure_output_dir,
                                     output_root, require_glob)
HERE = output_root() / "figures"    # $NN_L1_SBI_OUT/figures -- never the installed package.
                                    # Only built here; it is created in main(), so importing
                                    # this module never touches the filesystem.
use_style()

ORDER = ["l1_auto", "l1_product", "jointl1", "cnn"]
TERC = ("LOW", "MID", "HIGH")
SEEDS = (41, 42, 43)
ALPHA = np.linspace(0.0, 1.0, 61)
NBOOT = 200
# each arm -> compressor dump dir(s); seeds pooled within. no-BNT: auto/+product single-compressor,
# joint l1 = 3-compressor ensemble. BNT: ALL analytic arms are 3-compressor ensembles (the calibration
# fix; single BNT is over-confident), CNN single in both. Matches fig:sbc.
ARM_DIRS = {
    "nobnt": {
        "l1_auto":    [f"{A}/gate_l1none_rnvp/tarp_drp/dumps"],
        "l1_product": [f"{A}/gate_l1product_rnvp/tarp_drp/dumps"],
        "jointl1":    [f"{A}/jointl1_nobnt/gate/tarp_drp/dumps",
                       f"{A}/jointl1_nobnt_s42/gate/tarp_drp/dumps",
                       f"{A}/jointl1_nobnt_s43/gate/tarp_drp/dumps"],
        "cnn":        [f"{CNNP}/arch_sweep_2026_06_13/gate_c_resnet18/dumps"],
    },
    "bnt": {
        "l1_auto":    [f"{A}/bnt_auto/gate/tarp_drp/dumps",
                       f"{A}/ens_bnt_auto_s42/gate/tarp_drp/dumps",
                       f"{A}/ens_bnt_auto_s43/gate/tarp_drp/dumps"],
        "l1_product": [f"{A}/gate_l1product_bnt_rnvp/tarp_drp/dumps",
                       f"{A}/ens_bnt_product_s42/gate/tarp_drp/dumps",
                       f"{A}/ens_bnt_product_s43/gate/tarp_drp/dumps"],
        "jointl1":    [f"{A}/jointl1_bnt/gate/tarp_drp/dumps",
                       f"{A}/jointl1_bnt_s42/gate/tarp_drp/dumps",
                       f"{A}/jointl1_bnt_s43/gate/tarp_drp/dumps"],
        "cnn":        [f"{CNNP}/bnt_resnet18_2026_06_14/gate_c_bnt/dumps"],
    },
}


def _cache(basis):
    return DERIVED / (f"tarp_proper_bands{'_bnt' if basis == 'bnt' else ''}.npz")


def _dump(dumps_dir, terc, seed):
    g = require_glob(f"{dumps_dir}/*_{terc}/seed_{seed}/n*_m*/posterior_samples.npz", GEN_GATE,
                     what=f"the {terc} / seed {seed} gate dump under {dumps_dir}")
    return g[0]


def pooled_600(arm, basis="nobnt"):
    """(samp (600,K,3), theta (600,3)): the reported posterior over the 600 un-stratified sightlines
    (the 3 terciles concatenated), with seeds + compressors pooled along the sample axis M. theta is
    taken from the first compressor (tercile-key pooling for the ensemble, validated ~ per-obs)."""
    dirs = ARM_DIRS[basis][arm]
    per_seed, theta_ref = [], None
    for seed in SEEDS:
        terc_S, terc_T = [], []
        for terc in TERC:
            comp = [np.load(_dump(dd, terc, seed))["samples"][:, :, :3].astype(np.float32) for dd in dirs]
            terc_S.append(np.concatenate(comp, axis=1))                                  # pool compressors along M
            terc_T.append(np.load(_dump(dirs[0], terc, seed))["theta"][:, :3].astype(np.float32))
        per_seed.append(np.concatenate(terc_S, 0))                                       # (600, len(dirs)*M, 3)
        if theta_ref is None:
            theta_ref = np.concatenate(terc_T, 0)
    return np.concatenate(per_seed, axis=1), theta_ref                                    # pool seeds along M


def _ecp_once(samp, theta):
    ecp, a = tarp.get_tarp_coverage(np.transpose(samp, (1, 0, 2)), theta, references="random", norm=True)
    ecp = np.asarray(ecp); ecp = ecp.mean(0) if ecp.ndim == 2 else ecp
    return np.interp(ALPHA, np.asarray(a), ecp)


def tarp_band(samp, theta, seed=0):
    """Sightline bootstrap: resample the N sightlines, recompute ECP each draw -> mean + 1-sigma."""
    rng = np.random.default_rng(seed)
    N = theta.shape[0]
    boot = np.empty((NBOOT, ALPHA.size))
    for b in range(NBOOT):
        idx = rng.integers(0, N, N)
        boot[b] = _ecp_once(samp[idx], theta[idx])
    mean = boot.mean(0)
    lo, hi = np.percentile(boot, [16, 84], 0)
    net = float(np.trapz(mean - ALPHA, ALPHA) * 2)
    se = float(boot[:, np.argmin(np.abs(ALPHA - 0.5))].std())
    return mean, lo, hi, net, se


def compute(basis):
    out = {}
    for arm in ORDER:
        s, t = pooled_600(arm, basis)
        mean, lo, hi, net, se = tarp_band(s, t)
        out[arm] = dict(mean=mean, lo=lo, hi=hi, net=net)
        print(f"[{basis}] {arm:11s} net {net:+.4f}  SE@0.5 {se:.4f}  N={t.shape[0]}  K={s.shape[1]}", flush=True)
    cache = _cache(basis); cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez(cache, alpha=ALPHA,
             **{f"{a}_{k}": out[a][k] for a in out for k in ("mean", "lo", "hi", "net")})
    return out


def main(basis="nobnt", recompute=False):
    cache = _cache(basis)
    if cache.exists() and not recompute:
        z = np.load(cache); alpha = z["alpha"]
        out = {a: {k: z[f"{a}_{k}"] for k in ("mean", "lo", "hi", "net")} for a in ORDER}
    else:
        out = compute(basis); alpha = ALPHA

    ensure_output_dir("figures", "_preview")   # HERE and its preview dir: created on write, not on import
    fig, ax = plt.subplots(figsize=(WIDTH_SINGLE, WIDTH_SINGLE))
    ax.plot([0, 1], [0, 1], "k--", lw=1.0, alpha=0.7, zorder=1, label="ideal")
    for arm in ORDER:
        c = ARMS[arm]["color"]; r = out[arm]
        ax.fill_between(alpha, r["lo"], r["hi"], color=c, alpha=0.22, lw=0, zorder=2)
        ax.plot(alpha, r["mean"], color=c, lw=1.5, zorder=3,
                label=f"{ARMS[arm]['label']} ({float(r['net']):+.3f})")
    ax.set_aspect("equal"); ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.set_xlabel(r"credibility level $\alpha$")
    ax.set_ylabel("expected coverage probability")
    ax.legend(loc="upper left", fontsize=6.6, frameon=False, handlelength=1.4)
    suffix = "_bnt" if basis == "bnt" else ""
    out_pdf = HERE / f"fig_tarp_overlay{suffix}.pdf"
    fig.savefig(out_pdf); fig.savefig(HERE / "_preview" / f"fig_tarp_overlay{suffix}.png", dpi=150)
    plt.close(fig)
    print("wrote", out_pdf.name)


if __name__ == "__main__":
    rc = "--recompute" in sys.argv
    for b in ("nobnt", "bnt"):          # no-BNT loads from cache (instant); BNT computes if cache missing
        main(b, recompute=rc)
