"""One-time GPU dump of the mean-observation posteriors for fig:corner (no BNT) and fig:bnt.

Reuses the matched-pipeline ensemble logic of cnn_sbi's plot_contour_ensemble.py: for each arm,
train the RealNVP on the (VMIM-compressed) cache(s) and sample at the noiseless mean observation;
the joint l1 arm pools its 3 compressor-seed flows. Reads the 10-D summary caches under
<data_root>/analysis/...; saves the (Om,s8,w0) samples + FoM3 + truth to
<output_root>/derived/contour_{nobnt,bnt}_samples.npz, so the figure scripts are pure-CPU getdist
replots. The figure scripts read that dump from <data_root>/derived, so once it is regenerated
either copy it into <data_root>/derived or point $NN_L1_SBI_DATA at a root that contains it.

Run once on a free GPU (checked nvidia-smi). Default device 2 (foreign tenant idle), mem-capped.
"""
import argparse
from types import SimpleNamespace
import numpy as np
from pathlib import Path

from nn_l1_sbi.figures._data import A, CNNP, DERIVED, require
from nn_l1_sbi.paths import output_root

OUTDIR = output_root() / "derived"     # writable; never the (read-only) data bundle
TMP = OUTDIR / "_tmp_dump"
GEN_SUMM = ("nn_l1_sbi.compression.vmim_from_cache (l1 arms) / "
            "nn_l1_sbi.cnn.npe_cnn_nbody_tomo + nn_l1_sbi.cnn.build_fiducial_summaries_cnn "
            "(CNN arm) -- the 10-D summary caches and fiducial summaries")

ARMS = {
 "nobnt": [
   dict(key="l1_product", prefix="l1",
        caches=[f"{A}/l1product_vmim_s41/cache"], fids=[f"{A}/l1product_vmim_s41/fiducial_summaries.npz"]),
   dict(key="jointl1", prefix="l1",
        caches=[f"{A}/jointl1_nobnt/cache", f"{A}/jointl1_nobnt_s42/cache", f"{A}/jointl1_nobnt_s43/cache"],
        fids=[f"{A}/jointl1_nobnt/fiducial_summaries.npz", f"{A}/jointl1_nobnt_s42/fiducial_summaries.npz", f"{A}/jointl1_nobnt_s43/fiducial_summaries.npz"]),
   dict(key="cnn", prefix="cnn",
        caches=[f"{CNNP}/arch_sweep_2026_06_13/cnn_resnet18_s41/cache"], fids=[f"{CNNP}/arch_sweep_2026_06_13/fidsumm_resnet18.npz"]),
   # auto-only (= "none" operator). VMIM cache + fid produced by vmim_from_cache.py from the raw
   # l1_none datavector (see docs/FIGURE_MAP.md "Newly computed"). Add via --only l1_auto --merge.
   dict(key="l1_auto", prefix="l1",
        caches=[f"{A}/operator_realnvp/none/cache"], fids=[f"{A}/operator_realnvp/none/fid_summ.npz"]),
 ],
 "bnt": [
   dict(key="l1_product", prefix="l1",
        caches=[f"{A}/l1product_bnt_vmim_s41/cache"], fids=[f"{A}/l1product_bnt_vmim_s41/fiducial_summaries.npz"]),
   dict(key="jointl1", prefix="l1",
        caches=[f"{A}/jointl1_bnt/cache", f"{A}/jointl1_bnt_s42/cache", f"{A}/jointl1_bnt_s43/cache"],
        fids=[f"{A}/jointl1_bnt/fiducial_summaries.npz", f"{A}/jointl1_bnt_s42/fiducial_summaries.npz", f"{A}/jointl1_bnt_s43/fiducial_summaries.npz"]),
   dict(key="cnn", prefix="cnn",
        caches=[f"{CNNP}/bnt_resnet18_2026_06_14/cnn_resnet18_bnt_s41/cache"], fids=[f"{CNNP}/bnt_resnet18_2026_06_14/fidsumm_bnt_resnet18_s41.npz"]),
   # auto-only under BNT. bnt_auto/cache is already the VMIM 10-D cache (FoM3 425, x0.17 of no-BNT
   # 2448). Add via --basis bnt --only l1_auto --merge.
   dict(key="l1_auto", prefix="l1",
        caches=[f"{A}/bnt_auto/cache"], fids=[f"{A}/bnt_auto/fiducial_summaries.npz"]),
 ],
}
FLOW = SimpleNamespace(flow_total_steps=50000, flow_batch_size=128, flow_lr_init=1e-3,
                       flow_lr_end=1e-5, flow_save_every=2000, flow_patience=20,
                       flow_grad_clip=1.0, flow_weight_decay=1e-4)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cuda-visible-devices", default="2")
    ap.add_argument("--m-samples", type=int, default=20000)
    ap.add_argument("--nde-seed", type=int, default=41)
    ap.add_argument("--max-perm", type=int, default=50)
    ap.add_argument("--basis", default="nobnt,bnt", help="comma list: which bases to (re)compute")
    ap.add_argument("--only", default="", help="comma list of arm keys to compute (default: all in basis)")
    ap.add_argument("--merge", action="store_true",
                    help="merge computed arms into the existing npz instead of overwriting (keeps truth + other arms)")
    a = ap.parse_args()
    bases = [b for b in a.basis.split(",") if b]
    only = set(k for k in a.only.split(",") if k)
    OUTDIR.mkdir(parents=True, exist_ok=True)
    from nn_l1_sbi.cnn.train_jaxili_from_compressed import setup_env, compute_fom3, marginal_stats
    setup_env(a.cuda_visible_devices)
    import jax
    from nn_l1_sbi.inference.npe_l1norm_cross_jaxili_nbody_tomo import preprocess_summaries, filter_zero_variance_bins
    from nn_l1_sbi.compression.train_nde_from_compressed import train_sbilens_realnvp

    for basis in bases:
        truth = None; saved = {}
        for cfg in ARMS[basis]:
            if only and cfg["key"] not in only:
                continue
            pooled = []
            for ci, (cache, fid) in enumerate(zip(cfg["caches"], cfg["fids"])):
                cdir = Path(cache)
                tr = np.load(require(cdir / f"{cfg['prefix']}_train.npz", GEN_SUMM))
                theta_tr = tr["theta"].astype(np.float32); x_tr_raw = tr["x"].astype(np.float64)
                x_va_raw = np.load(require(cdir / f"{cfg['prefix']}_val.npz", GEN_SUMM))["x"].astype(np.float64)
                tr_p, _, _, mean, std = preprocess_summaries(
                    x_tr_raw, x_va_raw[:1], x_va_raw[:1], summary_transform="none", clip_value=None)
                mask, _ = filter_zero_variance_bins(tr_p, min_variance=1e-12, verbose=False)
                x_tr = tr_p[:, mask].astype(np.float32); dim = x_tr.shape[1]
                fz = np.load(require(fid, GEN_SUMM)); S = fz["S"].astype(np.float64); perm = fz["perm"]
                sel = np.where(perm < a.max_perm)[0]
                if truth is None:
                    t = next((fz[k] for k in ("truth", "theta") if k in fz.files), None)
                    truth = (t[sel[0]] if np.ndim(t) == 2 else t)[:3] if t is not None else None
                _, _, fid_p, _, _ = preprocess_summaries(
                    x_tr_raw, x_va_raw[:1], S[sel], summary_transform="none", clip_value=None, mean=mean, std=std)
                x_single = fid_p[:, mask].astype(np.float32).mean(0)
                out = TMP / f"{basis}_{cfg['key']}_{ci}"; out.mkdir(parents=True, exist_ok=True)
                samplers = train_sbilens_realnvp(4, 128, theta_tr, x_tr, [a.nde_seed], a.m_samples, dim, out, FLOW)
                seed, fn = samplers[0]
                k = jax.random.PRNGKey(seed * 100003 + ci)
                ps = np.asarray(fn(x_single, k)); pooled.append(ps[np.all(np.isfinite(ps), 1)])
            ps = np.concatenate(pooled, 0)
            saved[cfg["key"]] = ps[:, :3]
            print(f"[{basis}/{cfg['key']}] FoM3={compute_fom3(ps)['fom3']:.0f} N={ps.shape[0]}", flush=True)
        out_npz = OUTDIR / f"contour_{basis}_samples.npz"
        # --merge merges into the dump we already wrote under $NN_L1_SBI_OUT if there is one, else
        # into the copy shipped in the (read-only) data bundle -- so merging a single arm on top of
        # the bundle keeps truth + the other arms instead of silently writing a one-arm file.
        prev_npz = out_npz if out_npz.exists() else DERIVED / f"contour_{basis}_samples.npz"
        if a.merge and prev_npz.exists():
            prev = np.load(prev_npz)
            merged = {k: prev[k] for k in prev.files}     # keep truth + previously-computed arms
            merged.update(saved)                          # overwrite/add the just-computed arms
            if (truth is not None) and (np.size(merged.get("truth", [])) != 3):
                merged["truth"] = truth
            np.savez(out_npz, **merged)
            print(f"merged {sorted(saved)} from {prev_npz} into {out_npz} "
                  f"(now: {sorted(k for k in merged if k != 'truth')})", flush=True)
        else:
            np.savez(out_npz, truth=(truth if truth is not None else np.array([])), **saved)
            print(f"wrote {out_npz}", flush=True)
        # the figure scripts read this dump from the data root, which may be read-only:
        print(f"  fig_corner/fig_bnt read {DERIVED / out_npz.name} -- copy {out_npz} there, or "
              f"set NN_L1_SBI_DATA to a root that contains it.", flush=True)


if __name__ == "__main__":
    main()
