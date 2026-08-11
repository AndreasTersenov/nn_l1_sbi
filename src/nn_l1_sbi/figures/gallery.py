"""The paper's figures: what they are, whether their inputs are here, and how to draw one.

This is the registry behind ``notebooks/reproduce_paper.ipynb`` section B. Each entry is one
``\\includegraphics`` in the manuscript, in the order the paper presents them. Diagnostic
figures that the repository ships but the paper does not use are deliberately absent -- see
``docs/FIGURE_MAP.md`` for those.

Usage::

    from nn_l1_sbi.figures import gallery
    gallery.status()             # which figures can be drawn here, and why not
    gallery.show("fig:cp")       # draw one, display its preview inline

Figures are drawn by running their generator as a subprocess, exactly as the release runs
them, so what you see is what ``python -m nn_l1_sbi.figures.fig_cp`` produces.
"""

from __future__ import annotations

import glob
import os
import subprocess
import sys
import time
from pathlib import Path

from nn_l1_sbi.figures import _data
from nn_l1_sbi.paths import output_root, repo_root

__all__ = ["PAPER_FIGURES", "status", "show", "where"]

_A = Path(_data.A)
_CNNP = Path(_data.CNNP)
_FROOT = Path(_data.ROOT)
_MAPS = Path(_data.MAPS)
_DERIVED = Path(_data.DERIVED)

_GEN_CONTOUR = "nn_l1_sbi.figures.dump_contour_samples  [GPU, retrains the flow]"
_GEN_TARP = "nn_l1_sbi.inference.run_tarp_coverage"
_GEN_GATE = "nn_l1_sbi.inference.run_joint_gate_c  [the TARP-DRP gate dumps]"
_GEN_SWEEP = "nn_l1_sbi.inference.population_sweep"

#: One entry per figure in the paper, in manuscript order.
PAPER_FIGURES = [
    dict(label="fig:data", sec="3 Statistics", pdf="fig_data_flatlocal.pdf",
         module="fig_data", args=(), previews=("fig_data.png",), cost="seconds",
         inputs=[_MAPS / "cosmo_fiducial_perm0.npz"],
         gen="nn_l1_sbi.data.build_full_sphere_cross_cache",
         what="the 4 auto and 12 cross channels of one 10 deg patch"),
    dict(label="fig:l1", sec="3 Statistics", pdf="fig_l1_hist_vs_s8.pdf",
         module="fig_l1", args=("--panel", "a"), previews=("fig_l1_hist.png",),
         cost="seconds", inputs=[_DERIVED / "l1_hist_vs_s8_levels_cache.npz"],
         gen="no surviving generator for the level cache",
         what="per-channel l1 histograms against SNR, at two values of sigma_8"),
    dict(label="fig:l1joint", sec="3 Statistics", pdf="fig_l1_joint_cellmap.pdf",
         module="fig_l1", args=("--panel", "b"), previews=("fig_l1_joint.png",),
         cost="seconds; loads a 432 MB npz",
         inputs=[_A / "jointl1_nobnt_raw" / "fid.npz"],
         gen="nn_l1_sbi.data.build_flatsky_joint_arm --out-fid",
         what="the joint l1 (u_i, u_j) cell map for a representative bin"),
    dict(label="fig:pipeline", sec="4 Inference", pdf="fig_pipeline.pdf",
         module="fig_pipeline", args=(), previews=("fig_pipeline.png",), cost="seconds",
         inputs=[_MAPS / "cosmo_fiducial_perm0.npz"],
         gen="nn_l1_sbi.data.build_full_sphere_cross_cache",
         what="the end-to-end pipeline schematic"),
    dict(label="fig:ablation", sec="5 Results", pdf="fig_ablation.pdf",
         module="fig_ablation", args=(), previews=("fig_ablation.png",), cost="seconds",
         inputs=[_A / "operator_realnvp" / x / "n9000" / "median_summary.json"
                 for x in ("conv", "both")],
         gen=f"{_GEN_SWEEP}, on the operator_realnvp arms",
         what="the operator completeness ladder, auto to +conv to +product to joint"),
    dict(label="fig:cp", sec="5 Results", pdf="fig_cp_fom3_bars.pdf",
         module="fig_cp", args=(), previews=("fig_cp.png",), cost="seconds",
         inputs=[Path(_data.PER_PATCH[k]) for k in ("l1_product", "jointl1", "cnn")],
         gen=_GEN_SWEEP,
         what="FoM3 and marginal violins across the 9000 mocks, for three arms"),
    dict(label="fig:corner", sec="5 Results", pdf="fig_corner_cnn_vs_l1.pdf",
         module="fig_corner", args=(), previews=("fig_corner.png",), cost="~10 s",
         inputs=[_DERIVED / "contour_nobnt_samples.npz"], gen=_GEN_CONTOUR,
         what="the mean-observation posterior, CNN against joint l1"),
    dict(label="fig:bnt", sec="6 BNT", pdf="fig_bnt_analytic.pdf + fig_bnt_learned.pdf",
         module="fig_bnt", args=(),
         previews=("fig_bnt_analytic.png", "fig_bnt_learned.png"), cost="~15 s",
         inputs=[_DERIVED / "contour_nobnt_samples.npz",
                 _DERIVED / "contour_bnt_samples.npz"], gen=_GEN_CONTOUR,
         what="what the BNT transform costs the analytic and the learned summaries"),
    dict(label="fig:tarp", sec="A0 Calibration", pdf="fig_tarp_overlay.pdf",
         module="fig_tarp", args=(), previews=("fig_tarp_overlay.png",),
         cost="cached; ~6 min --recompute",
         inputs=[_DERIVED / "tarp_proper_bands.npz"], gen=_GEN_TARP,
         what="TARP coverage in the no-BNT frame"),
    dict(label="fig:tarp_bnt", sec="A0 Calibration", pdf="fig_tarp_overlay_bnt.pdf",
         module="fig_tarp", args=(), previews=("fig_tarp_overlay_bnt.png",),
         cost="drawn by the same run as fig:tarp",
         inputs=[_DERIVED / "tarp_proper_bands_bnt.npz"], gen=_GEN_TARP,
         what="TARP coverage in the BNT frame"),
    dict(label="fig:sbc", sec="A0 Calibration", pdf="fig_sbc_hist_combined.pdf",
         module="fig_sbc", args=(), previews=("fig_sbc_hist_combined.png",), cost="~1-2 min",
         inputs=[_data.GATE_SINGLE["l1_auto"], _data.GATE_SINGLE["l1_product"],
                 _data.GATE_SINGLE["cnn"]], gen=_GEN_GATE,
         what="SBC rank histograms for the three quoted arms"),
    dict(label="fig:fisher", sec="A4 Fisher", pdf="fig_fisher.pdf",
         module="fig_fisher", args=(), previews=("fig_fisher.png",),
         cost="tens of s; reads a 1.04 GB cache",
         inputs=[_FROOT / "l1_matrix" / "l1_none_cache" / "flat_local_none" / "l1_train.npz"],
         gen="nn_l1_sbi.inference.npe_l1norm_cross_jaxili_nbody_tomo (the raw 800-D cache)",
         what="per-bin diagonal Fisher information of the 800-D auto l1"),
]


def _exists(p) -> bool:
    p = str(p)
    return bool(glob.glob(p)) if any(c in p for c in "*?[") else Path(p).exists()


def _resolved():
    """Attach input availability to every entry (recomputed on each call)."""
    for f in PAPER_FIGURES:
        f["have"] = [_exists(x) for x in f["inputs"]]
        f["n_have"], f["n_need"] = sum(f["have"]), len(f["inputs"])
        f["ready"] = f["n_have"] == f["n_need"]
    return PAPER_FIGURES


def _figdir() -> Path:
    return Path(__file__).resolve().parent


def _out() -> Path:
    return output_root() / "figures"


def where() -> dict:
    """The three roots that matter, for reporting."""
    return {"data": Path(_data.data_root()), "figures out": _out(),
            "generators": _figdir(), "style": _figdir() / "paper.mplstyle"}


def status(verbose: bool = True) -> list:
    """Report which paper figures can be drawn from the data present here."""
    figs = _resolved()
    if verbose:
        w = where()
        for k in ("data", "figures out"):
            mark = "" if w[k].is_dir() else "   <-- does not exist"
            print(f"{k:<14}: {w[k]}{mark}")
        print(f"{'style sheet':<14}: "
              f"{'found' if w['style'].is_file() else 'MISSING -- no figure can be drawn'}")
        print()
        print(f"{'figure':<14}{'status':<10}{'inputs':<9}{'section':<17}what it shows")
        print("-" * 100)
        for f in figs:
            st = "ready" if f["ready"] else "no input"
            print(f"{f['label']:<14}{st:<10}{f['n_have']}/{f['n_need']:<7}"
                  f"{f['sec']:<17}{f['what'][:44]}")
        print("-" * 100)
        n = sum(f["ready"] for f in figs)
        print(f"{n} of {len(figs)} paper figures can be drawn here.")
    return figs


def status_markdown() -> str:
    """The same report as :func:`status`, as a Markdown table."""
    figs = _resolved()
    w = where()
    n = sum(f["ready"] for f in figs)
    out = [f"**{n} of {len(figs)}** paper figures can be drawn from the data present here.",
           "", f"- data root: `{w['data']}`",
           f"- figures written to: `{w['figures out']}`",
           f"- style sheet: {'found' if w['style'].is_file() else '**missing**'}", "",
           "| figure | paper section | inputs | what it shows |",
           "|---|---|:--:|---|"]
    for f in figs:
        mark = "yes" if f["ready"] else f"**no** ({f['n_have']}/{f['n_need']})"
        out.append(f"| `{f['label']}` | {f['sec']} | {mark} | {f['what']} |")
    return "\n".join(out)


def show(label: str, timeout: int = 1800):
    """Draw one paper figure and display it, or explain precisely why it cannot be drawn."""
    figs = {f["label"]: f for f in _resolved()}
    f = figs[label]
    print(f"{f['label']}   ({f['sec']})   ->  {f['pdf']}")

    if not f["ready"]:
        print(f"  not drawn: {f['n_have']}/{f['n_need']} inputs present")
        for p, ok in zip(f["inputs"], f["have"]):
            if not ok:
                print(f"    missing  {p}")
        print(f"  produced by: {f['gen']}")
        print("  Nothing is substituted; see docs/FIGURE_MAP.md.")
        return

    if not (_figdir() / "paper.mplstyle").is_file():
        print("  not drawn: paper.mplstyle is missing")
        return

    out = _out()
    src = repo_root()
    env = dict(os.environ, NN_L1_SBI_OUT=str(output_root()), MPLBACKEND="Agg")
    if src is not None:
        env["PYTHONPATH"] = os.pathsep.join(
            [str(src / "src"), os.environ.get("PYTHONPATH", "")]).rstrip(os.pathsep)

    def snap():
        return {q: q.stat().st_mtime_ns for q in out.rglob("*")
                if q.suffix.lower() in (".pdf", ".png")} if out.is_dir() else {}

    before, t0 = snap(), time.time()
    proc = subprocess.run([sys.executable, "-m", f"nn_l1_sbi.figures.{f['module']}", *f["args"]],
                          env=env, cwd=str(src) if src else None,
                          capture_output=True, text=True, timeout=timeout)
    dt = time.time() - t0
    produced = sorted(q for q, m in snap().items() if before.get(q) != m)

    if proc.returncode != 0:
        print(f"  FAILED (exit {proc.returncode}, {dt:.1f} s)")
        for line in proc.stderr.strip().splitlines()[-12:]:
            print("   |", line)
        return

    pdfs = [q.name for q in produced if q.suffix == ".pdf"]
    print(f"  drawn in {dt:.1f} s: {', '.join(pdfs) if pdfs else 'no new pdf'}")

    # A generator may write more previews than the paper uses (fig_bnt also writes a
    # no-product variant). Show only the manuscript's panels; disclose the rest.
    prev = [q for q in produced if q.parent.name == "_preview" and q.suffix == ".png"]
    extra = [q.name for q in prev if q.name not in f["previews"]]
    if extra:
        print(f"  (also written, not a paper panel: {', '.join(extra)})")
    try:
        from IPython.display import Image, display
    except ImportError:
        return
    for q in prev:
        if q.name in f["previews"]:
            display(Image(filename=str(q)))
