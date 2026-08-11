"""Reproduce Table 1 of the paper from the committed per-arm digests.

Everything here reads only ``data/table1/``, which ships in the repository: 48
``median_summary.json`` files (12 arms x 3 compressor seeds + 12 ensembles) plus four
precomputed tables used as cross-checks. No download, no GPU, no network.

The four rules that define the table
------------------------------------

1. **The error bar** is the standard deviation (ddof=1) over the three *pre-ensemble*
   compressor seeds s41/s42/s43, carried as a **relative** quantity ``std/mean`` and
   rescaled onto whichever central the row quotes.
2. **The central** is the published value where one exists, otherwise the retrained value
   (s41 for a row quoted as ``single``, the pooled ensemble for one quoted as ``ensemble``).
3. **The single -> ensemble shift is a bias**, reported in its own column and never summed
   into the error bar.
4. **The marginals** sigma(Om), sigma(s8), sigma(w0) track the row's quoted estimator, and
   get error bars by the same relative rule as the FoM.

Arms are resolved to directories *only* through ``arms.json``. Never derive a directory
name from an arm label: ``l1 +conv+product`` lives in ``joint_l1_*`` while the separate
``joint l1`` arm lives in ``jointl1_*``, and the two are different measurements.

Usage::

    from nn_l1_sbi.table1 import bundle
    arms, refs = bundle.load_all()
    rows = bundle.build_rows(arms)
    print(bundle.format_table(rows))
    bundle.verify(rows, refs)
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np

__all__ = ["PARAMS", "bundle_dir", "load", "load_all", "build_rows", "format_table",
           "verify", "published_vs_retrained", "VerificationError"]

#: The marginal-width keys, in the paper's parameter order (Omega_m, sigma_8, w_0).
PARAMS = ("sigma_Om", "sigma_s8", "sigma_w0")

#: How many checks :func:`verify` runs when nothing is missing.
#: 4 coverage + EV2 12x14 + CONV 4x31 + PAPER 8x7 + MARG 2x11.
EXPECTED_CHECKS = 4 + 12 * 14 + 4 * 31 + 8 * 7 + 2 * 11      # = 374


class VerificationError(AssertionError):
    """Raised when the rebuilt table disagrees with the committed cross-checks."""


def bundle_dir() -> Path:
    """Locate ``data/table1``.

    ``$NN_L1_SBI_REPO`` wins if set; otherwise the package locates its own clone. Falls
    back to walking up from the working directory, so a bare checkout works too.
    """
    env = os.environ.get("NN_L1_SBI_REPO")
    if env:
        p = Path(env).expanduser().resolve() / "data" / "table1"
        if (p / "arms.json").is_file():
            return p
        raise FileNotFoundError(f"NN_L1_SBI_REPO={env} has no data/table1/arms.json")

    from nn_l1_sbi.paths import table1_dir
    p = table1_dir()
    if (p / "arms.json").is_file():
        return p

    here = Path.cwd().resolve()
    for q in (here, *here.parents):
        if (q / "data" / "table1" / "arms.json").is_file():
            return q / "data" / "table1"
    raise FileNotFoundError(
        "cannot find data/table1/arms.json. Set NN_L1_SBI_REPO to the nn_l1_sbi checkout, "
        "or run from inside it.")


def load(name: str) -> dict:
    """Load one JSON file from the Table-1 bundle."""
    with open(bundle_dir() / name) as fh:
        return json.load(fh)


def load_all():
    """Return ``(arms, refs)``: the arm map, and the four cross-check tables by short name."""
    arms = load("arms.json")
    refs = {"EV2": load("TABLE1_ERRORBARS_V2.json"),
            "CONV": load("TABLE1_CONV_ARMS.json"),
            "PAPER": load("TABLE1_PAPER.json"),
            "MARG": load("TABLE1_MARGINALS_NEW.json")}
    return arms, refs


def rel_spread(values) -> float:
    """The error bar as a relative quantity: ``std(ddof=1) / mean``. Rule 1."""
    v = np.asarray(values, dtype=float)
    return float(np.std(v, ddof=1) / np.mean(v))


def build_rows(arms=None) -> list:
    """Apply the four rules to every arm and return one dict per row."""
    if arms is None:
        arms = load("arms.json")
    rows = []
    for label, a in arms.items():
        singles = [load(p) for p in a["singles"].values()]      # order: s41, s42, s43
        ens = load(a["ensemble"])
        src = singles[0] if a["quoted"] == "single" else ens    # rule 2: which estimator

        per_seed = [s["fom3"] for s in singles]
        rel = rel_spread(per_seed)                              # rule 1
        retrained = src["fom3"]
        central = a["published"] if a["published"] is not None else retrained

        sig = [src[k] for k in PARAMS]                          # rule 4
        sig_rel = [rel_spread([s[k] for s in singles]) for k in PARAMS]

        mean_ps = float(np.mean(per_seed))
        summary, frame = (t.strip() for t in label.split(","))
        rows.append(dict(
            label=label, summary=summary, frame=frame, quoted=a["quoted"],
            published=a["published"], retrained=retrained,
            central=central, pm=central * rel, rel=rel,
            per_seed=per_seed, mean=mean_ps,
            std=float(np.std(per_seed, ddof=1)),
            lo=float(min(per_seed)), hi=float(max(per_seed)),
            ensemble=ens["fom3"],
            # rule 3: two baselines exist. bias_mean is what the table prints; bias_s41 is
            # the convention the stored cross-checks use. For 'l1 auto, BNT' they differ in
            # sign, so verify() asserts both.
            bias_mean=100.0 * (ens["fom3"] - mean_ps) / mean_ps,
            bias_s41=100.0 * (ens["fom3"] - per_seed[0]) / per_seed[0],
            sigma=sig, sigma_pm=[v * r for v, r in zip(sig, sig_rel)], sigma_rel=sig_rel,
            sigma_ens=[ens[k] for k in PARAMS],
            sigma_per_seed=[[s[k] for s in singles] for k in PARAMS]))
    return rows


_SEP = "-" * 116


def format_table(rows) -> str:
    """Render the rows as the paper's Table 1."""
    out = [_SEP,
           f"{'arm':<20}{'frame':<8}{'quoted':<10}{'FoM3 +-':>15}"
           f"{'sigma(Om)':>18}{'sigma(s8)':>18}{'sigma(w0)':>18}{'bias%':>9}",
           _SEP]
    for r in rows:
        star = " " if r["published"] is not None else "*"
        fom = f"{r['central']:.0f} +- {r['pm']:.0f}{star}"
        sig = "".join(f"{v:.4f} +- {p:.4f}".rjust(18)
                      for v, p in zip(r["sigma"], r["sigma_pm"]))
        out.append(f"{r['summary']:<20}{r['frame']:<8}{r['quoted']:<10}{fom:>15}"
                   f"{sig}{r['bias_mean']:>+9.1f}")
    out += [_SEP,
            "+-    std over the three pre-ensemble compressor seeds, as std/mean, rescaled",
            "      onto the quoted central. Same rule for the marginals.",
            "*     no published value (new result); the central is the retrained one.",
            "bias% single -> ensemble shift. A bias, reported separately, never in the +-."]
    return "\n".join(out)


def published_vs_retrained(rows) -> str:
    """Show the published central beside the value this bundle retrains to."""
    out = [f"{'arm':<20}{'frame':<8}{'quoted':<10}{'published':>11}"
           f"{'retrained':>11}{'delta %':>10}{'+- (rel %)':>12}",
           "-" * 82]
    for r in rows:
        pub = f"{r['published']}" if r["published"] is not None else "new"
        delta = ("--" if r["published"] is None
                 else f"{100.0 * (r['retrained'] - r['published']) / r['published']:+.1f}")
        out.append(f"{r['summary']:<20}{r['frame']:<8}{r['quoted']:<10}{pub:>11}"
                   f"{r['retrained']:>11.1f}{delta:>10}{100.0 * r['rel']:>12.2f}")
    return "\n".join(out)


def load_per_obs(arm: str = "auto_nobnt_s41") -> dict:
    """The per-observation posterior covariances for one arm.

    ``data/table1/per_obs/<arm>.npz`` ships in the repository (430 kB): the 3x3 posterior
    covariance of (Om, s8, w0) for each of the 9000 mock observations, plus the patch and
    noise-realisation index of each. This is one step upstream of the digests -- it is what
    the digests were reduced *from* -- so the table can be recomputed rather than read.
    """
    z = np.load(bundle_dir() / "per_obs" / f"{arm}.npz")
    return {k: z[k] for k in z.files}


def fom3_from_cov(cov) -> np.ndarray:
    """FoM3 = 1/sqrt(det Cov), per observation. The definition, applied directly."""
    return 1.0 / np.sqrt(np.linalg.det(np.asarray(cov, dtype=float)))


def block_bootstrap(values, patch, n_rep: int = 10000, seed: int = 0, chunk: int = 500):
    """Bootstrap the median by resampling *patches*, not observations.

    The 9000 mocks are 180 patches x 50 noise realisations, so they are not independent:
    values sharing a patch share its signal realisation. Resampling observations would
    therefore understate the error. This resamples the 180 patches with replacement, keeping
    all 50 realisations of each, as ``docs/NOTE_FOM_ERROR_BARS.md`` specifies.

    Returns ``(se, lo, hi)``: the standard deviation of the bootstrap medians and the 16th
    and 84th percentiles.
    """
    values = np.asarray(values, dtype=float)
    patch = np.asarray(patch)
    ids = np.unique(patch)
    counts = np.array([(patch == p).sum() for p in ids])
    if len(set(counts)) != 1:
        raise ValueError(f"unbalanced patches: sizes {sorted(set(counts))}")
    # (n_patch, reps_per_patch) so a replicate is a row resample
    grouped = np.stack([values[patch == p] for p in ids])
    rng = np.random.default_rng(seed)
    meds = np.empty(n_rep)
    for i in range(0, n_rep, chunk):                       # chunked: 10^4 x 9000 at once is 720 MB
        k = min(chunk, n_rep - i)
        idx = rng.integers(0, len(ids), size=(k, len(ids)))
        meds[i:i + k] = np.median(grouped[idx].reshape(k, -1), axis=1)
    return float(meds.std(ddof=1)), float(np.percentile(meds, 16)), float(np.percentile(meds, 84))


def table_markdown(rows) -> str:
    """Table 1 as a Markdown table, for notebooks and anything that renders Markdown."""
    out = ["| arm | frame | quoted | FoM₃ | σ(Ω_m) | σ(σ₈) | σ(w₀) | bias % |",
           "|---|---|---|--:|--:|--:|--:|--:|"]
    for r in rows:
        star = "" if r["published"] is not None else " \\*"
        sig = " | ".join(f"{v:.4f} ± {p:.4f}" for v, p in zip(r["sigma"], r["sigma_pm"]))
        out.append(f"| {r['summary']} | {r['frame']} | {r['quoted']} | "
                   f"**{r['central']:.0f} ± {r['pm']:.0f}**{star} | {sig} | {r['bias_mean']:+.1f} |")
    out += ["", "**±** — spread over the three pre-ensemble compressor seeds, as `std/mean`, "
                "rescaled onto the quoted central; the marginals follow the same rule.  ",
            "**\\*** — no published value (new result); the central is the retrained one.  ",
            "**bias %** — the single→ensemble shift, reported separately and never folded "
            "into the ±."]
    return "\n".join(out)


def verify(rows, refs=None, verbose=True) -> int:
    """Check the rebuilt rows against every committed cross-check table.

    Returns the number of checks run. Raises :class:`VerificationError` on any mismatch, and
    also if the number of checks itself changes -- a suite that silently stopped checking
    (a truncated cross-check file, a dropped arm) must not be able to report success.
    """
    if refs is None:
        _, refs = load_all()
    EV2, CONV, PAPER, MARG = refs["EV2"], refs["CONV"], refs["PAPER"], refs["MARG"]
    by_label = {r["label"]: r for r in rows}
    failures, n = [], 0

    def check(name, got, want, rtol=1e-9, atol=1e-9):
        nonlocal n
        n += 1
        ok = (got == want if want is None or isinstance(want, (str, bool))
              else bool(np.isclose(float(got), float(want), rtol=rtol, atol=atol)))
        if not ok:
            failures.append(f"{name}: got {got!r}, expected {want!r}")

    # Coverage: the loops below iterate the STORED dicts, so their lengths are load-bearing.
    check("coverage / EV2 arms", len(EV2), 12)
    check("coverage / CONV arms", len(CONV), 4)
    check("coverage / PAPER rows", len(PAPER), 8)
    check("coverage / MARG rows", len(MARG), 2)

    for r in rows:                                          # all 12 arms
        e, tag = EV2[r["label"]], r["label"]
        check(f"{tag} / published", r["published"], e["published"])
        check(f"{tag} / quoted", r["quoted"], e["quoted"])
        for i, s in enumerate(("s41", "s42", "s43")):
            check(f"{tag} / per_seed {s}", r["per_seed"][i], e["per_seed"][s])
        for k in ("mean", "std", "lo", "hi", "ensemble"):
            check(f"{tag} / {k}", r[k], e[k])
        check(f"{tag} / pct", 100.0 * r["rel"], e["pct"])
        check(f"{tag} / central (retrained)", r["retrained"], e["central"])
        check(f"{tag} / bias_pct (s41 baseline)", r["bias_s41"], e["bias_pct"])
        # The convention the table PRINTS. Different baseline, sometimes a different sign.
        check(f"{tag} / bias_mean (mean baseline)", r["bias_mean"],
              100.0 * (e["ensemble"] - e["mean"]) / e["mean"])

    for label, c in CONV.items():                           # the 4 conv arms, with sigmas
        r = by_label[label]
        check(f"{label} / published_fom", r["published"], c["published_fom"])
        check(f"{label} / quoted", r["quoted"], c["quoted"])
        check(f"{label} / fom_central", r["central"], c["fom_central"])
        check(f"{label} / fom_pm", r["pm"], c["fom_pm"])
        check(f"{label} / fom_rel_pct", 100.0 * r["rel"], c["fom_rel_pct"])
        check(f"{label} / fom_ensemble", r["ensemble"], c["fom_ensemble"])
        check(f"{label} / bias_pct", r["bias_s41"], c["bias_pct"])
        for i, s in enumerate(c["fom_per_seed"]):
            check(f"{label} / fom_per_seed[{i}]", r["per_seed"][i], s)
        for j, k in enumerate(PARAMS):
            s = c["sigma"][k]
            check(f"{label} / {k} central", r["sigma"][j], s["central"])
            check(f"{label} / {k} pm", r["sigma_pm"][j], s["pm"])
            check(f"{label} / {k} rel_pct", 100.0 * r["sigma_rel"][j], s["rel_pct"])
            check(f"{label} / {k} ensemble", r["sigma_ens"][j], s["ensemble"])
            for i, v in enumerate(s["per_seed"]):
                check(f"{label} / {k} per_seed[{i}]", r["sigma_per_seed"][j][i], v)

    for label, p in PAPER.items():                          # 8 main rows, stored precision
        r = by_label[label]
        check(f"{label} / paper published", r["published"], p["published"])
        check(f"{label} / paper n_seeds", p["n_seeds"], 3)
        check(f"{label} / paper rel_pct", round(100.0 * r["rel"], 2), p["rel_pct"])
        # Assert the value the table prints, not a fresh re-derivation of it: re-deriving
        # would pass even if pm were computed wrongly upstream.
        check(f"{label} / paper pm", round(r["pm"], 1), p["pm"])
        check(f"{label} / paper retrained", round(r["retrained"], 1), p["retrained"])
        check(f"{label} / paper pm_retrained",
              round(r["retrained"] * r["rel"], 1), p["pm_retrained"])
        # median_se needs per_patch_metrics.npz, so only its consistency is checked here.
        check(f"{label} / paper median_se", round(EV2[label]["median_se"], 2), p["median_se"])

    for label, m in MARG.items():                           # the 2 new no-BNT conv rows
        r = by_label[label]
        check(f"{label} / marg published_fom", r["published"], m["published_fom"])
        check(f"{label} / marg fom_pm", r["pm"], m["fom_pm"])
        for j, k in enumerate(PARAMS):
            check(f"{label} / marg {k} value", r["sigma"][j], m[k]["value"])
            check(f"{label} / marg {k} pm", r["sigma_pm"][j], m[k]["pm"])
            check(f"{label} / marg {k} rel_pct", 100.0 * r["sigma_rel"][j], m[k]["rel_pct"])

    if failures:
        raise VerificationError(
            f"Table 1 does not reproduce: {len(failures)} of {n} checks failed.\n"
            + "\n".join("  " + f for f in failures[:20])
            + ("\n  ..." if len(failures) > 20 else ""))
    if n != EXPECTED_CHECKS:
        raise VerificationError(
            f"assertion coverage changed: {n} checks ran, {EXPECTED_CHECKS} expected. A "
            "cross-check file is truncated, an arm is missing, or an assertion was "
            "added or removed. The table is NOT validated.")
    if verbose:
        print(f"OK: {n} checks passed against all four committed cross-check tables "
              f"(both bias conventions included).")
    return n
