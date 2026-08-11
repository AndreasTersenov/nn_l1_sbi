"""Table 1 of the paper, reproduced from ``data/table1/`` and checked number by number.

This is the repository's central scientific claim in test form. It reads only the
committed 60-file Table-1 bundle (~110 kB): **stdlib only, no install, no network,
no GPU, well under a second.**

What "independent" means here
-----------------------------
The recipe below is re-implemented from its written specification
(``data/table1/README.md`` §6 and ``docs/NOTE_FOM_ERROR_BARS.md`` §5-6). It does
not import ``notebooks/reproduce_paper.ipynb``, ``make_table.py`` or anything
else that produced the numbers, and it uses :mod:`statistics` rather than NumPy,
so a NumPy-side change cannot make the test and the artefact agree by
construction. The precomputed ``TABLE1_*.json`` files are treated purely as
*cross-check oracles*: the test computes each value from the 54
``median_summary.json`` measurements and then demands the stored value agrees.

The recipe (the only thing that defines the +- in Table 1)
----------------------------------------------------------
For every arm:

* ``per_seed`` = ``fom3`` of the three **pre-ensemble compressor singles**
  ``s41, s42, s43`` -- in that order; ``s41`` is index 0.
* ``rel`` = ``stdev(per_seed, ddof=1) / mean(per_seed)`` -- the error bar, carried
  as a *relative* quantity.
* ``central`` = the published value when the paper has one, else the retrained
  value of whichever estimator the row quotes (``s41`` for ``quoted="single"``,
  the pooled ensemble for ``quoted="ensemble"``).
* ``pm`` = ``central * rel`` -- the relative bar rescaled onto the quoted central.

The single -> ensemble shift is a **bias**. It is reported separately and is never
summed into the ``+-``. The leave-one-out jackknife in ``final_bars_all_v2.json``
is a diagnostic and is not the bar. ``median_se`` is a caption-only subdominant
term and is not recomputable from this bundle.

Non-vacuity
-----------
An audit found that assertions of this shape can pass while checking nothing --
loop over a cross-check file that has been truncated to zero arms and every
assertion inside the loop is skipped. Every check here is therefore counted, and
each test asserts the exact number of comparisons it performed against a literal
derived from the documented bundle shape (12 arms x 3 singles + 1 ensemble, 54
sweep directories, 12/4/2/8 arms in the four cross-check files). Delete an arm,
truncate a file, or silently stop comparing a field, and the count moves and the
suite fails.
"""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
from statistics import fmean, stdev

REPO_ROOT = Path(__file__).resolve().parents[1]
TABLE1_DIR = REPO_ROOT / "data" / "table1"

# ---------------------------------------------------------------------------
# The published table. Central and +-, exactly as printed in the paper.
# ---------------------------------------------------------------------------
PUBLISHED_TABLE1 = {
    "l1 auto, no-BNT": (2448, 27),
    "l1 auto, BNT": (388, 43),
    "l1 +conv, no-BNT": (2671, 133),
    "l1 +conv, BNT": (458, 15),
    "l1 +product, no-BNT": (3045, 183),
    "l1 +product, BNT": (718, 29),
    "l1 +conv+product, no-BNT": (3255, 200),
    "l1 +conv+product, BNT": (704, 76),
    "joint l1, no-BNT": (3371, 96),
    "joint l1, BNT": (2424, 208),
    "CNN, no-BNT": (3326, 14),
    "CNN, BNT": (3186, 19),
}

# Canonical arm order: 6 summaries x 2 frames.
ARM_ORDER = tuple(PUBLISHED_TABLE1)

# The retrained table (bundle README §7): FoM3, raw spread over the three
# singles, and the three marginals of the quoted estimator. Rounded as published.
RETRAINED_TABLE = {
    #                            FoM3     std     sigma_Om  sigma_s8  sigma_w0
    "l1 auto, no-BNT": (2776.9, 30.5, 0.0503, 0.0799, 0.2340),
    "l1 auto, BNT": (390.7, 49.3, 0.0861, 0.1725, 0.3191),
    "l1 +conv, no-BNT": (2720.2, 129.9, 0.0518, 0.0814, 0.2366),
    "l1 +conv, BNT": (457.6, 16.7, 0.0836, 0.1643, 0.3155),
    "l1 +product, no-BNT": (3231.8, 199.3, 0.0462, 0.0746, 0.2271),
    "l1 +product, BNT": (758.3, 37.7, 0.0721, 0.1293, 0.2994),
    "l1 +conv+product, no-BNT": (3564.9, 207.4, 0.0452, 0.0714, 0.2202),
    "l1 +conv+product, BNT": (703.9, 84.9, 0.0755, 0.1349, 0.3030),
    "joint l1, no-BNT": (3379.5, 108.0, 0.0440, 0.0716, 0.2234),
    "joint l1, BNT": (2405.2, 259.6, 0.0513, 0.0856, 0.2366),
    "CNN, no-BNT": (3427.6, 14.1, 0.0444, 0.0702, 0.2288),
    "CNN, BNT": (3147.1, 18.5, 0.0461, 0.0730, 0.2311),
}

# Which estimator each row quotes, per the bundle README §3. Asserted against
# arms.json, so a silent flip of an arm from "single" to "ensemble" -- which
# would change a published central -- fails the suite.
QUOTED = {
    "l1 auto, no-BNT": "single",
    "l1 auto, BNT": "ensemble",
    "l1 +conv, no-BNT": "single",
    "l1 +conv, BNT": "ensemble",
    "l1 +product, no-BNT": "single",
    "l1 +product, BNT": "ensemble",
    "l1 +conv+product, no-BNT": "single",
    "l1 +conv+product, BNT": "ensemble",
    "joint l1, no-BNT": "ensemble",
    "joint l1, BNT": "ensemble",
    "CNN, no-BNT": "single",
    "CNN, BNT": "single",
}

# The two rows that are new results: no previously published central, so the
# retrained ensemble is quoted instead.
NEW_ROWS = ("l1 +conv, BNT", "l1 +conv+product, BNT")

SINGLE_KEYS = ("s41", "s42", "s43")
SIGMA_KEYS = ("sigma_Om", "sigma_s8", "sigma_w0")

N_ARMS = 12
N_SINGLES_PER_ARM = 3
N_SWEEP_DIRS = 54  # 12 arms x 4, plus the 6 extra-seed dirs behind the n=6 caveat
N_REFERENCED_DIRS = 48  # the 4 x 12 that arms.json points at

# Cross-check files and how many arms each is documented to carry. Asserted
# exactly: a truncated oracle must fail loudly, not silently check nothing.
CROSSCHECK_ARM_COUNTS = {
    "TABLE1_ERRORBARS_V2.json": 12,
    "TABLE1_CONV_ARMS.json": 4,
    "TABLE1_MARGINALS_NEW.json": 2,
    "TABLE1_PAPER.json": 8,
}

REL_TOL = 1e-9
ABS_TOL = 1e-15


# ---------------------------------------------------------------------------
# Counted assertions
# ---------------------------------------------------------------------------
class Checks:
    """An assertion counter. Every comparison a test makes goes through it."""

    def __init__(self) -> None:
        self.n = 0

    def eq(self, got, want, what: str) -> None:
        self.n += 1
        assert got == want, f"{what}: got {got!r}, want {want!r}"

    def close(self, got, want, what: str, rel: float = REL_TOL) -> None:
        self.n += 1
        assert isinstance(got, float) and isinstance(want, float), (
            f"{what}: expected floats, got {type(got).__name__} / {type(want).__name__}"
        )
        assert math.isclose(got, want, rel_tol=rel, abs_tol=ABS_TOL), (
            f"{what}: got {got!r}, want {want!r} (rel diff "
            f"{abs(got - want) / max(abs(want), 1e-300):.3e})"
        )

    def true(self, cond: bool, what: str) -> None:
        self.n += 1
        assert cond, what


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------
def _load(relpath: str):
    with open(TABLE1_DIR / relpath, encoding="utf-8") as fh:
        return json.load(fh)


def _arms():
    return _load("arms.json")


def _recipe(entry: dict) -> dict:
    """The Table-1 recipe for one arm. The only place it is implemented."""
    singles = [_load(entry["singles"][k]) for k in SINGLE_KEYS]
    ensemble = _load(entry["ensemble"])
    source = singles[0] if entry["quoted"] == "single" else ensemble

    per_seed = [s["fom3"] for s in singles]
    rel = stdev(per_seed) / fmean(per_seed)  # stdev() is the ddof=1 sample stdev
    central = entry["published"] if entry["published"] is not None else source["fom3"]

    sigma_central, sigma_rel = {}, {}
    for key in SIGMA_KEYS:
        vals = [s[key] for s in singles]
        sigma_central[key] = source[key]
        sigma_rel[key] = stdev(vals) / fmean(vals)

    return {
        "singles": singles,
        "ensemble": ensemble,
        "source": source,
        "per_seed": per_seed,
        "rel": rel,
        "central": central,
        "pm": central * rel,
        "retrained": source["fom3"],
        "bias_pct": 100.0 * (ensemble["fom3"] - per_seed[0]) / per_seed[0],
        "sigma_central": sigma_central,
        "sigma_rel": sigma_rel,
        "sigma_pm": {k: sigma_central[k] * sigma_rel[k] for k in SIGMA_KEYS},
    }


def _sweep_dir(relpath: str) -> str:
    """``sweeps/<dir>/median_summary.json`` -> ``<dir>``."""
    return Path(relpath).parent.name


# ---------------------------------------------------------------------------
# The checks. Each returns the number of comparisons it performed.
# ---------------------------------------------------------------------------
def _check_bundle_shape(c: Checks) -> int:
    """The bundle is the documented shape, so nothing below can pass vacuously."""
    c.true(TABLE1_DIR.is_dir(), f"missing Table-1 bundle: {TABLE1_DIR}")
    arms = _arms()
    c.eq(len(arms), N_ARMS, "number of arms in arms.json")
    c.eq(tuple(arms), ARM_ORDER, "arm labels and their order in arms.json")

    referenced = []
    for label, entry in arms.items():
        c.eq(tuple(entry["singles"]), SINGLE_KEYS, f"{label}: single keys and order")
        c.true(isinstance(entry["ensemble"], str), f"{label}: ensemble path is a string")
        c.true(
            entry["published"] is None or isinstance(entry["published"], int),
            f"{label}: published is int or null",
        )
        c.eq(entry["quoted"], QUOTED[label], f"{label}: quoted estimator")
        c.eq(
            entry["published"] is None,
            label in NEW_ROWS,
            f"{label}: published is null exactly for the two new rows",
        )
        referenced += [entry["singles"][k] for k in SINGLE_KEYS] + [entry["ensemble"]]

    c.eq(len(referenced), N_ARMS * 4, "paths referenced by arms.json")
    c.eq(len(set(referenced)), N_REFERENCED_DIRS, "distinct paths referenced by arms.json")
    for rel in referenced:
        c.true((TABLE1_DIR / rel).is_file(), f"referenced measurement missing: {rel}")

    on_disk = sorted(TABLE1_DIR.glob("sweeps/*/median_summary.json"))
    c.eq(len(on_disk), N_SWEEP_DIRS, "median_summary.json files on disk")

    n_members = {1: 0, 3: 0}
    for path in on_disk:
        with open(path, encoding="utf-8") as fh:
            d = json.load(fh)
        name = path.parent.name
        c.eq(d["arm"], name, f"{name}: 'arm' field matches its directory")
        c.eq(d["n"], 9000, f"{name}: number of mock observations")
        c.eq(d["nde_seeds"], [41, 42, 43], f"{name}: pooled NDE seeds")
        c.true(
            all(isinstance(d[k], float) for k in ("fom3",) + SIGMA_KEYS),
            f"{name}: fom3 and the three sigmas are floats",
        )
        n_members[d["n_members"]] = n_members.get(d["n_members"], 0) + 1
    c.eq(n_members[3], N_ARMS, "directories that are 3-member ensembles")
    c.eq(n_members[1], N_SWEEP_DIRS - N_ARMS, "directories that are single compressors")

    for fname, n_expected in CROSSCHECK_ARM_COUNTS.items():
        c.eq(len(_load(fname)), n_expected, f"{fname}: number of arms")
    c.eq(len(_load("final_bars_all_v2.json")), 8, "final_bars_all_v2.json: number of rows")
    return c.n


def _check_published_table1(c: Checks) -> int:
    """The headline: all 12 published (central +- pm) pairs, from the measurements."""
    arms = _arms()
    c.eq(len(arms), N_ARMS, "arms loaded")
    n_arms_checked = 0
    for label in ARM_ORDER:
        r = _recipe(arms[label])
        c.eq(len(r["singles"]), N_SINGLES_PER_ARM, f"{label}: compressor singles loaded")
        want_central, want_pm = PUBLISHED_TABLE1[label]
        c.eq(round(r["central"]), want_central, f"{label}: published central")
        c.eq(round(r["pm"]), want_pm, f"{label}: published +-")
        n_arms_checked += 1
    c.eq(n_arms_checked, N_ARMS, "arms actually compared against the published table")
    return c.n


def _check_retrained_table(c: Checks) -> int:
    """The retrained centrals and all three marginals, for all 12 arms."""
    arms = _arms()
    c.eq(len(arms), N_ARMS, "arms loaded")
    for label in ARM_ORDER:
        r = _recipe(arms[label])
        fom, spread, s_om, s_s8, s_w0 = RETRAINED_TABLE[label]
        c.eq(round(r["retrained"], 1), fom, f"{label}: retrained FoM3")
        c.eq(round(stdev(r["per_seed"]), 1), spread, f"{label}: raw spread over 3 singles")
        for key, want in zip(SIGMA_KEYS, (s_om, s_s8, s_w0)):
            c.eq(round(r["sigma_central"][key], 4), want, f"{label}: {key}")
    return c.n


def _check_errorbars_v2(c: Checks) -> int:
    """Cross-check every field of the canonical 12-arm oracle."""
    arms = _arms()
    oracle = _load("TABLE1_ERRORBARS_V2.json")
    c.eq(tuple(oracle), ARM_ORDER, "TABLE1_ERRORBARS_V2.json: arms and their order")
    for label in ARM_ORDER:
        e, r = oracle[label], _recipe(arms[label])
        c.eq(e["published"], arms[label]["published"], f"{label}: published mirrors arms.json")
        c.eq(e["quoted"], arms[label]["quoted"], f"{label}: quoted mirrors arms.json")
        c.eq(tuple(e["per_seed"]), SINGLE_KEYS, f"{label}: per_seed keys")
        for i, key in enumerate(SINGLE_KEYS):
            c.close(r["per_seed"][i], e["per_seed"][key], f"{label}: per_seed[{key}]")
        c.close(fmean(r["per_seed"]), e["mean"], f"{label}: mean")
        c.close(stdev(r["per_seed"]), e["std"], f"{label}: std (ddof=1)")
        c.close(min(r["per_seed"]), e["lo"], f"{label}: lo")
        c.close(max(r["per_seed"]), e["hi"], f"{label}: hi")
        c.close(100.0 * r["rel"], e["pct"], f"{label}: pct (relative error bar)")
        c.close(r["ensemble"]["fom3"], e["ensemble"], f"{label}: ensemble fom3")
        c.close(r["source"]["fom3"], e["central"], f"{label}: central (retrained)")
        # bias_pct is stored against the s41 baseline, NOT against the mean.
        c.close(r["bias_pct"], e["bias_pct"], f"{label}: bias_pct (s41 baseline)")
        c.true(math.isfinite(e["median_se"]), f"{label}: median_se present and finite")
    return c.n


def _check_conv_arms(c: Checks) -> int:
    """The sigma-carrying oracle: 4 arms, FoM3 + full sigma triplets with per-seed spread."""
    arms = _arms()
    oracle = _load("TABLE1_CONV_ARMS.json")
    c.eq(len(oracle), CROSSCHECK_ARM_COUNTS["TABLE1_CONV_ARMS.json"], "conv arms: count")
    for label, o in oracle.items():
        c.true(label in arms, f"{label}: is an arm in arms.json")
        r = _recipe(arms[label])
        c.eq(o["published_fom"], arms[label]["published"], f"{label}: published_fom")
        c.eq(o["quoted"], arms[label]["quoted"], f"{label}: quoted")
        c.eq(
            o["source"],
            "new (retrained ensemble)" if label in NEW_ROWS else "published",
            f"{label}: source",
        )
        c.close(float(r["central"]), float(o["fom_central"]), f"{label}: fom_central")
        c.close(r["pm"], o["fom_pm"], f"{label}: fom_pm")
        c.close(100.0 * r["rel"], o["fom_rel_pct"], f"{label}: fom_rel_pct")
        c.eq(len(o["fom_per_seed"]), N_SINGLES_PER_ARM, f"{label}: fom_per_seed length")
        for i, want in enumerate(o["fom_per_seed"]):
            c.close(r["per_seed"][i], want, f"{label}: fom_per_seed[{i}]")
        c.close(r["ensemble"]["fom3"], o["fom_ensemble"], f"{label}: fom_ensemble")
        c.close(r["bias_pct"], o["bias_pct"], f"{label}: bias_pct")
        c.eq(tuple(o["sigma"]), SIGMA_KEYS, f"{label}: sigma keys")
        for key in SIGMA_KEYS:
            s = o["sigma"][key]
            c.close(r["sigma_central"][key], s["central"], f"{label}/{key}: central")
            c.close(r["sigma_pm"][key], s["pm"], f"{label}/{key}: pm")
            c.close(100.0 * r["sigma_rel"][key], s["rel_pct"], f"{label}/{key}: rel_pct")
            c.close(r["ensemble"][key], s["ensemble"], f"{label}/{key}: ensemble")
            c.eq(len(s["per_seed"]), N_SINGLES_PER_ARM, f"{label}/{key}: per_seed length")
            for i, want in enumerate(s["per_seed"]):
                c.close(r["singles"][i][key], want, f"{label}/{key}: per_seed[{i}]")
    return c.n


def _check_marginals_new(c: Checks) -> int:
    """The sigma digest the paper text quotes for the two new no-BNT conv rows."""
    arms = _arms()
    oracle = _load("TABLE1_MARGINALS_NEW.json")
    c.eq(len(oracle), CROSSCHECK_ARM_COUNTS["TABLE1_MARGINALS_NEW.json"], "marginals: count")
    for label, o in oracle.items():
        c.true(label in arms, f"{label}: is an arm in arms.json")
        r = _recipe(arms[label])
        c.eq(o["published_fom"], arms[label]["published"], f"{label}: published_fom")
        c.close(r["pm"], o["fom_pm"], f"{label}: fom_pm")
        for key in SIGMA_KEYS:
            s = o[key]
            c.close(r["sigma_central"][key], s["value"], f"{label}/{key}: value")
            c.close(r["sigma_pm"][key], s["pm"], f"{label}/{key}: pm")
            c.close(100.0 * r["sigma_rel"][key], s["rel_pct"], f"{label}/{key}: rel_pct")
    return c.n


def _check_paper_digest(c: Checks) -> int:
    """The 8 paper-facing rows, including the n=6 caveat fields.

    ``n6_*`` is the only consumer of the six extra ``s44/s45/s46`` directories.
    An arm gets the 6-seed treatment iff those directories exist; the mapping is
    derived from the arms.json path, never from the label.
    """
    arms = _arms()
    oracle = _load("TABLE1_PAPER.json")
    c.eq(len(oracle), CROSSCHECK_ARM_COUNTS["TABLE1_PAPER.json"], "paper digest: rows")
    n_six_seed = 0
    for label, o in oracle.items():
        c.true(label in arms, f"{label}: is an arm in arms.json")
        entry = arms[label]
        r = _recipe(entry)
        c.eq(o["published"], entry["published"], f"{label}: published")
        c.eq(round(r["pm"], 1), o["pm"], f"{label}: pm")
        c.eq(round(100.0 * r["rel"], 2), o["rel_pct"], f"{label}: rel_pct")
        c.eq(round(r["retrained"], 1), o["retrained"], f"{label}: retrained")
        c.eq(round(r["retrained"] * r["rel"], 1), o["pm_retrained"], f"{label}: pm_retrained")
        c.true(math.isfinite(o["median_se"]), f"{label}: median_se present and finite")
        c.eq(o["n_seeds"], N_SINGLES_PER_ARM, f"{label}: n_seeds")

        s41_dir = _sweep_dir(entry["singles"]["s41"])
        six = [s41_dir.replace("_s41", f"_s{k}") for k in range(41, 47)]
        have_six = all((TABLE1_DIR / "sweeps" / d / "median_summary.json").is_file() for d in six)
        if have_six:
            n_six_seed += 1
            vals = [
                _load(f"sweeps/{d}/median_summary.json")["fom3"] for d in six
            ]
            rel6 = stdev(vals) / fmean(vals)
            c.eq(o["n6_seeds"], 6, f"{label}: n6_seeds")
        else:
            rel6 = r["rel"]
            c.eq(o["n6_seeds"], N_SINGLES_PER_ARM, f"{label}: n6_seeds")
        c.eq(round(100.0 * rel6, 2), o["n6_rel_pct"], f"{label}: n6_rel_pct")
        c.eq(round(r["central"] * rel6, 1), o["n6_pm"], f"{label}: n6_pm")
    c.eq(n_six_seed, 2, "rows backed by six compressor seeds")
    return c.n


def _check_arm_directory_mapping(c: Checks) -> int:
    """The naming trap: ``joint_l1_*`` and ``jointl1_*`` are different runs."""
    arms = _arms()
    dirs = {
        label: tuple(_sweep_dir(entry["singles"][k]) for k in SINGLE_KEYS)
        + (_sweep_dir(entry["ensemble"]),)
        for label, entry in arms.items()
    }

    conv_product = dirs["l1 +conv+product, no-BNT"]
    joint_l1 = dirs["joint l1, no-BNT"]
    c.true(
        all(d.startswith("joint_l1_") for d in conv_product),
        f"'l1 +conv+product, no-BNT' must map to joint_l1_* dirs, got {conv_product}",
    )
    c.true(
        all(d.startswith("jointl1_") for d in joint_l1),
        f"'joint l1, no-BNT' must map to jointl1_* dirs, got {joint_l1}",
    )
    c.eq(set(conv_product) & set(joint_l1), set(), "the two joint arms share no directory")

    # Not merely different names: different runs with different numbers.
    a = _recipe(arms["l1 +conv+product, no-BNT"])
    b = _recipe(arms["joint l1, no-BNT"])
    for i in range(N_SINGLES_PER_ARM):
        c.true(
            a["per_seed"][i] != b["per_seed"][i],
            f"joint_l1_* and jointl1_* seed {SINGLE_KEYS[i]} have identical fom3 -- "
            "the two arms have been collapsed onto one run",
        )
    c.true(
        a["ensemble"]["fom3"] != b["ensemble"]["fom3"],
        "joint_l1_* and jointl1_* ensembles have identical fom3",
    )

    # The trap itself: string-munging the label gives the *other* arm's directory.
    munged = "joint" + "l1" + "_nobnt_s41"  # 'joint l1' with the space removed
    c.eq(munged, joint_l1[0], "label-munging 'joint l1' lands on jointl1_nobnt_s41")
    c.true(
        munged != conv_product[0],
        "label-munging must not be able to reach the '+conv+product' directory",
    )

    # Every arm has its own four directories, and every directory is used once.
    all_dirs = [d for tup in dirs.values() for d in tup]
    c.eq(len(all_dirs), N_ARMS * 4, "directories referenced")
    c.eq(len(set(all_dirs)), N_REFERENCED_DIRS, "distinct directories referenced")
    c.eq(len(dirs), N_ARMS, "arms with a directory mapping")
    return c.n


def _check_bar_is_spread_not_bias(c: Checks) -> int:
    """The +- is the pre-ensemble spread, rescaled -- not the single->ensemble shift.

    Three confusable quantities (bundle README §6): the quotable bar
    ``central * rel``, the raw spread ``stdev(per_seed)`` (which is that same
    relative bar sitting on ``mean(per_seed)``), and the bias
    ``ensemble - s41``. This asserts the published bar is the first one.
    """
    arms = _arms()
    oracle = _load("TABLE1_ERRORBARS_V2.json")
    n_arms = 0
    for label in ARM_ORDER:
        r = _recipe(arms[label])
        raw = stdev(r["per_seed"])
        mean = fmean(r["per_seed"])
        n_arms += 1
        # The bar is carried as a relative quantity: it sits on the quoted
        # central with exactly the same fractional size it has on the mean.
        c.close(r["pm"] / r["central"], raw / mean, f"{label}: bar is relative")
        # ...and the quoted central is never the mean of the singles, so the
        # quoted bar is never the raw, unrescaled spread.
        c.true(abs(r["central"] - mean) > 1e-9, f"{label}: central is not mean(per_seed)")
        c.true(
            abs(r["pm"] - raw) > 1e-9,
            f"{label}: +- equals the unrescaled spread; the relative bar was not "
            "rescaled onto the quoted central",
        )
        # The single -> ensemble shift is a bias, and is never the bar.
        bias = abs(oracle[label]["ensemble"] - r["per_seed"][0])
        c.true(
            abs(r["pm"] - bias) > 1e-9,
            f"{label}: +- coincides with the single->ensemble bias",
        )
    c.eq(n_arms, N_ARMS, "arms checked for bar-vs-bias confusion")
    return c.n


# Every check, and the exact number of comparisons each performs. These literals
# follow from the documented bundle shape; if the bundle changes, they move.
_CHECKS = (
    # 3 + 12 arms x 5 + 2 + 48 files + 1 + 54 files x 4 + 2 + 4 + 1
    (_check_bundle_shape, 337),
    # 1 + 12 arms x 3 + 1
    (_check_published_table1, 38),
    # 1 + 12 arms x (FoM3, spread, 3 marginals)
    (_check_retrained_table, 61),
    # 1 + 12 arms x 15 fields
    (_check_errorbars_v2, 181),
    # 1 + 4 arms x (14 FoM3 fields + 3 sigmas x 8)
    (_check_conv_arms, 153),
    # 1 + 2 arms x (3 + 3 sigmas x 3)
    (_check_marginals_new, 25),
    # 1 + 8 rows x 11 fields + 1
    (_check_paper_digest, 90),
    (_check_arm_directory_mapping, 12),
    # 12 arms x 4 + 1
    (_check_bar_is_spread_not_bias, 49),
)
EXPECTED_TOTAL_CHECKS = 946


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------
def test_bundle_shape():
    """12 arms x (3 singles + 1 ensemble), 54 measurements, 4 intact oracles."""
    assert _check_bundle_shape(Checks()) == 337


def test_published_table1_reproduces():
    """All 12 published (central +- pm) pairs, recomputed from the measurements."""
    assert _check_published_table1(Checks()) == 38


def test_retrained_centrals_and_marginals():
    """The retrained FoM3, spread and three marginals, for all 12 arms."""
    assert _check_retrained_table(Checks()) == 61


def test_errorbars_v2_crosscheck():
    """Every field of the canonical 12-arm oracle agrees with the recipe."""
    assert _check_errorbars_v2(Checks()) == 181


def test_conv_arms_crosscheck():
    """The 4 conv arms, including per-seed sigma triplets."""
    assert _check_conv_arms(Checks()) == 153


def test_marginals_new_crosscheck():
    """The 2 new no-BNT conv rows the paper text quotes."""
    assert _check_marginals_new(Checks()) == 25


def test_paper_digest_crosscheck():
    """The 8 paper-facing rows, including the n=6 fields."""
    assert _check_paper_digest(Checks()) == 90


def test_arm_to_directory_mapping_goes_through_arms_json():
    """joint_l1_* vs jointl1_*: two different runs, never derived from the label."""
    assert _check_arm_directory_mapping(Checks()) == 12


def test_error_bar_is_spread_not_bias():
    """The +- is the rescaled pre-ensemble spread; the bias is reported separately."""
    assert _check_bar_is_spread_not_bias(Checks()) == 49


def test_total_coverage():
    """The suite performed exactly the number of comparisons it claims.

    The guard against a vacuous pass: a truncated cross-check file, a dropped
    arm, or a field that stopped being compared all move this number.
    """
    counts = {}
    total = 0
    for fn, expected in _CHECKS:
        n = fn(Checks())
        counts[fn.__name__] = (n, expected)
        assert n == expected, f"{fn.__name__} performed {n} checks, expected {expected}"
        total += n
    assert total == EXPECTED_TOTAL_CHECKS, (
        f"total checks {total} != {EXPECTED_TOTAL_CHECKS}; per-check: {counts}"
    )
    assert total > 900, "suspiciously few comparisons for a 12-arm, 60-file bundle"


if __name__ == "__main__":  # pragma: no cover - plain-script fallback, see _runner.py
    import sys

    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import _runner

    raise SystemExit(_runner.main(globals()))
