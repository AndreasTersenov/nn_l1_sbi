#!/usr/bin/env python3
"""Derive the committed ``data/table1/arms.json`` from ``data/table1/rows_table1_v2.json``.

``rows_table1_v2.json`` is the *provenance record*: the exact row spec that was handed to
:mod:`nn_l1_sbi.table1.make_table`, with absolute paths to the 153 MB per-arm
``per_patch_metrics.npz`` files. It is shipped unmodified so the derivation below is
auditable.

``arms.json`` is its portable derivative, and the only one the notebook reads. Two
substitutions, nothing else — ``published`` and ``quoted`` are copied through untouched,
and so is the key order:

1. **absolute → repo-relative**: ``<anything>/sweeps/<dir>/…`` → ``sweeps/<dir>/…``,
   relative to ``data/table1/``.
2. **npz → digest**: ``per_patch_metrics.npz`` → ``median_summary.json``.

Substitution 2 is a genuine substitution of one file for another, and it is safe because
the digest is exact, not a re-derivation::

    median_summary.json["fom3"]     == float(np.median(npz["fom3"]))
    median_summary.json["sigma_Om"] == float(np.median(npz["sigma"], axis=0)[0])
    median_summary.json["sigma_s8"] == float(np.median(npz["sigma"], axis=0)[1])
    median_summary.json["sigma_w0"] == float(np.median(npz["sigma"], axis=0)[2])

with ``param_order = (Omega_m, sigma_8, w_0)``; both are written in the same pass by
:mod:`nn_l1_sbi.inference.population_sweep`. So every Table-1 central and marginal is
recoverable from the 287-byte JSON, and the 153 MB array is needed only to *re*-run the
block bootstrap.

Usage
  # regenerate arms.json and byte-compare against the committed copy
  python -m nn_l1_sbi.table1.derive_arms

  # write it somewhere specific
  python -m nn_l1_sbi.table1.derive_arms --out /path/to/arms.json

Exit status is 0 when the regenerated file is byte-identical to the committed one, 1 when
it differs (and the differing arms are listed).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path, PurePosixPath

from nn_l1_sbi.paths import ensure_output_dir, table1_dir

#: Directory under ``data/table1/`` holding the 54 per-sweep digests.
SWEEPS_DIR = "sweeps"
#: The digest that replaces ``per_patch_metrics.npz``.
SUMMARY_NAME = "median_summary.json"
#: JSON dump options that reproduce the committed ``arms.json`` byte for byte.
DUMP_KWARGS = {"indent": 1}


def relativise(path: str) -> str:
    """``.../sweeps/<dir>/per_patch_metrics.npz`` -> ``sweeps/<dir>/median_summary.json``.

    Raises ``ValueError`` if *path* is not of that shape, rather than silently emitting a
    relative path that points nowhere.
    """
    p = PurePosixPath(str(path))
    if p.parent.name == "" or p.parent.parent.name != SWEEPS_DIR:
        raise ValueError(
            f"not a <...>/{SWEEPS_DIR}/<dir>/<file> path, refusing to relativise: {path}")
    return str(PurePosixPath(SWEEPS_DIR, p.parent.name, SUMMARY_NAME))


def derive(rows: dict) -> dict:
    """Row spec (absolute, ``.npz``) -> arm map (relative, digest). Key order preserved."""
    out = {}
    for label, cfg in rows.items():
        arm = dict(cfg)                       # keeps singles / ensemble / published / quoted order
        arm["singles"] = {tag: relativise(p) for tag, p in cfg["singles"].items()}
        arm["ensemble"] = relativise(cfg["ensemble"])
        out[label] = arm
    return out


def dumps(arms: dict) -> str:
    """Serialise exactly as the committed ``arms.json`` is serialised (indent=1, trailing \\n)."""
    return json.dumps(arms, **DUMP_KWARGS) + "\n"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--rows", type=Path, default=table1_dir() / "rows_table1_v2.json",
                    help="row spec to derive from (default: the committed rows_table1_v2.json)")
    ap.add_argument("--out", type=Path, default=None,
                    help="where to write (default: <output_root>/table1/arms.json)")
    ap.add_argument("--compare", type=Path, default=table1_dir() / "arms.json",
                    help="byte-compare the result against this file (default: the committed arms.json)")
    ap.add_argument("--no-compare", dest="compare", action="store_const", const=None,
                    help="skip the byte comparison")
    a = ap.parse_args(argv)

    rows = json.loads(Path(a.rows).read_text())
    text = dumps(derive(rows))

    out = Path(a.out) if a.out else ensure_output_dir("table1") / "arms.json"
    out.write_text(text)
    print(f"wrote {out}  ({len(text)} B, {len(rows)} arms)")

    if a.compare is None:
        return 0
    ref = Path(a.compare)
    if not ref.exists():
        print(f"cannot compare: {ref} does not exist")
        return 1
    ref_text = ref.read_text()
    if ref_text == text:
        print(f"byte-identical to {ref}")
        return 0
    print(f"DIFFERS from {ref}")
    got, want = json.loads(text), json.loads(ref_text)
    for label in sorted(set(got) | set(want)):
        if got.get(label) != want.get(label):
            print(f"  {label}:\n    derived  = {got.get(label)}\n    committed= {want.get(label)}")
    return 1


if __name__ == "__main__":
    sys.exit(main())
