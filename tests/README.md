# `tests/` — the suite that runs anywhere

Everything here is **stdlib (+ NumPy at most)**: no install, no network, no GPU, no
Zenodo download, a few seconds end to end. That is deliberate — a reproduction
repository whose tests need a GPU is a repository whose tests nobody runs.

```bash
# from the repository root
PYTHONPATH=src pytest tests/ -q

# no pytest? every module runs as a plain script
PYTHONPATH=src python tests/test_table1.py
```

## What is covered

| file | what it pins |
|---|---|
| `test_table1.py` | **Table 1 of the paper**, reproduced from `data/table1/` and compared against all 12 published `central ± pm` pairs, the retrained centrals and marginals, and the four precomputed cross-check JSONs. |
| `test_imports.py` | `import nn_l1_sbi` and `nn_l1_sbi.paths` work with *only the standard library*; every light module imports; nothing is created on disk at import time. |
| `test_paths.py` | The precedence rules and error messages of `nn_l1_sbi.paths` — the single place any data or output root is resolved. |

`test_table1.py` re-implements the error-bar recipe from its written specification
(`data/table1/README.md` §6, `docs/NOTE_FOM_ERROR_BARS.md` §5–6) rather than importing
the notebook or `nn_l1_sbi.table1`, so the test and the artefact cannot agree by
construction. It uses `statistics` instead of NumPy for the same reason.

## Counted assertions

Assertions that iterate over a data file pass vacuously when that file is empty or
truncated — the loop body simply never runs, and the suite goes green having checked
nothing. Every comparison in `test_table1.py` therefore goes through a counter, and each
test asserts the exact number of comparisons it performed (946 in total) against a
literal derived from the documented bundle shape. Drop an arm, truncate a cross-check
file, or stop comparing a field, and the count moves and the suite fails.

## Light vs heavy

* **unmarked** — stdlib + NumPy only. This is what CI runs
  (`.github/workflows/ci.yml`), with `-m "not heavy"`, on Python 3.10–3.12, plus one
  job with no NumPy at all.
* **`@pytest.mark.heavy`** — needs JAX / TensorFlow / Torch / healpy, the Zenodo data
  bundle, or a GPU. Never run on CI; run by hand on a machine that has them.

Anything that cannot honestly run on a clean runner with `pip install pytest numpy`
must carry the marker. The marker is registered in `conftest.py`, and CI passes
`--strict-markers` so a typo fails loudly.

## `_runner.py`

A ~130-line pytest stand-in so each test module also runs as a plain script
(`python tests/test_paths.py`). It supplies the only two fixtures this suite uses —
`tmp_path` and `monkeypatch` — from stdlib equivalents. It exists because the cluster
environments this work runs on ship the science stack but not the dev extras, and CI
exercises it on every run so it cannot rot. It is not a pytest replacement: a test that
needs more than those two fixtures should be pytest-only.
