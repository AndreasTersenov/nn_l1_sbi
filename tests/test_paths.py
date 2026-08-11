"""``nn_l1_sbi.paths`` -- the one place any root in this package is resolved.

The module exists to enforce two properties that a released package must have and
that the original research scripts did not:

1. Nothing is written inside the installation directory. Output roots come from
   the *current working directory* or an environment variable, never from
   ``Path(__file__).parent``.
2. No home directory is hard-coded. Every root is an environment variable with a
   documented, portable fallback -- or, where no sensible fallback exists
   (``NN_L1_SBI_COSMOGRID``), a clear error rather than a guess.

These tests pin the precedence rules and the failure messages. They are stdlib
only and touch nothing outside ``tmp_path``.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

# Run against the source clone whether or not the package is installed, and
# whether or not pytest (and therefore tests/conftest.py) is in play.
_SRC = str(Path(__file__).resolve().parents[1] / "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from nn_l1_sbi import paths  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]

ROOT_ENV_VARS = (
    "NN_L1_SBI_DATA",
    "NN_L1_SBI_OUT",
    "NN_L1_SBI_COSMOGRID",
    "NN_L1_SBI_TFDS",
    "TFDS_DATA_DIR",
)


def _isolate(monkeypatch, tmp_path) -> Path:
    """Clear every root variable and point ``$HOME`` at a fresh directory.

    Returns the fake home. Nothing under it exists yet, which is what makes the
    "no accessor creates anything" assertions meaningful.
    """
    for var in ROOT_ENV_VARS:
        monkeypatch.delenv(var, raising=False)
    home = tmp_path / "home"
    home.mkdir(exist_ok=True)
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("USERPROFILE", str(home))  # Path.home() on Windows
    return home


def _raises(exc_type, fn, *args, **kwargs) -> str:
    """Call *fn*, require it to raise *exc_type*, return the message.

    Written by hand rather than with ``pytest.raises`` so that this module also
    runs under ``tests/_runner.py`` when pytest is not installed.
    """
    try:
        result = fn(*args, **kwargs)
    except exc_type as exc:
        return str(exc)
    raise AssertionError(
        f"{getattr(fn, '__name__', fn)} returned {result!r}; expected {exc_type.__name__}"
    )


# ---------------------------------------------------------------------------
# data_root(): env var > repo-local bundle > ~/.cache
# ---------------------------------------------------------------------------
def test_data_root_env_var_wins(monkeypatch, tmp_path):
    _isolate(monkeypatch, tmp_path)
    bundle = tmp_path / "zenodo_bundle"
    monkeypatch.setenv("NN_L1_SBI_DATA", str(bundle))
    assert paths.data_root() == bundle.resolve()


def test_data_root_env_var_expands_tilde(monkeypatch, tmp_path):
    home = _isolate(monkeypatch, tmp_path)
    monkeypatch.setenv("NN_L1_SBI_DATA", "~/bundle")
    assert paths.data_root() == (home / "bundle").resolve()


def test_data_root_empty_env_var_counts_as_unset(monkeypatch, tmp_path):
    """``NN_L1_SBI_DATA=`` must fall back, not resolve to the current directory."""
    _isolate(monkeypatch, tmp_path)
    monkeypatch.setenv("NN_L1_SBI_DATA", "")
    monkeypatch.chdir(tmp_path)
    assert paths.data_root() != Path.cwd()
    assert paths.data_root() == _expected_data_fallback(tmp_path / "home")


def test_data_root_falls_back_when_env_unset(monkeypatch, tmp_path):
    """With no env var: the repo-local bundle if there is one, else ``~/.cache``."""
    home = _isolate(monkeypatch, tmp_path)
    got = paths.data_root()
    assert got == _expected_data_fallback(home)
    # The fallback is a *location*, not a side effect: nothing is created.
    assert not (home / ".cache").exists()


def _expected_data_fallback(home: Path) -> Path:
    """The documented fallback chain, evaluated for a given ``$HOME``.

    ``data/zenodo/`` is where this repository unpacks the Zenodo bundle (it is
    listed in ``.gitignore``). When it is present in a clone it takes precedence
    over the user-level cache; when it is not, the cache is the only fallback
    left. Written as a chain so the test states the whole precedence rule rather
    than whichever arm of it happens to apply on this machine.
    """
    repo = paths.repo_root()
    if repo is not None and (repo / "data" / "zenodo").is_dir():
        return repo / "data" / "zenodo"
    return home / ".cache" / "nn_l1_sbi" / "data"


# ---------------------------------------------------------------------------
# output_root(): env var > $CWD/nn_l1_sbi_out, never the package directory
# ---------------------------------------------------------------------------
def test_output_root_env_var_wins(monkeypatch, tmp_path):
    _isolate(monkeypatch, tmp_path)
    out = tmp_path / "results"
    monkeypatch.setenv("NN_L1_SBI_OUT", str(out))
    monkeypatch.chdir(tmp_path)
    assert paths.output_root() == out.resolve()


def test_output_root_is_cwd_relative_not_package_relative(monkeypatch, tmp_path):
    """Two different working directories give two different output roots."""
    _isolate(monkeypatch, tmp_path)
    a, b = tmp_path / "run_a", tmp_path / "run_b"
    a.mkdir()
    b.mkdir()

    monkeypatch.chdir(a)
    root_a = paths.output_root()
    monkeypatch.chdir(b)
    root_b = paths.output_root()

    assert root_a != root_b, "output_root() ignored the working directory"
    assert root_a.name == root_b.name == "nn_l1_sbi_out"
    assert root_a.parent.resolve() == a.resolve()
    assert root_b.parent.resolve() == b.resolve()

    # Rule 1 of the module docstring: never inside the installation.
    package_dir = Path(paths.__file__).resolve().parent
    for root in (root_a, root_b):
        assert package_dir not in root.resolve().parents
        assert root.resolve() != package_dir


def test_output_root_does_not_create_anything(monkeypatch, tmp_path):
    _isolate(monkeypatch, tmp_path)
    monkeypatch.chdir(tmp_path)
    root = paths.output_root()
    assert not root.exists()
    assert paths.save_params_dir() == root / "save_params"
    assert not paths.save_params_dir().exists()


def test_ensure_output_dir_is_the_only_mkdir(monkeypatch, tmp_path):
    _isolate(monkeypatch, tmp_path)
    monkeypatch.setenv("NN_L1_SBI_OUT", str(tmp_path / "out"))
    made = paths.ensure_output_dir("figures", "corner")
    assert made.is_dir()
    assert made == (tmp_path / "out" / "figures" / "corner").resolve()
    assert paths.ensure_output_dir("figures", "corner") == made  # idempotent


# ---------------------------------------------------------------------------
# cosmogrid_root(): no default, so it must fail loudly and actionably
# ---------------------------------------------------------------------------
def test_cosmogrid_root_raises_clear_runtime_error(monkeypatch, tmp_path):
    _isolate(monkeypatch, tmp_path)
    msg = _raises(RuntimeError, paths.cosmogrid_root)
    assert "NN_L1_SBI_COSMOGRID" in msg, msg
    assert "export" in msg, f"the error should tell the user how to fix it: {msg}"
    assert "not set" in msg, msg


def test_cosmogrid_root_empty_env_var_still_raises(monkeypatch, tmp_path):
    _isolate(monkeypatch, tmp_path)
    monkeypatch.setenv("NN_L1_SBI_COSMOGRID", "")
    _raises(RuntimeError, paths.cosmogrid_root)


def test_cosmogrid_root_and_metainfo_when_set(monkeypatch, tmp_path):
    _isolate(monkeypatch, tmp_path)
    grid = tmp_path / "CosmoGridV1"
    monkeypatch.setenv("NN_L1_SBI_COSMOGRID", str(grid))
    assert paths.cosmogrid_root() == grid.resolve()
    assert paths.cosmogrid_metainfo() == grid.resolve() / "CosmoGridV1_metainfo.h5"


def test_cosmogrid_metainfo_propagates_the_unset_error(monkeypatch, tmp_path):
    _isolate(monkeypatch, tmp_path)
    msg = _raises(RuntimeError, paths.cosmogrid_metainfo)
    assert "NN_L1_SBI_COSMOGRID" in msg, msg


# ---------------------------------------------------------------------------
# tfds_dir(): NN_L1_SBI_TFDS > TFDS_DATA_DIR > ~/tensorflow_datasets
# ---------------------------------------------------------------------------
def test_tfds_dir_precedence(monkeypatch, tmp_path):
    home = _isolate(monkeypatch, tmp_path)
    assert paths.tfds_dir() == home / "tensorflow_datasets"

    monkeypatch.setenv("TFDS_DATA_DIR", str(tmp_path / "tfds_generic"))
    assert paths.tfds_dir() == (tmp_path / "tfds_generic").resolve()

    monkeypatch.setenv("NN_L1_SBI_TFDS", str(tmp_path / "tfds_ours"))
    assert paths.tfds_dir() == (tmp_path / "tfds_ours").resolve()


# ---------------------------------------------------------------------------
# repo_root() / table1_dir(): finding the committed bundle
# ---------------------------------------------------------------------------
def test_repo_root_finds_the_clone():
    root = paths.repo_root()
    assert root == REPO_ROOT
    assert (root / "pyproject.toml").is_file()
    assert (root / "src").is_dir()


def test_table1_dir_is_the_committed_bundle(monkeypatch, tmp_path):
    """The Table-1 bundle is found from a clone without any env var set."""
    _isolate(monkeypatch, tmp_path)
    monkeypatch.chdir(tmp_path)  # and without depending on the working directory
    bundle = paths.table1_dir()
    assert bundle == REPO_ROOT / "data" / "table1"
    assert (bundle / "arms.json").is_file()
    assert len(list(bundle.glob("sweeps/*/median_summary.json"))) == 54


# ---------------------------------------------------------------------------
# require(): the actionable missing-input message
# ---------------------------------------------------------------------------
def test_require_returns_an_existing_path(tmp_path):
    target = tmp_path / "present.npz"
    target.write_bytes(b"")
    assert paths.require(target) == target
    assert paths.require(str(target)) == target


def test_require_raises_actionable_error(tmp_path):
    missing = tmp_path / "absent.npz"
    msg = _raises(
        FileNotFoundError, paths.require, missing, produced_by="population_sweep.py"
    )
    assert str(missing) in msg
    assert "population_sweep.py" in msg
    assert "NN_L1_SBI_DATA" in msg, f"should point at the data bundle: {msg}"


# ---------------------------------------------------------------------------
# Global invariant
# ---------------------------------------------------------------------------
def test_no_accessor_creates_anything(monkeypatch, tmp_path):
    """Only ``ensure_output_dir`` may touch the filesystem."""
    home = _isolate(monkeypatch, tmp_path)
    sandbox = tmp_path / "sandbox"
    sandbox.mkdir()
    monkeypatch.chdir(sandbox)
    monkeypatch.setenv("NN_L1_SBI_DATA", str(sandbox / "data"))
    monkeypatch.setenv("NN_L1_SBI_OUT", str(sandbox / "out"))
    monkeypatch.setenv("NN_L1_SBI_COSMOGRID", str(sandbox / "cosmogrid"))
    monkeypatch.setenv("NN_L1_SBI_TFDS", str(sandbox / "tfds"))

    for name in ("data_root", "output_root", "cosmogrid_root", "cosmogrid_metainfo",
                 "tfds_dir", "save_params_dir", "repo_root", "table1_dir"):
        getattr(paths, name)()

    assert list(sandbox.iterdir()) == [], f"an accessor created {list(sandbox.iterdir())}"
    assert list(home.iterdir()) == [], f"an accessor wrote into $HOME: {list(home.iterdir())}"


def test_public_api_is_complete():
    for name in paths.__all__:
        assert hasattr(paths, name), f"paths.__all__ advertises a missing name: {name}"
    assert "data_root" in paths.__all__ and "output_root" in paths.__all__


if __name__ == "__main__":  # pragma: no cover - plain-script fallback, see _runner.py
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import _runner

    raise SystemExit(_runner.main(globals()))
