"""Central path resolution for :mod:`nn_l1_sbi`.

Every input root and output root used anywhere in this package resolves through
this module. There are two rules, and both matter for a released package:

1. **Nothing is written inside the installation directory.** The original
   research scripts lived in a flat directory and derived their output roots
   from ``Path(__file__).parent``. Once the code is a real package that idiom
   points at ``site-packages/nn_l1_sbi/...``, which is read-only on a shared
   machine and simply wrong everywhere else. Outputs go under
   :func:`output_root` instead.

2. **No user's home directory is hard-coded.** Each root is taken from an
   environment variable with a documented, portable fallback.

Environment variables
---------------------

======================  ===================================================
``NN_L1_SBI_DATA``      Root of the unpacked Zenodo bundle (read-only inputs:
                        sweeps, caches, maps). Resolved in this order, and
                        this module is the single source of truth for it:

                        1. ``$NN_L1_SBI_DATA``, if set;
                        2. ``<repo>/data/zenodo``, if that directory exists.
                           Only a source clone has a ``<repo>``; this is the
                           location ``README.md``, ``data/ZENODO_MANIFEST.md``
                           and ``docs/FIGURE_MAP.md`` tell the reader to
                           unpack the bundle into, and it is gitignored;
                        3. ``~/.cache/nn_l1_sbi/data``, where the notebook's
                           download cell writes, so a reader who runs the
                           notebook top to bottom sets nothing.

                        ``scripts/slurm/env.sh`` mirrors exactly these three
                        steps in shell, so a SLURM job and a hand-run driver
                        resolve the same root.
``NN_L1_SBI_OUT``       Where results, figures and caches are written.
                        Default: ``./nn_l1_sbi_out`` in the current directory.
``NN_L1_SBI_COSMOGRID`` Root of the raw CosmoGridV1 simulation tree. Only
                        needed to rebuild datasets from scratch; there is no
                        sensible default, so it raises if unset.
``NN_L1_SBI_TFDS``      TensorFlow Datasets directory. Default:
                        ``$TFDS_DATA_DIR`` if set, else ``~/tensorflow_datasets``.
======================  ===================================================

All accessors return :class:`pathlib.Path` and never create directories as a
side effect, except :func:`ensure_output_dir`, which does so explicitly.
"""

from __future__ import annotations

import os
from pathlib import Path

__all__ = [
    "data_root",
    "output_root",
    "ensure_output_dir",
    "cosmogrid_root",
    "cosmogrid_metainfo",
    "tfds_dir",
    "save_params_dir",
    "repo_root",
    "table1_dir",
    "require",
]


def _env_path(var: str, default: Path | None) -> Path:
    """Return ``$var`` as a Path, else *default*, else raise."""
    raw = os.environ.get(var)
    if raw:
        return Path(raw).expanduser().resolve()
    if default is None:
        raise RuntimeError(
            f"{var} is not set and has no default.\n"
            f"Set it to the appropriate directory, e.g.:\n"
            f"    export {var}=/path/to/data"
        )
    return Path(default).expanduser()


def data_root() -> Path:
    """Root of the unpacked Zenodo data bundle (read-only inputs).

    Three steps, in order (see the module docstring):

    1. ``$NN_L1_SBI_DATA`` if set;
    2. ``<repo>/data/zenodo`` if that directory exists -- the documented
       install location, so ``mv nn_l1_sbi_data_v1 <repo>/data/zenodo`` works
       with no environment variable at all;
    3. ``~/.cache/nn_l1_sbi/data``, the notebook's download target.

    Step 2 needs a source clone; :func:`repo_root` returns ``None`` for an
    installed package, and the step is skipped.
    """
    raw = os.environ.get("NN_L1_SBI_DATA")
    if raw:
        return Path(raw).expanduser().resolve()
    root = repo_root()
    if root is not None:
        bundled = root / "data" / "zenodo"
        if bundled.is_dir():
            return bundled
    return Path.home() / ".cache" / "nn_l1_sbi" / "data"


def output_root() -> Path:
    """Root for everything this package writes.

    Deliberately relative to the *current working directory* rather than to the
    package, so that running a driver never writes into ``site-packages``.
    """
    return _env_path("NN_L1_SBI_OUT", Path.cwd() / "nn_l1_sbi_out")


def ensure_output_dir(*parts: str) -> Path:
    """``output_root()/parts...``, created if missing. The only mkdir here."""
    p = output_root().joinpath(*parts)
    p.mkdir(parents=True, exist_ok=True)
    return p


def cosmogrid_root() -> Path:
    """Root of the raw CosmoGridV1 tree. Raises if ``NN_L1_SBI_COSMOGRID`` is unset.

    On Jean-Zay (IDRIS) this is ``/lustre/fsmisc/dataset/CosmoGridV1``.
    """
    return _env_path("NN_L1_SBI_COSMOGRID", None)


def cosmogrid_metainfo() -> Path:
    """``CosmoGridV1_metainfo.h5`` inside :func:`cosmogrid_root`."""
    return cosmogrid_root() / "CosmoGridV1_metainfo.h5"


def tfds_dir() -> Path:
    """TensorFlow Datasets directory."""
    tfds = os.environ.get("NN_L1_SBI_TFDS") or os.environ.get("TFDS_DATA_DIR")
    if tfds:
        return Path(tfds).expanduser().resolve()
    return Path.home() / "tensorflow_datasets"


def save_params_dir() -> Path:
    """Where trained network parameters are written."""
    return output_root() / "save_params"


def repo_root() -> Path | None:
    """Repository root when running from a source clone, else ``None``.

    Used only to locate the committed ``data/table1`` bundle. Returns ``None``
    for an installed package, where no clone exists.
    """
    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / "pyproject.toml").exists() and (parent / "src").is_dir():
            return parent
    return None


def table1_dir() -> Path:
    """The committed Table-1 bundle (``data/table1``).

    Prefers the source clone; falls back to ``data_root()/table1`` for an
    installed package pointed at an unpacked Zenodo bundle.
    """
    root = repo_root()
    if root is not None and (root / "data" / "table1").is_dir():
        return root / "data" / "table1"
    return data_root() / "table1"


def require(path, produced_by: str | None = None) -> Path:
    """Return *path* if it exists, else raise with an actionable message.

    Parameters
    ----------
    path
        The file or directory that must exist.
    produced_by
        Name of the script or job that generates it, quoted back to the user.
    """
    p = Path(path)
    if p.exists():
        return p
    msg = [f"Required input not found:\n    {p}"]
    if produced_by:
        msg.append(f"It is produced by: {produced_by}")
    msg.append(
        "If you have the Zenodo bundle, either point NN_L1_SBI_DATA at its unpacked root:\n"
        "    export NN_L1_SBI_DATA=/path/to/nn_l1_sbi_data\n"
        "or move it to where a source clone looks by default:\n"
        "    mv nn_l1_sbi_data /path/to/repo/data/zenodo\n"
        "See data/ZENODO_MANIFEST.md for the bundle layout."
    )
    raise FileNotFoundError("\n".join(msg))
