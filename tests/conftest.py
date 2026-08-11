"""Pytest configuration for the nn_l1_sbi test suite.

Two jobs:

1. Put ``src/`` on ``sys.path`` so the suite runs against a source clone without
   an install and without the caller having to export ``PYTHONPATH=src``. An
   installed copy of the package still wins if one is already imported.
2. Register the ``heavy`` marker. Tests in this suite are split in two:

   * unmarked -- stdlib (+ numpy) only, no network, no GPU, sub-second. These are
     what CI runs, with ``-m "not heavy"``.
   * ``@pytest.mark.heavy`` -- needs JAX / TensorFlow / Torch / healpy, the
     Zenodo data bundle, or a GPU. Never run on CI.

   Anything that cannot honestly run on a clean runner with ``pip install pytest
   numpy`` must carry the marker.
"""

from __future__ import annotations

import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parents[1] / "src"
if _SRC.is_dir() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))


def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "heavy: needs GPU / JAX / TensorFlow / Torch / the Zenodo bundle; "
        "excluded from CI via -m 'not heavy'",
    )
