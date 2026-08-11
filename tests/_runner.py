"""A minimal pytest stand-in, so every test module also runs as a plain script.

The supported way to run this suite is pytest (see ``tests/README.md``). But a
test suite you cannot run is a test suite nobody trusts, and pytest is not always
installed -- on a compute-cluster login node, inside a bare reproduction
environment, or in the conda env that ships the science dependencies but no dev
extras. So each test module ends with::

    if __name__ == "__main__":
        import _runner
        raise SystemExit(_runner.main(globals()))

:func:`main` collects the ``test_*`` callables from the module namespace,
supplies the only two pytest fixtures this suite uses (``tmp_path`` and
``monkeypatch``) from stdlib equivalents, runs them and prints a pytest-shaped
summary.

This implements exactly what this suite needs and nothing else. It is not a
pytest replacement, and no test should be written against features it lacks: if
a test needs more than ``tmp_path`` and ``monkeypatch``, make it a pytest-only
test and mark it, rather than growing this file.
"""

from __future__ import annotations

import inspect
import os
import shutil
import sys
import tempfile
import traceback
from pathlib import Path

__all__ = ["main", "MiniMonkeyPatch"]


_NOTSET = object()


class MiniMonkeyPatch:
    """The subset of ``pytest``'s ``monkeypatch`` fixture used by this suite.

    Records every mutation and reverses it in :meth:`undo`, in reverse order, so
    a failing test cannot leak environment changes into the next one.
    """

    def __init__(self) -> None:
        self._undo: list = []

    def setenv(self, name: str, value: str) -> None:
        self._undo.append(("env", name, os.environ.get(name, _NOTSET)))
        os.environ[name] = str(value)

    def delenv(self, name: str, raising: bool = True) -> None:
        if name not in os.environ:
            if raising:
                raise KeyError(name)
            return
        self._undo.append(("env", name, os.environ[name]))
        del os.environ[name]

    def setattr(self, target, name: str, value) -> None:
        self._undo.append(("attr", target, name, getattr(target, name, _NOTSET)))
        setattr(target, name, value)

    def chdir(self, path) -> None:
        self._undo.append(("cwd", os.getcwd()))
        os.chdir(str(path))

    def undo(self) -> None:
        while self._undo:
            record = self._undo.pop()
            kind = record[0]
            if kind == "env":
                _, name, old = record
                if old is _NOTSET:
                    os.environ.pop(name, None)
                else:
                    os.environ[name] = old
            elif kind == "attr":
                _, target, name, old = record
                if old is _NOTSET:
                    delattr(target, name)
                else:
                    setattr(target, name, old)
            elif kind == "cwd":
                # A test may have chdir'd into a directory that a later cleanup
                # (or the test itself) removed; restoring it is then impossible
                # and irrelevant -- the next record is the older, outer cwd.
                if os.path.isdir(record[1]):
                    os.chdir(record[1])


def _make_fixtures(params, undos, removals):
    """Build the argument list for *params*.

    Monkeypatch reversals go on *undos* and temporary trees on *removals*; the
    caller runs every undo before any removal, so restoring a working directory
    cannot race a directory that is about to be deleted.
    """
    args = []
    for name in params:
        if name == "tmp_path":
            d = Path(tempfile.mkdtemp(prefix="nn_l1_sbi_test_"))
            removals.append(lambda d=d: shutil.rmtree(d, ignore_errors=True))
            args.append(d)
        elif name == "monkeypatch":
            mp = MiniMonkeyPatch()
            undos.append(mp.undo)
            args.append(mp)
        else:  # pragma: no cover - guards against a test this runner cannot serve
            raise RuntimeError(
                f"tests/_runner.py cannot supply the fixture {name!r}. "
                f"Run this file under pytest instead."
            )
    return args


def main(namespace: dict) -> int:
    """Run every ``test_*`` callable in *namespace*. Return a process exit code."""
    tests = [
        (name, obj)
        for name, obj in sorted(namespace.items())
        if name.startswith("test_") and callable(obj)
    ]
    if not tests:
        print("no tests found -- did the module namespace get passed in?")
        return 2

    module = namespace.get("__file__", "<module>")
    print(f"running {len(tests)} tests from {Path(module).name} "
          f"(plain-script mode; pytest not required)")

    failures = []
    for name, fn in tests:
        undos: list = []
        removals: list = []
        try:
            args = _make_fixtures(list(inspect.signature(fn).parameters), undos, removals)
            fn(*args)
        except BaseException:  # noqa: BLE001 - report anything a test raises
            failures.append((name, traceback.format_exc()))
            print(f"  FAIL  {name}")
        else:
            print(f"  ok    {name}")
        finally:
            for cleanup in [*reversed(undos), *reversed(removals)]:
                cleanup()

    print()
    for name, tb in failures:
        print(f"===== FAILED {name} =====")
        print(tb)
    n_ok = len(tests) - len(failures)
    print(f"{n_ok} passed, {len(failures)} failed")
    return 1 if failures else 0


if __name__ == "__main__":  # pragma: no cover
    print(__doc__)
    sys.exit(0)
