"""Import hygiene for the ``nn_l1_sbi`` package.

Two properties this release has to have, and that are easy to lose silently:

1. **``import nn_l1_sbi`` costs nothing.** The top-level package pulls in no
   submodule, so it imports with *only the standard library present* -- no JAX,
   TensorFlow, Torch or healpy. Someone who just wants ``nn_l1_sbi.paths`` (to
   find the data bundle, or to print Table 1) must not be forced to install a
   GPU stack. This is asserted the hard way: a child interpreter installs an
   import hook that refuses every non-stdlib module, and then imports the
   package.

2. **Nothing is created on disk at import time.** The original research scripts
   derived their output directories from ``Path(__file__).parent`` and made them
   eagerly. In a package that means writing into ``site-packages``. Every light
   module is therefore imported inside a sandbox -- fresh working directory,
   fresh ``$HOME``, output and data roots pointed inside it -- and the sandbox
   must be byte-for-byte empty afterwards.

Light vs heavy
--------------
Modules are classified by a static AST scan of their top-level imports, not by a
hand-maintained list: a module is *light* when every third-party package it
imports is in :data:`LIGHT_THIRD_PARTY` (numpy) and is actually importable here.
Everything else -- JAX, TensorFlow, Torch, healpy, matplotlib, wandb ... -- is
heavy and is only classified, never imported. So this file stays honest as the
package grows, and stays green on a CI runner that has nothing but pytest and
numpy.
"""

from __future__ import annotations

import ast
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC = REPO_ROOT / "src"
PKG = SRC / "nn_l1_sbi"

#: Third-party packages a "light" module is allowed to import.
LIGHT_THIRD_PARTY = frozenset({"numpy"})

#: Modules that must always be importable with nothing but the standard library.
STDLIB_ONLY_MODULES = ("nn_l1_sbi", "nn_l1_sbi.paths")

#: A few modules that are known-heavy. Asserted, so that a bug in the classifier
#: which marked everything light would be caught rather than celebrated.
KNOWN_HEAVY = (
    "nn_l1_sbi.flatsky.flatsky_cross",
    "nn_l1_sbi.cnn.npe_cnn_nbody_tomo",
    "nn_l1_sbi.inference.population_sweep",
    "nn_l1_sbi.data.tf_dataset_nbody_tomo",
)

#: Lower bound on the number of modules in the package. Guards against a scan
#: that silently found nothing and then "passed".
MIN_MODULES_SCANNED = 45
MIN_LIGHT_MODULES = 8


# ---------------------------------------------------------------------------
# Static scan
# ---------------------------------------------------------------------------
def _module_name(path: Path) -> str:
    rel = path.relative_to(SRC).with_suffix("")
    parts = list(rel.parts)
    if parts[-1] == "__init__":
        parts.pop()
    return ".".join(parts)


def _top_level_imports(path: Path) -> set[str]:
    """Third-party top-level packages imported at module scope (or anywhere)."""
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    roots: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                roots.add(alias.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom):
            if node.level == 0 and node.module:  # level > 0 is intra-package
                roots.add(node.module.split(".")[0])
    stdlib = set(sys.stdlib_module_names)
    return {r for r in roots if r not in stdlib and r != "nn_l1_sbi"}


def _importable(name: str) -> bool:
    try:
        __import__(name)
    except Exception:  # noqa: BLE001 - a broken dependency is "not importable"
        return False
    return True


def _scan() -> dict[str, dict]:
    """``module name -> {"deps": set, "light": bool}`` for every module in the package."""
    available = {dep: _importable(dep) for dep in LIGHT_THIRD_PARTY}
    out: dict[str, dict] = {}
    for path in sorted(PKG.rglob("*.py")):
        deps = _top_level_imports(path)
        light = deps <= LIGHT_THIRD_PARTY and all(available[d] for d in deps)
        out[_module_name(path)] = {"deps": sorted(deps), "light": light}
    return out


def _light_modules() -> list[str]:
    return [name for name, info in _scan().items() if info["light"]]


# ---------------------------------------------------------------------------
# Child-interpreter probes
# ---------------------------------------------------------------------------
_STDLIB_ONLY_PROBE = r"""
import sys

_ALLOWED_ROOTS = set(sys.stdlib_module_names) | {"nn_l1_sbi"}


class _RefuseThirdParty:
    '''Refuse any import whose root package is not stdlib and not nn_l1_sbi.'''

    def find_spec(self, fullname, path=None, target=None):
        root = fullname.split(".")[0]
        if root not in _ALLOWED_ROOTS and fullname not in sys.modules:
            raise ImportError(
                "blocked by the stdlib-only probe: " + fullname
            )
        return None  # defer to the real finders


sys.meta_path.insert(0, _RefuseThirdParty())

for name in sys.argv[1:]:
    __import__(name)

print("OK " + " ".join(sys.argv[1:]))
"""

_SANDBOX_PROBE = r"""
import importlib
import json
import sys
from pathlib import Path

sandbox = Path(sys.argv[1])
report = Path(sys.argv[2])
modules = sys.argv[3:]


def listing():
    return sorted(str(p.relative_to(sandbox)) for p in sandbox.rglob("*"))


before = listing()
errors = {}
imported = []
for name in modules:
    try:
        importlib.import_module(name)
    except BaseException as exc:  # noqa: BLE001 - report, do not raise
        errors[name] = "%s: %s" % (type(exc).__name__, exc)
    else:
        imported.append(name)
after = listing()

report.write_text(json.dumps({
    "imported": imported,
    "errors": errors,
    "created": [p for p in after if p not in before],
    "cwd": str(Path.cwd()),
    "home": str(Path.home()),
}))
"""


def _run_probe(source: str, args: list[str], *, cwd=None, env_extra=None):
    """Run *source* in a child interpreter that can import the package from ``src/``.

    ``SRC`` is prepended to ``PYTHONPATH`` as an absolute path, so the child does
    not have to be started from the repository root -- and it never is: a probe
    with no explicit *cwd* runs in a throwaway directory, so nothing it might
    leave behind (bytecode, a core file, an eagerly created output root) lands in
    the working tree.
    """
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(
        [str(SRC)] + ([env["PYTHONPATH"]] if env.get("PYTHONPATH") else [])
    )
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env.update(env_extra or {})
    with tempfile.TemporaryDirectory(prefix="nn_l1_sbi_probe_") as scratch:
        script = os.path.join(scratch, "probe.py")
        with open(script, "w", encoding="utf-8") as fh:
            fh.write(source)
        return subprocess.run(
            [sys.executable, script, *args],
            cwd=scratch if cwd is None else str(cwd),
            env=env,
            capture_output=True,
            text=True,
            timeout=300,
        )


def _sandbox_result(tmp_root: Path) -> dict:
    """Import every light module inside a pristine sandbox; return the child's report."""
    sandbox = tmp_root / "sandbox"
    home = sandbox / "home"
    home.mkdir(parents=True)
    report = tmp_root / "report.json"  # outside the sandbox, so it is not "created" in it
    modules = _light_modules()

    proc = _run_probe(
        _SANDBOX_PROBE,
        [str(sandbox), str(report), *modules],
        cwd=sandbox,
        env_extra={
            "HOME": str(home),
            "USERPROFILE": str(home),
            # Unset roots: any module that eagerly materialises one would then
            # create it inside the sandbox, where this test can see it.
            "NN_L1_SBI_OUT": "",
            "NN_L1_SBI_DATA": "",
            "NN_L1_SBI_COSMOGRID": "",
        },
    )
    assert proc.returncode == 0, (
        f"sandbox probe crashed (rc={proc.returncode})\n"
        f"stdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"
    )
    data = json.loads(report.read_text())
    data["_modules"] = modules
    # `home/` is ours; everything else under the sandbox was made by an import.
    data["created"] = [p for p in data["created"] if p != "home"]
    return data


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------
def test_package_imports_with_stdlib_only():
    """``import nn_l1_sbi`` and ``nn_l1_sbi.paths`` need nothing but the stdlib."""
    proc = _run_probe(_STDLIB_ONLY_PROBE, list(STDLIB_ONLY_MODULES))
    assert proc.returncode == 0, (
        "importing "
        + ", ".join(STDLIB_ONLY_MODULES)
        + " required a third-party package.\n"
        + f"stdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"
    )
    assert proc.stdout.startswith("OK "), proc.stdout


def test_stdlib_only_probe_actually_blocks():
    """The probe above is not vacuous: it really does refuse a third-party import."""
    proc = _run_probe(_STDLIB_ONLY_PROBE, ["nn_l1_sbi", "this_module_does_not_exist_xyz"])
    assert proc.returncode != 0, (
        "the stdlib-only probe let a non-stdlib import through, so "
        "test_package_imports_with_stdlib_only proves nothing"
    )
    assert "blocked by the stdlib-only probe" in proc.stderr, proc.stderr


def test_module_classification_is_sane():
    """The light/heavy split found the package, and did not mark everything light."""
    scanned = _scan()
    n_py = len(list(PKG.rglob("*.py")))
    assert len(scanned) == n_py, "a module was dropped between globbing and scanning"
    assert len(scanned) >= MIN_MODULES_SCANNED, (
        f"only {len(scanned)} modules found under {PKG}; the scan is not seeing the package"
    )

    light = {name for name, info in scanned.items() if info["light"]}
    assert len(light) >= MIN_LIGHT_MODULES, f"only {len(light)} light modules: {sorted(light)}"
    for name in STDLIB_ONLY_MODULES:
        assert name in light, f"{name} must be light, got deps {scanned[name]['deps']}"
    for name in ("nn_l1_sbi.cnn", "nn_l1_sbi.compression", "nn_l1_sbi.data",
                 "nn_l1_sbi.figures", "nn_l1_sbi.flatsky", "nn_l1_sbi.inference"):
        assert name in light, f"subpackage {name} must import without heavy dependencies"
    for name in KNOWN_HEAVY:
        assert name in scanned, f"{name} disappeared from the package"
        assert not scanned[name]["light"], (
            f"{name} was classified light, but it imports {scanned[name]['deps']}; "
            "the classifier is broken"
        )


def test_light_modules_import_cleanly(tmp_path):
    """Every light module imports, in a sandbox with no data roots configured."""
    result = _sandbox_result(tmp_path)
    assert result["errors"] == {}, f"light modules failed to import: {result['errors']}"
    assert sorted(result["imported"]) == sorted(result["_modules"])
    assert len(result["imported"]) >= MIN_LIGHT_MODULES


def test_no_directory_created_at_import(tmp_path):
    """Importing the package creates nothing on disk -- no output dir, no cache."""
    result = _sandbox_result(tmp_path)
    assert result["created"] == [], (
        "importing the package created files/directories: "
        f"{result['created']} (under {tmp_path / 'sandbox'}). "
        "Nothing may be made at import time; use paths.ensure_output_dir() at call time."
    )
    # The sandbox really was in force, otherwise the assertion above is empty.
    assert Path(result["cwd"]).resolve() == (tmp_path / "sandbox").resolve()
    assert Path(result["home"]).resolve() == (tmp_path / "sandbox" / "home").resolve()


if __name__ == "__main__":  # pragma: no cover - plain-script fallback, see _runner.py
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import _runner

    raise SystemExit(_runner.main(globals()))
