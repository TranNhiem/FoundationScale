"""Packaging control for defect #219: the decision plane must decide when installed.

Before the fix, ``foundationscale.gates.adjudication`` imported ``derive_declared``
and ``run_alias_control`` from ``tools/real_checkpoint_probe.py`` behind a
try/except ImportError ladder. ``tools/`` is not distributed
(``[tool.setuptools.packages.find] where = ["src"]``), so on a clean install the
ladder always fell through: the import *succeeded*, the helpers stayed ``None``,
and every call into ``derive_declared_block`` refused. A test that merely checks
the import would have been GREEN against the broken layout, which is exactly the
shape of miss this file exists to close: both legs below assert on a real
outcome of the decision path, not on the import statement.

MUST_PASS: with ONLY ``src/`` importable (repo root and ``tools/`` provably off
sys.path), the decision path must actually derive a declared block.

MUST_FIRE: the same subprocess, run against a doctored copy of the package
whose moved helpers are unavailable, must fail. A control that cannot fail
measures nothing.

The two legs above put ``src/`` on sys.path, which SIMULATES an install. The
third leg builds a real wheel and decides from its extracted contents, where
``tools/`` is absent from disk rather than merely absent from sys.path. The
distinction is the whole of #219: the pre-fix layout also had ``tools/`` off
sys.path in plenty of situations and still imported fine -- it bound None and
refused later. Only an artifact whose namelist provably contains no ``tools/``
member can settle what a consumer actually receives.

No skip anywhere: this repo runs CI with FS_FORBID_SKIPS=1, where a skip is a
failure. No network, no GPU, no torch -- the gates plane is stdlib-only by
design, so an isolated interpreter suffices. The wheel leg builds with
--no-build-isolation first precisely so it needs no network either, and FAILS
rather than skips if no build route works: "could not measure" is a state this
repo declares out loud, never one it reports as green.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC = REPO_ROOT / "src"

# Executed by an isolated interpreter (-I: no PYTHONPATH, no user site, no cwd
# prepend). The child itself inserts src/ and then PROVES the repo root and
# tools/ are absent from sys.path, so the positive control measures the same
# sys.path shape as a clean pip install rather than asserting a hope.
_CHILD = """
import sys
from pathlib import Path

src = Path(sys.argv[1]).resolve()
sys.path.insert(0, str(src))
repo = src.parent
leaked = [
    p for p in sys.path
    if p and Path(p).resolve() in (repo, repo / "tools")
]
assert not leaked, f"repo root or tools/ leaked onto sys.path: {leaked}"

# An absence is evidence only if the same call can see a presence. Without
# this line the three checks below would also "pass" against an interpreter
# where find_spec were broken outright.
import importlib.util

assert importlib.util.find_spec("foundationscale.gates") is not None, (
    "find_spec cannot see the installed package, so the absences below "
    "would be vacuous"
)
# Probed as TOP-LEVEL names on purpose. find_spec("tools.real_checkpoint_probe")
# imports the parent package first and RAISES ModuleNotFoundError when it is
# missing, rather than returning None -- so the dotted form cannot express the
# very state being tested, and a check that cannot express its own state is not
# a check. An absent parent makes the dotted children unreachable anyway, which
# is the stronger claim.
reachable = [
    n for n in ("tools", "real_checkpoint_probe", "live_save_gate")
    if importlib.util.find_spec(n) is not None
]
assert not reachable, (
    f"tools/ scripts are importable from here: {reachable} -- the derivation "
    "below would be measuring the checkout, not the installed package"
)

from foundationscale.gates import adjudication, probe

# The decision layer's derive and control steps must BE the packaged helpers,
# not a None left behind by a failed optional import. Pre-#219, adjudication
# imported with only src/ on sys.path just fine -- and bound None. The identity
# checks are what separate "the import succeeded" from "the decision path can
# actually decide".
assert adjudication._probe_derive_declared is probe.derive_declared
assert adjudication._probe_alias_control is probe.run_alias_control

# A real derivation, not a typecheck: an affirmative dense statement
# (text_config.enable_moe_block = false) corroborated by a zero expert census
# must mint the corroborated 0, and a stated num_moe_layers must come back
# stated. Pre-fix this attribute was None here, and calling it fails; the
# refusal branch upstream of it existed for exactly that case.
declared = adjudication._probe_derive_declared(
    {"text_config": {"enable_moe_block": False, "num_moe_layers": 48}},
    expert_family_census=0,
)
assert declared["num_experts"] == 0, declared["basis"]["num_experts"]
assert declared["num_moe_layers"] == 48, declared["basis"]["num_moe_layers"]
assert declared["declared_fqns"] is None
assert declared["expected_expert_bytes"] is None

# Same shape the original defect bit on: config states nothing, census was
# measured, and the honest answer is UNKNOWN (None), not a laundered zero.
declared_silent = adjudication._probe_derive_declared(
    {"model_type": "unrelated"},
    expert_family_census=0,
)
assert declared_silent["num_experts"] is None, declared_silent["basis"]["num_experts"]

print("PROBE-PACKAGING-OK")
"""


def _run_child(src_dir: Path) -> subprocess.CompletedProcess[bytes]:
    env = dict(os.environ)
    env.pop("PYTHONPATH", None)
    return subprocess.run(
        [sys.executable, "-I", "-c", _CHILD, str(src_dir)],
        capture_output=True,
        env=env,
        timeout=120,
        check=False,
    )


def test_must_pass_decision_path_derives_a_declared_block_with_only_src_importable() -> None:
    """The #219 control: installed (src/ only), the decision path must work."""
    result = _run_child(SRC)
    assert result.returncode == 0, (
        "with ONLY src/ on sys.path the decision path must derive a declared "
        "block; before #219 the helpers came from tools/, stayed unbound, and "
        "every call refused\n"
        f"stdout:\n{result.stdout.decode(errors='replace')}\n"
        f"stderr:\n{result.stderr.decode(errors='replace')}"
    )
    assert b"PROBE-PACKAGING-OK" in result.stdout


def test_must_fire_detector_goes_red_when_the_moved_helpers_are_unavailable(
    tmp_path: Path,
) -> None:
    """The control behind the control: sabotage the helpers, demand red.

    The package is copied into tmp_path and ``derive_declared`` is renamed,
    which is the installed-defect's shape (helper unavailable) reconstructed
    in-package. Post-#219 the import is in-package, so the package must fail
    to load -- fail-closed, which is the correct behaviour. If this leg ever
    goes GREEN-quiet the first leg's control has lost its denominator.
    """
    doctored_src = tmp_path / "src"
    shutil.copytree(SRC / "foundationscale", doctored_src / "foundationscale")
    probe_py = doctored_src / "foundationscale" / "gates" / "probe.py"
    original = probe_py.read_text(encoding="utf-8")
    sabotaged = original.replace("def derive_declared(", "def derive_declared_sabotaged(", 1)
    assert sabotaged != original, "control: the doctoring must actually take"
    probe_py.write_text(sabotaged, encoding="utf-8")

    result = _run_child(doctored_src)
    assert result.returncode != 0, (
        "the packaged decision path loaded with its moved helpers unavailable "
        "-- a control that cannot fail measures nothing\n"
        f"stdout:\n{result.stdout.decode(errors='replace')}\n"
        f"stderr:\n{result.stderr.decode(errors='replace')}"
    )
    assert b"derive_declared" in result.stderr, (
        f"the failure must name the missing helper:\n{result.stderr.decode(errors='replace')}"
    )


def _build_wheel(out_dir: Path) -> Path:
    """Build a real wheel from the checkout, or fail loudly saying why.

    --no-build-isolation is tried first so the common path needs no network:
    isolated builds fetch the `requires` list from an index, which would make
    this leg red on an offline runner for a reason that has nothing to do with
    the property under test. The isolated build is the fallback for an
    environment without an ambient setuptools.
    """
    attempts: list[tuple[str, subprocess.CompletedProcess[bytes]]] = []
    for label, extra in (("--no-build-isolation", ["--no-build-isolation"]), ("isolated", [])):
        argv = [sys.executable, "-m", "pip", "wheel", "--no-deps", *extra]
        argv += ["--wheel-dir", str(out_dir), str(REPO_ROOT)]
        proc = subprocess.run(
            argv,
            capture_output=True,
            cwd=REPO_ROOT,
            timeout=600,
            check=False,
        )
        built = sorted(out_dir.glob("foundationscale-*.whl"))
        if proc.returncode == 0 and built:
            return built[0]
        attempts.append((label, proc))
    detail = "\n\n".join(
        f"--- {label} (rc={p.returncode}) ---\n{p.stdout.decode(errors='replace')[-1500:]}"
        f"\n{p.stderr.decode(errors='replace')[-1500:]}"
        for label, p in attempts
    )
    raise AssertionError(
        "could not build a wheel by either route, so the packaging claim is "
        "UNMEASURED. Deliberately a failure and not a skip: an unmeasured "
        "packaging property that reports green is the #219 shape itself.\n" + detail
    )


def test_must_pass_a_real_wheel_carries_the_decision_plane_and_no_tools_scripts(
    tmp_path: Path,
) -> None:
    """The strongest form of #219: decide from a built artifact, not a path trick.

    Two claims, each with its denominator stated from the wheel's own namelist:
    the distributed package must CONTAIN gates/probe.py, and must contain NO
    tools/ member. The second is what makes the first leg's derivation mean
    something -- a wheel that shipped tools/ would let the decision path work
    for the pre-#219 reason, and the fix would be untested.
    """
    wheel = _build_wheel(tmp_path / "dist")
    names = zipfile.ZipFile(wheel).namelist()

    package = [n for n in names if n.startswith("foundationscale/")]
    assert "foundationscale/gates/probe.py" in names, (
        "the moved decision machinery is not in the distributed package; "
        f"{len(package)} package member(s) present: {sorted(package)[:12]}"
    )
    shipped_tools = [
        n
        for n in names
        if n.startswith("tools/")
        or Path(n).name in ("real_checkpoint_probe.py", "live_save_gate.py")
    ]
    assert not shipped_tools, (
        "the wheel ships tools/ scripts, so an installed consumer could import "
        f"them and this suite would stop measuring the fix: {shipped_tools}"
    )

    # Extract rather than pip-install into a throwaway venv: same bytes, same
    # sys.path shape for our purposes, without paying venv creation on every
    # CI run across three Pythons. The child re-derives tools/ unreachability
    # from inside, so the extraction cannot quietly re-introduce it.
    site = tmp_path / "site"
    zipfile.ZipFile(wheel).extractall(site)
    result = _run_child(site)
    assert result.returncode == 0, (
        "the decision path could not derive a declared block from the built "
        "wheel alone\n"
        f"stdout:\n{result.stdout.decode(errors='replace')}\n"
        f"stderr:\n{result.stderr.decode(errors='replace')}"
    )
    assert b"PROBE-PACKAGING-OK" in result.stdout
