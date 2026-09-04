"""The package front door: one import surface, resolved lazily.

``src/foundationscale/__init__.py`` publishes the framework's public names
through PEP 562. Three claims need pinning, and each one carries a
DENOMINATOR:

1. **Resolution.** Every name in the public surface resolves to a real
   object. Denominator: ``foundationscale._EXPORTS`` — the mapping that *is*
   the surface. A name added there without a working home fails here rather
   than at a user's site.
2. **Laziness.** ``import foundationscale`` leaves torch out of
   ``sys.modules``, and so does reaching the gate plane / topology /
   provenance / adapters through it. That is what lets the gate plane run on
   hosts where torch is absent by design.
3. **Partition.** Every exported name is declared either torch-free or
   torch-requiring. Adding a 20th export forces that decision instead of
   silently escaping claim 2.

Claim 2 **cannot** be measured in-process. pytest has already imported torch
from other test modules, so an in-process ``"torch" not in sys.modules``
assertion is spuriously red — or, if this file happens to be collected
first, vacuously green. It is measured in an isolated child interpreter.

And the child probe is itself controlled. ``MUST_FIRE`` touches
``TrainConfig``, which legitimately pulls torch: the probe must report torch
PRESENT. Same program, same map, same child shape as the green legs — so a
torch-free verdict is evidence that the sensor looked, not silence.
"""

from __future__ import annotations

import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

import foundationscale

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
PROBE_TIMEOUT_SECONDS = 180

#: The heavy roots that must stay out of a bare ``import foundationscale``.
FORBIDDEN_ROOTS = ("torch", "numpy", "transformers", "safetensors")

#: Names reachable without paying for torch. The gate plane, the cluster
#: profile, provenance topology and adapter selection all sit here on
#: purpose: gates run where torch is not installed.
TORCH_FREE_NAMES = (
    "AdapterRefusal",
    "ClusterProfile",
    "Coverage",
    "GateRegistry",
    "GateReport",
    "REGISTRY",
    "Topology",
    "TopologyConsistency",
    "Verdict",
    "select_adapter",
    "verify_controls",
)

#: Names that legitimately pull the training stack in. Not a defect — this is
#: the half of the surface that exists to drive torch.
TORCH_REQUIRING_NAMES = (
    "CheckpointFormatError",
    "EXIT_PASS",
    "EXIT_RED",
    "EXIT_REFUSE",
    "EXIT_UNMEASURED",
    "FoundationScaleSaveGate",
    "TrainConfig",
)

_MARKER = "__FS_FRONT_DOOR__ "

# The child runs under ``-I``: no CWD on sys.path, no PYTHONPATH, no user
# site. It imports foundationscale, optionally touches attributes through the
# front door, and reports which forbidden roots ended up in sys.modules. Any
# internal failure exits 3 in-band rather than looking like a clean probe.
_CHILD_PROGRAM = r"""
import json, sys

src_root, forbidden_csv, touch_csv = sys.argv[1], sys.argv[2], sys.argv[3]
sys.path.insert(0, src_root)
forbidden = [r for r in forbidden_csv.split(",") if r]
touch = [n for n in touch_csv.split(",") if n]
stage = "import"
payload = {"stage": stage, "touched": [], "present": [], "error": None}
try:
    import foundationscale
    payload["present"] = sorted(r for r in forbidden if r in sys.modules)
    payload["after_import"] = list(payload["present"])
    stage = "touch"
    payload["stage"] = stage
    for name in touch:
        getattr(foundationscale, name)
        payload["touched"].append(name)
    payload["present"] = sorted(r for r in forbidden if r in sys.modules)
except BaseException as exc:  # noqa: BLE001 - reported in-band, never silent
    payload["error"] = f"{type(exc).__name__}: {exc}"
    print(_M + json.dumps(payload))
    sys.exit(3)
print(_M + json.dumps(payload))
""".replace("_M", repr(_MARKER))


@dataclass(frozen=True)
class ProbeResult:
    """A child-interpreter measurement, rc first and payload second."""

    returncode: int
    stdout: str
    stderr: str
    payload: dict[str, Any] | None


def _probe(touch: tuple[str, ...] = ()) -> ProbeResult:
    proc = subprocess.run(
        [
            sys.executable,
            "-I",
            "-c",
            _CHILD_PROGRAM,
            str(SRC_ROOT),
            ",".join(FORBIDDEN_ROOTS),
            ",".join(touch),
        ],
        capture_output=True,
        text=True,
        check=False,
        timeout=PROBE_TIMEOUT_SECONDS,
    )
    payload: dict[str, Any] | None = None
    for line in proc.stdout.splitlines():
        if line.startswith(_MARKER):
            try:
                payload = json.loads(line[len(_MARKER) :])
            except json.JSONDecodeError:
                payload = None
    return ProbeResult(proc.returncode, proc.stdout, proc.stderr, payload)


def _assert_probe_ran(result: ProbeResult) -> dict[str, Any]:
    """rc FIRST, then the marker. A dead probe must not read as a pass."""
    assert result.returncode == 0, (
        f"front-door probe exited {result.returncode}\n"
        f"stdout: {result.stdout}\nstderr: {result.stderr}"
    )
    assert result.payload is not None, (
        f"front-door probe emitted no {_MARKER!r} line — it did not measure.\n"
        f"stdout: {result.stdout}\nstderr: {result.stderr}"
    )
    return result.payload


class TestPublicSurface:
    """Claim 1 and claim 3: what is exported, and is it all accounted for."""

    def test_export_map_is_non_empty(self) -> None:
        # Guards the vacuous case: an empty map would make every loop below
        # pass over zero units.
        assert foundationscale._EXPORTS, "the export map is empty — nothing measured"

    def test_all_is_the_export_map_plus_version(self) -> None:
        assert foundationscale.__all__ == [
            *sorted(foundationscale._EXPORTS),
            "__version__",
        ]

    def test_every_exported_name_resolves(self) -> None:
        unresolved: list[str] = []
        for name in sorted(foundationscale._EXPORTS):
            try:
                getattr(foundationscale, name)
            except Exception as exc:  # noqa: BLE001 - collected, not swallowed
                unresolved.append(f"{name} -> {type(exc).__name__}: {exc}")
        assert not unresolved, "front-door names that do not resolve:\n" + "\n".join(unresolved)

    def test_exported_name_is_the_object_from_its_home_module(self) -> None:
        import importlib

        for name, home in sorted(foundationscale._EXPORTS.items()):
            expected = getattr(importlib.import_module(home), name)
            assert getattr(foundationscale, name) is expected, (
                f"foundationscale.{name} is not {home}.{name}"
            )

    def test_torch_partition_covers_the_whole_surface(self) -> None:
        partition = set(TORCH_FREE_NAMES) | set(TORCH_REQUIRING_NAMES)
        surface = set(foundationscale._EXPORTS)
        assert not surface - partition, (
            "exported but unclassified — declare each in TORCH_FREE_NAMES or "
            f"TORCH_REQUIRING_NAMES: {sorted(surface - partition)}"
        )
        assert not partition - surface, (
            f"classified but not exported: {sorted(partition - surface)}"
        )
        assert not set(TORCH_FREE_NAMES) & set(TORCH_REQUIRING_NAMES)

    def test_unknown_attribute_raises_attribute_error(self) -> None:
        with pytest.raises(AttributeError, match="no attribute 'NoSuchThing'"):
            # No `type: ignore` here: __getattr__ makes every name typecheck,
            # which is exactly why this runtime leg has to exist.
            foundationscale.NoSuchThing  # noqa: B018

    def test_dir_lists_the_public_surface(self) -> None:
        # Python sorts whatever __dir__ returns, so compare against the sorted
        # surface rather than __all__'s version-last ordering.
        assert dir(foundationscale) == sorted(foundationscale.__all__)

    def test_no_export_collides_with_a_submodule_name(self) -> None:
        """A submodule shadows ``__getattr__`` — such a name can never resolve.

        Real attributes win over PEP 562, and importing ``foundationscale.X``
        binds ``X`` on the parent package. So an export named after a
        subpackage silently resolves to the module forever. ``train`` is the
        instance that caught this; the gate is over the class.
        """
        package_dir = Path(foundationscale.__file__).parent
        submodules = {p.stem for p in package_dir.glob("*.py") if p.stem != "__init__"}
        submodules |= {p.name for p in package_dir.iterdir() if (p / "__init__.py").is_file()}
        assert submodules, "found no submodules — the collision gate cannot fire"
        collisions = sorted(submodules & set(foundationscale._EXPORTS))
        assert not collisions, (
            "exported names that a submodule of the same name will shadow: "
            f"{collisions}. Python binds the submodule as a real attribute on "
            "import, and real attributes take precedence over __getattr__, so "
            "these can only ever resolve to the module. Re-export under a "
            "different name or document the submodule path instead."
        )

    def test_version_is_exported(self) -> None:
        assert foundationscale.__version__


class TestLazinessInAChildInterpreter:
    """Claim 2, measured where pytest's own imports cannot contaminate it."""

    def test_bare_import_pulls_in_nothing_heavy(self) -> None:
        payload = _assert_probe_ran(_probe())
        assert payload["present"] == [], (
            "import foundationscale is no longer torch-free; it pulled in "
            f"{payload['present']} — the front door must stay lazy"
        )

    def test_gate_plane_stays_torch_free_through_the_front_door(self) -> None:
        payload = _assert_probe_ran(_probe(TORCH_FREE_NAMES))
        assert payload["touched"] == list(TORCH_FREE_NAMES), (
            f"probe touched {payload['touched']}, expected {list(TORCH_FREE_NAMES)}"
        )
        assert payload["present"] == [], (
            f"reaching {TORCH_FREE_NAMES} pulled in {payload['present']}; the gate "
            "plane must import on hosts without torch"
        )

    def test_must_fire_touching_the_training_surface_does_pull_torch(self) -> None:
        # The control. Same program, same map, same child shape as the two
        # green legs above — so their [] is a measurement, not a blind sensor.
        payload = _assert_probe_ran(_probe(("TrainConfig",)))
        assert payload["after_import"] == [], "child was contaminated before touching"
        assert "torch" in payload["present"], (
            "MUST_FIRE did not fire: touching TrainConfig left torch out of "
            f"sys.modules (present={payload['present']}). The laziness probe "
            "above cannot be trusted until this leg goes red on eager imports."
        )

    def test_must_fire_a_broken_name_is_reported_not_silently_green(self) -> None:
        # Second control, on the harness rather than the package: a name that
        # cannot resolve must surface as rc 3 with an error, never as a clean
        # torch-free pass.
        result = _probe(("NoSuchThing",))
        assert result.returncode == 3, (
            f"probe returned {result.returncode} for an unresolvable name; a "
            "failed child must not read as a pass"
        )
        assert result.payload is not None
        assert "AttributeError" in (result.payload["error"] or "")
