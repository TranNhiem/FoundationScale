"""``examples/`` sits in a test denominator, because it did not and it rotted.

The suite was ~1,100 tests green while ``python examples/train_tiny.py`` — the
one command a new user runs first — died on its own line 22 with
``ValueError: node_pattern is not a valid regex``. The example declared
``"compute-[01-08]"``, a Slurm hostlist; ``ClusterProfile`` compiles
``node_pattern`` as a regex, and ``[01-08]`` contains the range ``1-0``.

The reason that was invisible is the interesting part, and it is why this file
is generic rather than a one-line fixture patch: the two tests that need the
same profile (``test_train_entry``, ``test_train_execution``) each carry their
own **copy** of it with the correct spelling. Three copies of one fixture, and
the copy under test was not the copy that shipped. Anything not in a denominator
drifts; ``examples/`` was in none.

So the check here is over *every* example, discovered — a new example is covered
the day it lands, without anyone remembering to add a test. Import is the whole
mechanism: every example guards its work behind ``if __name__ == "__main__"``,
so importing one executes exactly its declarations (profiles, configs, constants)
and trains nothing. That guard is asserted, not assumed — without it this file
would launch real training runs.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

EXAMPLES = Path(__file__).resolve().parents[2] / "examples"

# The broken spelling that shipped, kept verbatim as the MUST_FIRE control's
# payload. When the control stops raising, the import check has gone inert.
HOSTLIST_NOT_REGEX = '"compute-[01-08]"'


def _example_files() -> list[Path]:
    return sorted(p for p in EXAMPLES.glob("*.py") if not p.name.startswith("_"))


def _import_isolated(path: Path) -> object:
    """Import a file as a throwaway module, leaving ``sys.modules`` as found.

    The name is derived from the path so two examples cannot collide, and the
    entry is removed in ``finally`` so a failed import cannot leave a half-built
    module behind for the next test to import successfully.
    """
    name = f"_fs_example_{path.stem}"
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None, f"no import spec for {path}"
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    finally:
        sys.modules.pop(name, None)
    return module


def test_examples_directory_is_not_empty() -> None:
    """The denominator is real.

    Every other test in this file iterates ``_example_files()``. If ``examples/``
    is renamed or emptied, parametrisation collapses to zero cases and the whole
    file passes green over nothing — ``all([]) is True``, this repository's
    founding shape. This is the one assertion that cannot be satisfied vacuously.
    """
    found = _example_files()
    assert found, f"no example modules under {EXAMPLES} — the suite below measures nothing"


@pytest.mark.parametrize("path", _example_files(), ids=lambda p: p.name)
def test_example_guards_its_entry_point(path: Path) -> None:
    """No example may do real work at import time.

    Two reasons, and the second is why this runs before the import test:
    a user who imports an example to crib from it should not start a training
    run, and ``test_example_imports_cleanly`` below is only safe while this
    holds.
    """
    source = path.read_text(encoding="utf-8")
    assert '__name__ == "__main__"' in source, (
        f"{path.name} has no __main__ guard: importing it would run it"
    )


@pytest.mark.parametrize("path", _example_files(), ids=lambda p: p.name)
def test_example_imports_cleanly(path: Path) -> None:
    """Executes each example's declarations — the half that shipped broken.

    A ``ClusterProfile`` is validated in ``__post_init__``, so constructing one
    is a real check of the values, not a syntax check.
    """
    _import_isolated(path)


def test_declared_gpu_count_has_exactly_one_source() -> None:
    """One knob, three readers.

    ``train_tiny`` previously wrote the GPU count in three places — the profile,
    ``gpus_per_node`` and ``dp`` — so a 4-GPU first run meant editing three
    fields consistently and debugging a topology RED when you missed one. The
    example now derives all three from ``GPUS``; this pins that, because the
    drift is silent and only bites users who are not on a 1-GPU box.
    """
    module = _import_isolated(EXAMPLES / "train_tiny.py")
    gpus = getattr(module, "GPUS", None)
    assert isinstance(gpus, int) and gpus >= 1, "train_tiny must expose a GPUS int"
    assert module.PROFILE.gpus_per_node == gpus, (
        "PROFILE.gpus_per_node drifted from GPUS — the count has two sources again"
    )


def test_the_shipped_defect_would_now_be_caught(tmp_path: Path) -> None:
    """MUST_FIRE: the import check catches the exact bug that got through.

    A check that has never been observed failing is not evidence that it works.
    This plants the original hostlist spelling into a real copy of the example
    and asserts the same machinery the tests above use raises on it. Only after
    that does it assert the shipped file is clean — a negative control alone
    would pass just as well against an import that quietly swallows errors.
    """
    source = (EXAMPLES / "train_tiny.py").read_text(encoding="utf-8")
    assert HOSTLIST_NOT_REGEX not in source, "the hostlist spelling is back in the shipped example"

    # r"compute-0[1-8]" -> "compute-[01-08]": valid regex to Slurm hostlist.
    doctored = source.replace('r"compute-0[1-8]"', HOSTLIST_NOT_REGEX, 1)
    assert doctored != source, (
        "could not plant the defect — the node_pattern literal moved, so this "
        "control is measuring nothing and must be re-anchored"
    )

    planted = tmp_path / "train_tiny.py"
    planted.write_text(doctored, encoding="utf-8")
    with pytest.raises(ValueError, match="node_pattern is not a valid regex"):
        _import_isolated(planted)
