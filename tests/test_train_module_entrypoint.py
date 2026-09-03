"""Module-entry control for defect #246: ``python -m foundationscale.train`` must be a real entry.

Before the fix there was no ``src/foundationscale/train/__main__.py``. Two
invocation forms did work, and still do: the ``foundationscale-train`` console
script declared under ``[project.scripts]`` (#224's fix), and ``python -m
foundationscale.train.cli`` -- the SUBMODULE form, which runs because cli.py
carries its own ``if __name__ == "__main__"`` guard. What did not work is the
PACKAGE form, ``python -m foundationscale.train``, which died with runpy's
"No module named foundationscale.train.__main__; 'foundationscale.train' is a
package and cannot be directly executed".

The denominator for "documented" is one site, stated here rather than rounded
up into a plural: ``docs/deliverables/B2_scaling.md:196`` declares
``entrypoint: "foundationscale.train"`` -- the package path, not the submodule
path. No prose runbook in this repository invokes the package form as a
command; ``README.md:248`` and ``pyproject.toml:98`` both name the ``.cli``
form, which was never broken. The defect is therefore narrow and it is real:
one machine-readable entrypoint declaration names a module path that imports
and cannot be executed, so a consumer resolving that declaration the obvious
way gets an error claiming the package cannot be EXECUTED rather than one
admitting the documented target does not exist. Finding #246. The shape
rhymes with #224 (an advertised command pip never created): an invocation form
indistinguishable in a document from a real one will be read as real, so the
realness has to be *run*, not narrated.

Why ``--version`` is the measurement and not an import. A test that merely
imports ``foundationscale.train`` would have been GREEN against the broken
layout: the package always imported fine, because what was missing was the
file runpy looks for, and runpy is only exercised by actually running ``-m``
in a child interpreter. Each leg below therefore measures a subprocess. Each
leg states its denominator: one invocation of one exact argv, against one
exactly specified ``src/`` tree, with a cwd that provably cannot supply the
package via the interpreter's sys.path[0] prepend.

Why a MUST_FIRE leg is what makes the MUST_PASS leg mean anything. Leg 1
points ``PYTHONPATH`` at ``SRC`` and demands green. Without leg 2, that green
is compatible with a story where some unrelated installed distribution
shadowed the import and answered ``--version`` on this checkout's behalf --
the same green, measuring someone else's bytes. Leg 2 replays the identical
command against a copy of the package with ``__main__.py`` deleted and
demands red, so the green in leg 1 is attributable to the presence of that
specific file and nothing else.

No ``pytest.skip`` anywhere in this module, for any reason: CI runs with
``FS_FORBID_SKIPS=1``, where a skip is itself a failure. No network, no GPU,
and -- a future reader will assume otherwise, because the module under test
is the training entry point -- NO ``train`` extra and no torch import.
``--version`` and ``--help`` both exit inside argparse, before ``train()`` is
called, and the no-argument leg exits inside argparse's usage error, so the
heavy optional dependencies are never touched on any path this file
exercises. The helpers this module pre-imports (``build_parser``,
``fs_version``) come from modules whose import graphs are stdlib-only at
module level; the lazily loaded torch-dependent code sits behind function
calls this suite never makes.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

from foundationscale.train.cli import build_parser
from foundationscale.train.loop import fs_version

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC = REPO_ROOT / "src"


def _child_env(src_root: Path) -> dict[str, str]:
    """One environment builder for every leg, so the four legs cannot drift.

    The legs disagree ONLY about argv and about which ``src/`` tree they
    measure; if each assembled its own environment the suite would quietly
    diverge in the variable that decides what gets measured. ``PYTHONPATH``
    is *overwritten*, never appended to: an ambient PYTHONPATH pointing at a
    developer's editable install is precisely the shadowing arrangement leg 2
    exists to expose, and inheriting it would re-introduce that arrangement
    behind the test's back.
    """
    env = dict(os.environ)
    env["PYTHONPATH"] = str(src_root)
    return env


def _run_module(
    args: list[str],
    src_root: Path,
    workdir: Path,
) -> subprocess.CompletedProcess[bytes]:
    """Run ``python -m foundationscale.train <args>`` against ``src_root``.

    Callers pass ``workdir=tmp_path``, never ``REPO_ROOT``: with ``-m`` the
    interpreter prepends the cwd to sys.path, and a cwd of the repo root
    would let the package resolve from the checkout no matter what
    PYTHONPATH said, turning every leg into a measurement of argv[0]'s
    neighbourhood rather than of the tree under test. A timeout raises
    rather than skips -- an entry point that hangs is red, not unmeasured,
    and this repo reports unmeasured out loud rather than as green anyway.
    """
    return subprocess.run(
        [sys.executable, "-m", "foundationscale.train", *args],
        capture_output=True,
        env=_child_env(src_root),
        cwd=workdir,
        timeout=120,
        check=False,
    )


def test_must_pass_module_entrypoint_runs_and_reports_fs_version(tmp_path: Path) -> None:
    """The #246 positive control: ``-m foundationscale.train`` must execute.

    Denominator: exactly one invocation of ``python -m foundationscale.train
    --version`` with cwd=tmp_path (an empty directory, so the interpreter's
    sys.path[0] prepend cannot supply the package) and PYTHONPATH=SRC, so the
    bytes that answer are this checkout's and not whatever happens to be
    installed on the developer's machine. ``--version`` is chosen because it
    exercises the CLI's argparse prologue -- the part a launcher actually
    depends on -- while exiting before ``train()`` can ask for torch.
    """
    result = _run_module(["--version"], SRC, tmp_path)
    assert result.returncode == 0, (
        "python -m foundationscale.train --version must exit 0 against this "
        "checkout; before #246 there was no train/__main__.py for runpy to "
        "find and this command died with 'No module named "
        "foundationscale.train.__main__'\n"
        f"stdout:\n{result.stdout.decode(errors='replace')}\n"
        f"stderr:\n{result.stderr.decode(errors='replace')}"
    )
    stdout = result.stdout.decode(errors="replace")
    assert fs_version() in stdout, (
        "the module entry must report the same version the library reports; "
        f"fs_version() returned {fs_version()!r}, the child printed:\n{stdout}"
    )


def test_must_fire_deleting___main___py_turns_the_same_command_red(tmp_path: Path) -> None:
    """The control behind the control: the same command, minus the fixed file, must fail.

    Leg 1's green is evidence only if the same detector can produce red on
    the same corpus. So this leg copies the package, proves the copy is
    healthy by running the IDENTICAL command and demanding green (without
    that pre-flight, the red below could be blamed on a broken copy rather
    than on the deletion), then removes ``train/__main__.py`` -- the exact
    file whose absence was #246 -- and demands red, with stderr naming
    ``__main__``. Denominator: two invocations of one exact argv against one
    copy of the package, changing precisely one file between them. If this
    leg ever went GREEN-quiet, leg 1 would be measuring whether SOME
    foundationscale answered on this machine, not whether THIS checkout has a
    module entry point.
    """
    doctored_root = tmp_path / "src"
    shutil.copytree(SRC / "foundationscale", doctored_root / "foundationscale")

    intact = _run_module(["--version"], doctored_root, tmp_path)
    assert intact.returncode == 0, (
        "pre-flight: the PRISTINE copy must pass, so the failure below is "
        "attributable to the deletion and not to copying\n"
        f"stdout:\n{intact.stdout.decode(errors='replace')}\n"
        f"stderr:\n{intact.stderr.decode(errors='replace')}"
    )

    main_py = doctored_root / "foundationscale" / "train" / "__main__.py"
    assert main_py.exists(), "control: #246's fix file must be present in the pristine copy"
    main_py.unlink()

    doctored = _run_module(["--version"], doctored_root, tmp_path)
    assert doctored.returncode != 0, (
        "the module entry point still worked with train/__main__.py deleted "
        "-- a control that cannot fail measures nothing, and leg 1's green "
        "has lost its denominator\n"
        f"stdout:\n{doctored.stdout.decode(errors='replace')}\n"
        f"stderr:\n{doctored.stderr.decode(errors='replace')}"
    )
    assert "__main__" in doctored.stderr.decode(errors="replace"), (
        "the failure must be runpy's 'No module named "
        "foundationscale.train.__main__', not some unrelated crash:"
        f"\n{doctored.stderr.decode(errors='replace')}"
    )


def test_identity_module_help_is_the_console_script_help(tmp_path: Path) -> None:
    """``-m foundationscale.train`` and ``foundationscale-train`` must be the SAME parser.

    Denominator: the complete ``--help`` text of one subprocess invocation,
    compared byte-for-byte against one ``build_parser().format_help()`` from
    this same interpreter (same Python, same argparse formatter, so no
    version skew can live in the comparison). The comparison is meaningful
    only because ``cli.build_parser`` sets ``prog="foundationscale-train"``
    explicitly: argparse would otherwise derive prog from ``sys.argv[0]`` and
    the two invocation forms would print two different names, telling the
    reader they are two different tools. The explicit prog= is what makes the
    module form and the script form indistinguishable to the person reading
    the help -- which is the entire point of #246. One entry point,
    documented two ways, that prints two different banners is the defect
    re-expressing itself one layer down.
    """
    result = _run_module(["--help"], SRC, tmp_path)
    assert result.returncode == 0, (
        "python -m foundationscale.train --help must exit 0"
        f"\nstderr:\n{result.stderr.decode(errors='replace')}"
    )
    expected = build_parser().format_help()
    assert result.stdout.decode(errors="replace") == expected, (
        "the module entry's --help differs from the console script's parser; "
        "#246's fix is 'one entry point reached two ways', and a divergent "
        "help text is two entry points\n"
        f"--- module form ---\n{result.stdout.decode(errors='replace')}\n"
        f"--- console script parser ---\n{expected}"
    )


def test_exit_code_hygiene_no_arguments_exits_two_without_a_traceback(tmp_path: Path) -> None:
    """A bare invocation must be a usage error, not a crash.

    Denominator: one invocation of ``python -m foundationscale.train`` with no
    arguments. The required options (``--model``, ``--dataset``,
    ``--output-dir``, ``--nodes``, ``--gpus-per-node``, a profile) are absent
    by construction, so argparse itself must reject the call with exit code 2
    and a usage line. The ``"Traceback" not in stderr`` clause is #171/#169's
    rule applied to the module form: the training plane declares a 0/5/95/96
    exit namespace, and an entry point that leaks a traceback has left that
    namespace without saying so -- a reader of the exit code cannot tell a
    refusal from a crash, and a reader of the log cannot tell either from a
    bug in the gate itself. Argparse's own error handling, exercised here, is
    what keeps the module form inside the declared contract before any GPU-
    adjacent code can run.
    """
    result = _run_module([], SRC, tmp_path)
    stderr = result.stderr.decode(errors="replace")
    assert result.returncode == 2, (
        "argparse's usage error for missing required options is exit code 2; "
        f"the module entry exited {result.returncode}\n"
        f"stdout:\n{result.stdout.decode(errors='replace')}\n"
        f"stderr:\n{stderr}"
    )
    assert "Traceback" not in stderr, (
        "a traceback on the usage-error path means the entry point left the "
        f"declared 0/5/95/96 namespace (#171/#169):\n{stderr}"
    )
    assert "usage:" in stderr, (
        "exit 2 must be argparse's own usage error, not some other failure "
        f"that happens to share the exit code:\n{stderr}"
    )
