"""``train()`` is TOTAL over the exit contract -- finding #380.

``loop.train``'s docstring has always said "unexpected trainer exceptions
adjudicate as RED". Before #380 that sentence was true of exactly one site.
An AST census over the function body -- 21 ``try`` blocks whose handlers
return an ``EXIT_`` constant, measured against every top-level statement --
found **65** statements standing outside all of them, including
``_effective_topology``, both gate-callback constructors, ``_emit_manifest``
and ``_run_save_gates``. The last one parses safetensors headers off disk and
is the likeliest raiser in the set.

Nothing downstream contained the escape. ``cli.py``'s handoff is a bare
``return train(cfg)``, and that is deliberate: its docstring explains that
wrapping the call in the neighbouring ``except ValueError`` would report a
training failure as a refusal. So a raise left ``train``, left ``main``, and
reached the interpreter, which prints a traceback and exits **1** -- outside
0/5/95/96, and the exact code #171 showed a launcher cannot interpret.

#381 then measured the OTHER side of that handoff and found the same hole.
``train`` is total, but everything ``main`` does BEFORE it -- building the
parser (which asks importlib for installed distribution metadata), parsing,
and binding forty-odd namespace attributes into a ``TrainConfig`` -- was
guarded for ``ValueError`` and nothing else. A ``TypeError`` there exited 1
just as surely, one function earlier. The last three arms pin that boundary
and the split that makes it correct: the pre-handoff region is wrapped, the
handoff is not, and argparse's own ``SystemExit`` still passes through.

This module pins the boundary rather than any one site, because the defect
class is "a statement was added and nobody wrapped it". Guarding 65 sites
would be correct until the 66th. The three arms below are chosen so that each
one fails for a different reason:

* a crash adjudicates RED -- the contract itself;
* a crash whose REPORTING also fails still adjudicates RED -- otherwise the
  handler is a second unprotected region and the fix reintroduces the bug it
  closes;
* ``KeyboardInterrupt`` still propagates -- ``except Exception`` is a
  deliberate choice over ``except BaseException``, and a choice nothing
  measures is a comment.

Every arm is torch-free by construction: the raise is injected at
``Step.START``, the first statement of the body, so no model, dataset or
profile is ever touched and the module costs milliseconds.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from foundationscale.train import cli, loop

_EXIT_RED = 5
_EXIT_REFUSE = 96


def _cfg(tmp_path: Path) -> Any:
    """A config that is valid but never used -- the run dies on line one.

    ``profile_name`` is a placeholder for the same reason it is one in
    ``test_unwired_axes_refuse``: every arm here raises before profile
    resolution, so a real profile would be scenery.
    """
    return loop.TrainConfig(
        model="fake-model",
        dataset="fake-dataset",
        output_dir=tmp_path / "out",
        nodes=1,
        gpus_per_node=1,
        profile_name="synthetic-profile",
        max_steps=2,
        save_interval=1,
    )


def _raise_at_start(exc: BaseException, *, also_fail_reporting: bool = False) -> Any:
    """Build a ``_mark`` replacement that raises at the START marker.

    ``Step.START`` is statement #1 of the body and #1 of the 65 unprotected
    ones, so this reproduces the class at its cheapest point. Later markers
    are forwarded to the real ``_mark`` unless ``also_fail_reporting``, which
    is what lets one arm exercise the boundary with a WORKING reporter and
    another with a broken one.
    """
    real_mark = loop._mark

    def _mark(step: str, msg: str = "") -> None:
        if step == loop.Step.START or also_fail_reporting:
            raise exc
        real_mark(step, msg)

    return _mark


def test_unhandled_exception_adjudicates_red_instead_of_exiting_one(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """The contract: a crash is a verdict, not a traceback."""
    monkeypatch.setattr(loop, "_mark", _raise_at_start(RuntimeError("simulated site failure")))
    cfg = _cfg(tmp_path)

    rc = loop.train(cfg)

    assert rc == _EXIT_RED, (
        f"an unhandled RuntimeError inside the thin path returned {rc}, not "
        f"{_EXIT_RED}. Before #380 it did not return at all -- it propagated "
        "out of train(), out of cli.main(), and exited 1, which is outside "
        "the 0/5/95/96 contract"
    )
    out = capsys.readouterr().out
    assert loop.Step.RED in out, (
        "the crash was adjudicated but not ANNOUNCED: no "
        f"{loop.Step.RED!r} marker in stdout. A verdict a log reader cannot "
        "attribute to a step is #312's defect, and here it would leave the "
        "operator with an exit code and no named cause"
    )
    assert "RuntimeError" in out, (
        "the marker does not name the exception type, so the operator learns "
        f"that something failed but not what. stdout was: {out[-400:]!r}"
    )
    manifest = Path(cfg.output_dir) / loop.MANIFEST_NAME
    assert manifest.exists(), (
        f"no manifest at {manifest}: a crashed run left no record it ran at "
        "all, which is the #225 shape -- the gates read this file, and its "
        "absence is indistinguishable from a run that never started"
    )
    record = json.loads(manifest.read_text())
    assert "crashed" in json.dumps(record), (
        "the manifest exists but does not record stage='crashed', so the "
        "artifact cannot distinguish an unhandled crash from an orderly RED"
    )


def test_a_failing_reporter_does_not_replace_the_verdict(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The handler must be total too, or the fix is the bug one layer up.

    ``_mark`` raises on EVERY call here, including the one the boundary
    handler itself makes. If the handler were unguarded, this arm would
    propagate exactly like the pre-#380 code and the fix would have moved the
    unprotected region rather than removed it.
    """
    monkeypatch.setattr(
        loop,
        "_mark",
        _raise_at_start(RuntimeError("reporter is broken too"), also_fail_reporting=True),
    )

    rc = loop.train(_cfg(tmp_path))

    assert rc == _EXIT_RED, (
        f"returned {rc} when BOTH the run and its reporting failed. The "
        "verdict must survive a broken reporter: an operator whose stdout is "
        "closed still needs an exit code inside the contract"
    )


def test_keyboard_interrupt_still_propagates(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """``except Exception`` over ``except BaseException`` is a real choice.

    Swallowing ``KeyboardInterrupt`` would turn Ctrl-C into a RED verdict --
    an operator cancelling a run would be told the run FAILED, and a wrapper
    would record a defect that never happened. ``SystemExit`` is the same
    story for any caller that raises it deliberately.
    """
    monkeypatch.setattr(loop, "_mark", _raise_at_start(KeyboardInterrupt()))

    with pytest.raises(KeyboardInterrupt):
        loop.train(_cfg(tmp_path))


# --- the cli.main boundary (#381) -------------------------------------------
#
# These arms never reach train(): each one raises in the pre-handoff region,
# so no model, dataset or profile is touched and they cost microseconds. The
# argv below is the minimum the parser requires, and it is never honoured.

_MIN_ARGV = [
    "--model",
    "m",
    "--dataset",
    "d",
    "--output-dir",
    "o",
    "--nodes",
    "1",
    "--gpus-per-node",
    "1",
    "--profile-name",
    "synthetic-profile",
]


def test_cli_crash_before_the_handoff_adjudicates_red(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """A crash in parser construction is a verdict, not a traceback.

    ``build_parser`` calls ``fs_version()``, and importlib raises when the
    distribution is not installed the way it expects. Before #381 that left
    ``main``, reached the interpreter and exited 1.
    """

    def _boom() -> str:
        raise RuntimeError("distribution metadata unavailable")

    monkeypatch.setattr(cli, "fs_version", _boom)

    rc = cli.main(_MIN_ARGV)

    assert rc == _EXIT_RED, (
        f"main() returned {rc} when parser construction raised, not "
        f"{_EXIT_RED}. Before #381 it did not return at all -- the raise "
        "reached the interpreter and exited 1, outside 0/5/95/96"
    )
    out = capsys.readouterr().out
    assert "RuntimeError" in out, (
        "the verdict does not name the exception type, so the operator "
        f"learns that something failed but not what. stdout was: {out[-300:]!r}"
    )


def test_cli_non_valueerror_is_red_not_refuse(monkeypatch: pytest.MonkeyPatch) -> None:
    """The crash guard must not be reachable THROUGH the refusal handler.

    ``_build_config`` is wrapped for ``ValueError`` because that is how
    ``TrainConfig`` rejects a declaration. Anything else out of it is a defect
    in the plane, not a refusal by the operator, and reporting it as 96 would
    tell a launcher that the operator asked for something impossible when in
    fact the code broke. This arm fails if the two handlers are ever merged.
    """

    def _boom(argv: object, args: object) -> object:
        raise TypeError("argparse action produced an unexpected type")

    monkeypatch.setattr(cli, "_build_config", _boom)

    rc = cli.main(_MIN_ARGV)

    assert rc == _EXIT_RED, (
        f"a TypeError out of _build_config returned {rc}. {_EXIT_RED} (RED) "
        f"is the answer: {_EXIT_REFUSE} (REFUSE) would claim the operator's "
        "declaration was rejected, which is a different fact"
    )


def test_cli_splits_argparse_exits_by_kind() -> None:
    """``except Exception`` over ``except BaseException``, one layer along.

    argparse exits 2 on a usage error and 0 on ``--help``/``--version``, both
    by raising ``SystemExit`` from inside the stdlib. This arm originally
    required BOTH to pass through untouched, on the reasoning that catching
    them would let this plane relabel another library's contract.

    #387 kept half of that and reversed the other half, because the two are
    not one surface. Passing 2 through does not preserve argparse's contract
    for the caller -- it hands a LAUNCHER a code outside the 0/5/95/96
    namespace this plane publishes, and 2 is precisely the code its case
    statements do not handle. A usage error is the same event as a rejected
    declaration (the operator stated something the plane will not honour, and
    nothing was measured), so ``main`` now translates it to EXIT_REFUSE while
    leaving argparse's own message on stderr untouched. ``--help`` and
    ``--version`` still pass through at 0: in-contract by value, and the
    tool answering a question about itself is not a verdict about a run.

    What this arm still does is what it was written for -- it fails if the
    translation stops discriminating. Blanket-catching ``SystemExit`` and
    refusing on all of it turns the ``--help`` leg red.
    """
    rc = cli.main(["--model", "m"])  # --dataset and --output-dir are required
    assert rc == _EXIT_REFUSE, (
        f"argparse's usage exit came back as {rc!r}, not REFUSE "
        f"({_EXIT_REFUSE}). If this is 2, #387's translation is gone and a raw "
        "stdlib code is reaching launchers again; if it is 5, a usage error "
        "is being reported as a crash in this plane"
    )

    with pytest.raises(SystemExit) as helped:
        cli.main(["--help"])
    assert helped.value.code == 0, (
        f"--help exited {helped.value.code!r}, not 0 -- asking for help is "
        "not a failure, and a wrapper that reads the code would record one. "
        f"{_EXIT_REFUSE} here means the #387 translation stopped testing the "
        "code and started swallowing every SystemExit alike"
    )
