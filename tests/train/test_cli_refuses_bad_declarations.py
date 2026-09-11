"""``main()`` must refuse a bad declaration itself -- finding #384.

``checks/coverage_floor.py`` went RED on all three CI Python legs for commit
7b033ec: ``src/foundationscale/train/cli.py`` measured 93.9% against a
ratchet floor of 97. The four uncovered statements were cli.py lines 304,
317 and 318 -- the entire ``except ValueError`` refusal arm of ``main()``
-- plus line 380, ``raise SystemExit(main())`` under the ``__main__`` guard,
which no test can cover and which stays uncovered.

The other three lines are a real gap, not ratchet arithmetic. Finding #373
added nine tier-0 declaration flags to this CLI, and
``TrainConfig.__post_init__`` refuses their bad values by raising
``ValueError``; until now no test had ever driven one of those raises
through ``main()``. The code that converts the raise into EXIT_REFUSE --
contract value 96, the refusal a launcher can tell apart from RED (5) and
UNMEASURED (95), replacing the uncaught-traceback exit-1 of the #171 class
-- was therefore asserted by nothing. Releasing the ratchet to 93 would
institutionalise that.

The three arms below are chosen so that each can fail for a different
reason:

* the refusal itself, over four DISTINCT ``__post_init__`` clauses on four
  distinct flags, with ``train`` replaced by a recorder that must stay
  silent -- a refusal that still trains would be worse than none;
* the inverse control: the same argv with a legal value must reach the
  recorder and return what it returned. Without it the refusal arm could be
  passing because argparse rejected the flag upstream of ``_build_config``,
  and nothing here could tell;
* the channel separation. This arm originally pinned argparse's own
  ``SystemExit(2)`` for an unknown flag as distinct from the 96 path, to
  catch a future rewrite that broadened the ``except`` until it swallowed
  argparse's exit. Finding #387 then decided the opposite about the CODE --
  a usage error is the same event as a rejected declaration, so it now
  returns 96 -- WITHOUT giving up the discrimination, because the two
  questions were never the same one. What must not happen is a widening
  that reports an internal crash as a rejected declaration; that is now
  pinned directly, by driving a ``TypeError`` out of ``parse_args`` and
  requiring RED. So the module still fails if the ``except`` broadens, and
  it fails on the arm that actually names the hazard.

Every arm is torch-free: the recorder replaces ``train``, which is where
the heavy imports live, so no real run ever starts.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from foundationscale.train import cli
from foundationscale.train.loop import EXIT_PASS, EXIT_RED, EXIT_REFUSE

# The marker matches train()'s own refusal site so one grep finds both.
# Stated once here so a drift in the literal fails four assertions, not
# one comment nobody greps.
_REFUSE_MARKER = "[fs:train:refuse] declaration rejected:"

# (flag, refused value, legal value, verbatim fragment of the ValueError that
# __post_init__ raises for it). Each row is a DIFFERENT validation clause on
# a DIFFERENT flag, so the refusal arm cannot pass by exercising one check
# four times. The warmup legal value is 0, not 1: the clause is ``>= 0``,
# and the control should sit on the boundary it guards, not clear of it.
_CASES: tuple[tuple[str, str, str, str], ...] = (
    ("--max-grad-norm", "0", "1.0", "max_grad_norm must be > 0"),
    (
        "--gradient-accumulation-steps",
        "0",
        "4",
        "gradient_accumulation_steps must be >= 1",
    ),
    ("--warmup-steps", "-1", "0", "warmup_steps must be >= 0"),
    ("--precision", "int4-bogus", "bf16", "precision='int4-bogus' is not one of"),
)


def _base_argv(tmp_path: Path, *extra: str) -> list[str]:
    """The cheapest argv that reaches ``_build_config`` with one flag varied.

    ``--profile-name`` is the cheapest of the two profile spellings:
    ``TrainConfig.__post_init__`` requires exactly one of profile /
    profile_path / profile_name and checks nothing about the NAME, because
    name resolution lives in ``train()`` -- and every arm here has replaced
    ``train`` with a recorder, so a real profile would be scenery. No flag
    below is invented; each appears in ``cli.build_parser``.
    """
    return [
        "--model",
        "tiny/model",
        "--dataset",
        "tiny/dataset",
        "--output-dir",
        str(tmp_path / "out"),
        "--nodes",
        "1",
        "--gpus-per-node",
        "1",
        "--profile-name",
        "synthetic-profile",
        *extra,
    ]


def _install_recorder(monkeypatch: pytest.MonkeyPatch, calls: list[object]) -> None:
    """Replace ``cli.train`` with a recorder and hand the caller its log.

    Patched in the CLI module's namespace, not in ``loop``'s: ``main`` calls
    the name it imported, and patching anywhere else would test a binding
    the shipping code does not use. The recorder returns EXIT_PASS so ARM 2
    can pin that ``main`` returns whatever ``train`` returned.
    """

    def _recorder(cfg: object) -> int:
        calls.append(cfg)
        return EXIT_PASS

    monkeypatch.setattr(cli, "train", _recorder)


@pytest.mark.parametrize("case", _CASES, ids=[case[0] for case in _CASES])
def test_main_refuses_a_bad_declaration_before_train(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
    case: tuple[str, str, str, str],
) -> None:
    """A bad tier-0 declaration is refused by the CLI, never trained.

    This is the arm that was uncovered on 7b033ec: it drives a ValueError
    out of ``TrainConfig.__post_init__`` through ``main()`` and asserts all
    four halves of the refusal contract -- the exit code, the shared marker,
    the named field, and the silence of ``train``. Each assertion can fail
    alone: deleting the ``except`` arm returns 1 (the #171 class, a
    traceback a launcher cannot interpret); drifting the marker text breaks
    the grep that unifies this site with train()'s; anonymising the message
    leaves the operator with 96 and no cause; and forwarding anyway would
    mean the plane executed under a declaration it had just rejected --
    #375's defect class, worse than no refusal at all.
    """
    flag, bad, _legal, fragment = case
    calls: list[object] = []
    _install_recorder(monkeypatch, calls)

    rc = cli.main(_base_argv(tmp_path, flag, bad))

    out = capsys.readouterr().out
    # EXIT_REFUSE is the contract value 96. What it replaced here was the
    # interpreter catching the raise, printing a traceback and exiting 1 --
    # a code this plane does not define, in the one namespace (0/5/95/96)
    # whose purpose is that a caller can read it without parsing prose.
    assert rc == EXIT_REFUSE, (
        f"main() returned {rc} for {flag} {bad}, not EXIT_REFUSE (96). If "
        "this is 1, __post_init__'s ValueError escaped the except arm; if "
        "it is 0, a rejected declaration trained. Both are what this module "
        "exists to pin (#384)"
    )
    assert _REFUSE_MARKER in out, (
        f"the refusal printed no {_REFUSE_MARKER!r} marker; stdout was "
        f"{out[-300:]!r}. The marker is the shared refusal spelling a "
        "launcher greps across this site and train()'s -- 96 without it is "
        "a verdict with no named cause"
    )
    assert fragment in out, (
        f"the refusal does not name the offending declaration: expected "
        f"{fragment!r} in stdout, got {out[-300:]!r}. An operator handed 96 "
        "must learn WHICH declaration was rejected, not merely that one was"
    )
    assert calls == [], (
        f"train() was invoked {len(calls)} time(s) with a declaration "
        "main() had just refused. The discrimination this arm exists for: a "
        "refusal that still trains looks identical to a refusal in the log"
    )


@pytest.mark.parametrize("case", _CASES, ids=[case[0] for case in _CASES])
def test_legal_value_reaches_train_and_main_returns_its_verdict(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
    case: tuple[str, str, str, str],
) -> None:
    """The inverse control, and what keeps the refusal arm non-vacuous.

    The SAME argv with the offending value replaced by a legal one must
    print no refusal marker, must reach ``train`` exactly once, and must
    return what ``train`` returned. Absent this arm, every refusal case
    above could be passing for the wrong reason -- argparse rejecting the
    flag upstream of ``_build_config``, an unrelated guard refusing first --
    and the module could not tell. It fails, separately from the refusal
    arm, if the legal form is itself refused, if ``main`` stops propagating
    train's verdict, or if the rail around the recorder is satisfied without
    the plane ever being consulted.
    """
    flag, _bad, legal, _fragment = case
    calls: list[object] = []
    _install_recorder(monkeypatch, calls)

    rc = cli.main(_base_argv(tmp_path, flag, legal))

    out = capsys.readouterr().out
    assert _REFUSE_MARKER not in out, (
        f"a LEGAL value for {flag} ({legal!r}) was refused: {out[-300:]!r}. "
        "If the legal arm refuses, the bad arm's refusal proves nothing -- "
        "the CLI would be closed to this flag entirely, not to its bad "
        "values"
    )
    assert len(calls) == 1, (
        f"train() received {len(calls)} call(s) for a legal declaration, "
        "expected exactly one. Zero means some guard refused before the "
        "recorder and the refusal arm is passing on that guard's work; two "
        "means main invoked the plane twice for one argv"
    )
    assert rc == EXIT_PASS, (
        f"main() returned {rc} for the legal value {flag} {legal}; the "
        "recorder returns EXIT_PASS, so this must equal EXIT_PASS exactly. "
        "Anything else breaks the last link of the contract: that main "
        "forwards train's verdict unmodified"
    )


def test_unknown_flag_refuses_96_and_keeps_argparse_diagnosis(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """A mistyped flag is a rejected declaration -- finding #387.

    An unknown flag makes ``parse_args`` raise ``SystemExit(2)`` from inside
    the stdlib, before ``_build_config`` runs. Until #387 that 2 reached the
    shell: a code outside 0/5/95/96, the one a launcher's case statement does
    not handle, for an event -- "you stated something this plane will not
    honour, and nothing was measured" -- that the arm above already reports
    as 96. Both halves are asserted here, because translating the code would
    be a bad trade if it cost the operator argparse's own message: the code
    a launcher reads changes, the evidence a human reads does not.

    The unknown flag rides on a COMPLETE argv rather than standing alone,
    which is not cosmetic: argparse checks required arguments before it
    reports unrecognised ones, so ``main(["--no-such-flag"])`` exits 2 with a
    message that never mentions the flag. That argv would still satisfy the
    exit-code half while making the stderr half meaningless -- it would pass
    for the wrong reason. This one makes the mistyped flag the SOLE defect,
    so argparse's message is about the thing the test claims to be about.
    """
    rc = cli.main(_base_argv(tmp_path, "--no-such-flag"))

    captured = capsys.readouterr()
    assert rc == EXIT_REFUSE, (
        f"main() returned {rc} for an unknown flag, not EXIT_REFUSE (96). "
        "If this is 2, argparse's own exit reached the caller and #387 has "
        "regressed; if it is 5, the usage error was adjudicated as a crash"
    )
    assert _REFUSE_MARKER in captured.out, (
        f"the usage error printed no {_REFUSE_MARKER!r} marker; stdout was "
        f"{captured.out[-300:]!r}. 96 is the same verdict the declaration "
        "arm gives, so it must carry the same marker -- one grep, both sites"
    )
    assert "--no-such-flag" in captured.err, (
        "argparse's own diagnosis is no longer on stderr: "
        f"{captured.err[-300:]!r}. The translation must change the exit code "
        "and nothing else. An operator handed 96 with no usage text is worse "
        "off than one handed 2 with it"
    )


@pytest.mark.parametrize("flag", ["--help", "--version"])
def test_help_and_version_still_exit_zero(flag: str) -> None:
    """#387's other half: the self-describing flags are re-raised untouched.

    ``--help`` and ``--version`` are the tool answering a question about
    itself, not a training run reporting a verdict. Exit 0 is in-contract by
    value and is what every other CLI on the machine does, so the refusal
    translation must test the code and let this one through. Without this
    arm the obvious over-reach -- catching ``SystemExit`` and refusing on all
    of it -- would pass every other test in this module while making
    ``--help`` return 96.
    """
    with pytest.raises(SystemExit) as excinfo:
        cli.main([flag])

    code = 0 if excinfo.value.code is None else excinfo.value.code
    assert code == 0, (
        f"{flag} exited {excinfo.value.code!r}, not 0. If this is "
        f"{EXIT_REFUSE}, the #387 translation swallowed the self-describing "
        "flags along with the usage errors"
    )


class _DuckCode(Exception):
    """An internal fault that happens to carry a ``code``, like SystemExit.

    Contrived on purpose. ``except SystemExit`` is a TYPE test, and the only
    mutation it can have that this module would otherwise miss is one that
    replaces the type test with something weaker -- a wider ``except``, or a
    duck-type check on ``.code``. Every exception whose ``.code`` is missing
    is caught by the widening ANYWAY, because the handler's own
    ``exc.code`` lookup then raises ``AttributeError`` and the outer handler
    still answers RED: the right verdict reached by the wrong route, which
    means such an exception cannot tell the two implementations apart.
    This one can. ``code = 2`` is the value a widened handler would read and
    translate, so it is the single input on which "narrow" and "wide" give
    DIFFERENT answers.
    """

    code = 2


@pytest.mark.parametrize(
    ("exc", "why"),
    [
        (TypeError("simulated internal argparse fault"), "the realistic fault"),
        (_DuckCode("internal fault that duck-types SystemExit"), "the type probe"),
    ],
    ids=["typeerror", "duck_typed_code"],
)
def test_internal_crash_in_parse_args_is_red_not_a_refusal(
    exc: Exception,
    why: str,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """The discrimination the old channel-separation arm existed for.

    #387 made a usage error return 96, which removes the old signal that a
    broadened ``except`` had swallowed argparse. The hazard behind that
    signal is unchanged and is pinned here directly: an internal fault
    during parsing -- a ``TypeError`` from a mistyped argparse action, the
    exact example ``main``'s docstring names -- is NOT a rejected
    declaration, and reporting it as one would tell the operator to fix
    their command line for a defect in ours. It must fall past the narrow
    ``except SystemExit`` to the outer handler and adjudicate RED (5).

    Two faults, because ONE of them cannot do the job alone. Measured while
    writing this: widening the ``except`` to ``BaseException`` and re-running
    left the ``TypeError`` arm GREEN -- the widened handler catches the
    ``TypeError``, its own ``exc.code`` lookup then raises ``AttributeError``,
    and the outer handler answers RED regardless. A control that passes under
    the mutation it names is not a control. ``_DuckCode`` is the arm that
    actually kills it, and the ``TypeError`` arm is kept because it is the
    fault an operator will really meet.
    """

    class _ExplodingParser:
        def parse_args(self, argv: object) -> object:
            raise exc

    monkeypatch.setattr(cli, "build_parser", _ExplodingParser)

    rc = cli.main(["--model", "tiny/model"])

    captured = capsys.readouterr()
    assert rc == EXIT_RED, (
        f"an internal {type(exc).__name__} during parsing ({why}) returned "
        f"{rc}, not EXIT_RED (5). {EXIT_REFUSE} would mean the crash was "
        "laundered into a rejected declaration -- our bug reported as the "
        "operator's, and the except is no longer a SystemExit type test"
    )
    assert _REFUSE_MARKER not in captured.out, (
        "the refusal marker appeared for an internal crash: "
        f"{captured.out[-300:]!r}. The marker means 'your declaration was "
        "rejected'; a fault in our own parser is not that"
    )
