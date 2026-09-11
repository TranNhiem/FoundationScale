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
* the channel separation: argparse's own error for an unknown flag is
  ``SystemExit(2)``, not the 96 path, and carries no refusal marker, so a
  future rewrite that broadens the ``except`` to swallow argparse's exit is
  caught.

Every arm is torch-free: the recorder replaces ``train``, which is where
the heavy imports live, so no real run ever starts.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from foundationscale.train import cli
from foundationscale.train.loop import EXIT_PASS, EXIT_REFUSE

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


def test_unknown_flag_is_argparse_error_not_a_refusal(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """The two error channels stay distinct.

    An unknown flag is argparse's own rejection: ``parse_args`` raises
    ``SystemExit(2)`` on the spot, before ``_build_config`` runs. This pins
    that the 96 arm is reachable ONLY through ``TrainConfig.__post_init__``:
    a future edit that widened ``except ValueError`` to ``except Exception``
    -- or wrapped ``parse_args`` -- would launder argparse's 2 into a
    refusal-shaped 96, and this arm, not the refusal arm, is what fails.
    No ``train`` recorder is needed: nothing downstream of the parse runs.
    """
    with pytest.raises(SystemExit) as excinfo:
        cli.main(["--no-such-flag"])

    assert excinfo.value.code == 2, (
        f"argparse rejected the unknown flag with exit "
        f"{excinfo.value.code}, not its own 2. If this were "
        f"{EXIT_REFUSE} (96), the refusal except would have swallowed "
        "argparse's exit -- the 96 path must be reachable only through "
        "_build_config, and any broadening must fail right here"
    )
    captured = capsys.readouterr()
    assert _REFUSE_MARKER not in captured.out, (
        "the refusal marker appeared on stdout for an ARGPARSE error. "
        f"{_REFUSE_MARKER!r} is reserved for rejected declarations; "
        "reusing it for a mistyped flag teaches launchers the wrong cause"
    )
    assert _REFUSE_MARKER not in captured.err, (
        "the refusal marker appeared even in argparse's own stderr channel: "
        f"{captured.err[-300:]!r}. The channels are distinct on purpose, "
        "and this arm exists to keep them that way"
    )
