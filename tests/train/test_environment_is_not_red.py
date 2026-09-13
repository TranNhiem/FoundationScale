"""An environment fault is not this plane's failure -- finding #409.

MEASURED on a GB200 estate, and that measurement is what filed this. Four
telemetry tests in ``test_train_execution.py`` failed their own
``rc in (EXIT_PASS, EXIT_UNMEASURED)`` precondition on **rc=5**. The framework
was fine. ``$HOME`` is NFS, NFS serves no ``flock``, and the HuggingFace
filelock raised ``OSError(37, 'No locks available')`` before a single parameter
was read. ``except Exception -> RED`` then published that as a broken training
plane. The identical suite returned 0 once ``HF_HOME`` pointed at node-local
scratch; nothing about the framework differed between the two runs.

That is the #406/#408 shape a third time -- a verdict stated with more
confidence than the measurement supports -- and it is the one that would reach
the public worst. Most HPC estates mount ``$HOME`` over NFS, so the default
first run of FoundationScale on such an estate would accuse FoundationScale.

The split this module pins:

* a bad model id, an unsupported architecture, a genuinely missing weight file
  -> **RED (5)**. The operator asked for something the plane cannot build.
* no file locking, a full disk, a read-only mount, an exhausted quota, an
  unreachable hub -> **REFUSE (96)**. Construction never got far enough to
  measure anything, so there is no run to score.

Four arms, each failing for a different reason:

* the classifier fires on a DIRECT environment errno -- the contract;
* it fires on a WRAPPED one -- libraries wrap ``OSError`` in their own type far
  more often than they let it through, so a top-frame-only check would classify
  every real-world ENOLCK as RED and the split would do nothing at all;
* it does NOT fire on ``ENOENT`` -- a missing weight file IS a real RED, and a
  classifier that refuses everything is not a fix, it is an amnesty;
* every ``except -> EXIT_RED`` site in the module is either classified or
  declares itself excluded -- because #409's first pass covered ONE of four
  sites, a denominator narrower than its own claim, and the next site added
  will be the fifth.

Torch-free by construction: the behavioural arms inject at ``Step.START``
using ``test_train_boundary_is_total``'s idiom, so no model, dataset or profile
is touched; the census arm reads source text.
"""

from __future__ import annotations

import ast
import errno
import json
from pathlib import Path
from typing import Any

import pytest

from foundationscale.train import loop

_EXIT_RED = 5
_EXIT_REFUSE = 96

#: The sentence a handler writes to say "the classifier is deliberately absent
#: here". Kept as a constant so the census and the source cannot drift apart by
#: a reworded comment -- #83's axis, applied to prose.
_EXCLUSION_MARKER = "Deliberately NOT run through _environment_failure_reason"


def _cfg(tmp_path: Path) -> Any:
    """A valid config that is never used -- every arm dies on line one."""
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


def _raise_at_start(exc: BaseException) -> Any:
    """``_mark`` replacement that raises at START and forwards the rest.

    Forwarding matters here and not only for brevity: the REFUSE marker these
    arms assert on is emitted by the handler itself, so a replacement that
    swallowed later markers would make the arm pass without the verdict ever
    being announced.
    """
    real_mark = loop._mark

    def _mark(step: str, msg: str = "") -> None:
        if step == loop.Step.START:
            raise exc
        real_mark(step, msg)

    return _mark


# --------------------------------------------------------------------------
# Behavioural: the boundary classifies
# --------------------------------------------------------------------------


def test_enolck_at_the_boundary_refuses_rather_than_blaming_the_plane(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """The measured case: an NFS home with no locking is a 96, not a 5."""
    monkeypatch.setattr(
        loop,
        "_mark",
        _raise_at_start(OSError(errno.ENOLCK, "No locks available", "/home/u/.cache")),
    )
    cfg = _cfg(tmp_path)

    rc = loop.train(cfg)

    assert rc == _EXIT_REFUSE, (
        f"an ENOLCK escaping the thin path returned {rc}, not {_EXIT_REFUSE}. "
        "This is the exact fault measured on a GB200 estate, where it returned "
        "5 and four telemetry tests reported the training plane broken. The "
        "machine failed; the plane was never exercised, so there is nothing to "
        "score RED"
    )
    out = capsys.readouterr().out
    assert loop.Step.REFUSE in out, (
        f"exit was {_EXIT_REFUSE} but no {loop.Step.REFUSE!r} marker reached "
        "stdout, so a log reader sees a refusal with no named cause -- #312's "
        f"shape. stdout was: {out[-400:]!r}"
    )
    assert "no file locking" in out, (
        "the refusal does not say WHICH environment fault it found, so the "
        "operator cannot act on it. The whole value of 96 over 5 here is that "
        f"it names the fix. stdout was: {out[-400:]!r}"
    )
    manifest = Path(cfg.output_dir) / loop.MANIFEST_NAME
    assert manifest.exists(), (
        f"no manifest at {manifest}: a refused run must still leave a record, "
        "or the artifact cannot distinguish 'the machine refused' from 'the "
        "run never started' -- and those carry opposite follow-up actions"
    )
    record = json.loads(manifest.read_text())
    assert json.dumps(record).count(str(_EXIT_REFUSE)) >= 1, (
        f"the manifest records no {_EXIT_REFUSE}, so the artifact still reads "
        f"as the RED it used to be. Record: {json.dumps(record)[:400]!r}"
    )


def test_a_wrapped_environment_errno_is_still_an_environment_fault(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The arm that makes the fix worth having in the real world.

    HuggingFace does not hand an ``OSError`` to its caller; ``filelock``,
    ``datasets`` and ``transformers`` each raise their own type with the
    original attached as ``__cause__``. A classifier that judged only the
    outermost exception would be correct on this test suite and inert on every
    real estate -- which is precisely the failure mode #409 exists to name.
    """

    class _LibraryError(RuntimeError):
        """Stand-in for the wrapper types the hub libraries actually raise."""

    try:
        raise OSError(errno.ENOSPC, "No space left on device")
    except OSError as cause:
        wrapped: BaseException = _LibraryError("could not prepare the cache")
        wrapped.__cause__ = cause

    monkeypatch.setattr(loop, "_mark", _raise_at_start(wrapped))

    rc = loop.train(_cfg(tmp_path))

    assert rc == _EXIT_REFUSE, (
        f"a _LibraryError wrapping ENOSPC returned {rc}, not {_EXIT_REFUSE}. "
        "The classifier is reading only the outermost exception, so it will "
        "fire on the synthetic direct-OSError arm above and on nothing a real "
        "library ever raises"
    )


def test_enoent_is_still_red_because_a_missing_weight_file_is_a_real_failure(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The must-NOT-fire control. Without it, 'refuse everything' passes.

    ``ENOENT`` is an ``OSError`` like the others and is deliberately absent
    from the table: a weight file that is not there is the operator asking for
    something that does not exist, which is the definition of RED. An arm that
    only ever checks that refusals happen cannot tell a classifier from an
    amnesty.
    """
    monkeypatch.setattr(
        loop,
        "_mark",
        _raise_at_start(OSError(errno.ENOENT, "No such file or directory", "model.safetensors")),
    )

    rc = loop.train(_cfg(tmp_path))

    assert rc == _EXIT_RED, (
        f"a missing weight file returned {rc}, not {_EXIT_RED}. The #409 "
        "classifier has over-reached: it is now laundering real failures into "
        "refusals, which is worse than the defect it was written to fix -- a "
        "false RED is noisy, a false 96 is silent"
    )


# --------------------------------------------------------------------------
# Unit: the chain walk itself
# --------------------------------------------------------------------------


def test_a_self_referential_chain_terminates() -> None:
    """An exception raised while handling itself must not hang the walk.

    Cheap to write, and the alternative to writing it is a training job that
    spins forever inside its own error handler.
    """
    exc = RuntimeError("loops back on itself")
    exc.__cause__ = exc

    assert loop._environment_failure_reason(exc) is None, (
        "a self-referential exception chain was classified as an environment "
        "fault, which means the cycle guard is also deciding the verdict"
    )


def test_a_plain_value_error_is_not_an_environment_fault() -> None:
    """The other must-not-fire direction, at unit scope."""
    assert loop._environment_failure_reason(ValueError("model id does not exist")) is None, (
        "a ValueError was classified as an environment fault -- the table is "
        "being consulted for exceptions that carry no errno at all"
    )


# --------------------------------------------------------------------------
# Census: the fix must cover its own claim
# --------------------------------------------------------------------------


def _red_returning_handlers() -> list[tuple[int, str]]:
    """Every ``except`` handler in loop.py that can return ``EXIT_RED``.

    This is the denominator #409's first pass got wrong. The claim is about a
    CLASS of sites -- 'a caught exception becomes RED' -- and the first fix
    covered one of four members of it. Deriving the denominator from the source
    rather than from a hand-kept list is what stops the fifth site from being
    added unclassified and nobody noticing.

    Nested function definitions inside a handler are excluded: their returns
    belong to the inner function's contract, not to the handler's.
    """
    source = Path(loop.__file__).read_text()
    lines = source.splitlines()
    tree = ast.parse(source)
    found: list[tuple[int, str]] = []

    for handler in ast.walk(tree):
        if not isinstance(handler, ast.ExceptHandler):
            continue
        returns_red = False
        for node in ast.walk(handler):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)):
                continue
            if (
                isinstance(node, ast.Return)
                and isinstance(node.value, ast.Name)
                and node.value.id == "EXIT_RED"
            ):
                returns_red = True
                break
        if not returns_red:
            continue
        end = handler.end_lineno or handler.lineno
        found.append((handler.lineno, "\n".join(lines[handler.lineno - 1 : end])))
    return found


def _is_classified(body: str) -> bool:
    """Does this handler's source consult the classifier, or opt out by name?"""
    return "_environment_failure_reason" in body or _EXCLUSION_MARKER in body


def test_every_red_returning_handler_is_classified_or_declares_its_exclusion() -> None:
    """The class, not the instance -- #205's lesson applied to #409.

    A site that turns an exception into RED without asking whether it was the
    machine's fault is the defect. Pinning the three sites the first fix
    touched would leave the fourth silently wrong and the fifth free to be
    added, which is how #409 came to exist in the first place.
    """
    handlers = _red_returning_handlers()

    assert len(handlers) >= 4, (
        f"the census found only {len(handlers)} RED-returning handlers in "
        f"{Path(loop.__file__).name}, and the measured population at the time "
        "this was written was 4. A census that suddenly sees fewer sites is "
        "far more likely to be a broken extractor than a refactored module, "
        "and a broken extractor here reads as a clean bill of health"
    )

    unclassified = [line for line, body in handlers if not _is_classified(body)]
    assert not unclassified, (
        f"{len(unclassified)} of {len(handlers)} handlers that return EXIT_RED "
        f"neither consult _environment_failure_reason nor declare themselves "
        f"excluded -- at {Path(loop.__file__).name} lines {unclassified}. Each "
        "one turns a full disk, a read-only mount or an NFS home into a public "
        "accusation against the training plane. Either classify the site or "
        f"write {_EXCLUSION_MARKER!r} in the handler and say why it cannot see "
        "an environment errno"
    )


def test_the_census_can_fail() -> None:
    """The census's own positive control -- #positive-control-must-self-match.

    The extractor above is the instrument that certifies the class. Run it on a
    handler that is unambiguously unclassified and it must flag it; if it
    cannot, the green above means only that the extractor found nothing.
    """
    unclassified_body = 'except Exception as exc:\n    _mark(Step.RED, "boom")\n    return EXIT_RED'
    classified_body = (
        "except Exception as exc:\n"
        "    if _environment_failure_reason(exc) is not None:\n"
        "        return EXIT_REFUSE\n"
        "    return EXIT_RED"
    )

    assert not _is_classified(unclassified_body), (
        "the classification test passes a handler that does nothing of the "
        "kind, so it cannot distinguish a classified site from any other text "
        "-- the census above is vacuous"
    )
    assert _is_classified(classified_body), (
        "the classification test rejects a handler that plainly IS classified, "
        "so the census would report false unclassified sites and be switched "
        "off by the first person who read it"
    )
