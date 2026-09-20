"""Controls for #441: an arm failure must be diagnosable from its own receipt.

The MUST-FIRE fixture in this module is not synthetic. ``P439_STDOUT`` and
``P439_STDERR`` are trimmed transcripts of a real T1-9 arm that failed on a
GB200 tray during pass p439, reproduced by hand with the streams inherited
rather than captured. Against the pre-fix idiom -- ``proc.stderr`` only, and
``lines[-1]`` of it -- these exact bytes produce the string

    last line: ============================================================

which is what every one of p439's 29 arm records actually said. A control that
cannot reproduce that is not measuring the defect, so
``test_prefix_idiom_is_blind_to_the_refusal`` asserts the OLD behaviour
directly: it must keep passing, because it is the evidence that the new
behaviour is a change and not a coincidence.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

_MATRIX = Path(__file__).resolve().parents[2] / "validation_campaigns" / "verification_matrix"


def _load_arm_diagnosis():
    """Import the helper by path.

    validation_campaigns/ is not a package and is not on sys.path under pytest
    (#352 is the same boundary), so a plain import would resolve to nothing and
    the whole module would error at collection rather than fail a leg.
    """
    path = _MATRIX / "arm_diagnosis.py"
    spec = importlib.util.spec_from_file_location("fs_arm_diagnosis", path)
    assert spec is not None and spec.loader is not None, f"cannot load {path}"
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


ad = _load_arm_diagnosis()


# The refusal line the trainer printed. Verbatim shape from the tray, truncated
# in the middle only where the enum listing repeats.
REFUSE_LINE = (
    "[fs:train:refuse]        transformers 5.13.0 rejected the declared config "
    "at Trainer/TrainingArguments construction: adamw is not a valid "
    "OptimizerNames, please select one of ['adamw_torch', 'adamw_torch_fused', "
    "'adafactor', 'sgd']. Refusing (96) rather than retrying with a guessed "
    "value -- the operator declared it, so the operator corrects it"
)

P439_STDOUT = "\n".join(
    [
        "[fs:train:deps]          importing torch/transformers/datasets",
        "[fs:train:consistency]   precision='bf16' bound at load and verified",
        "[fs:train:data]          120000 examples tokenized (split=train)",
        "[fs:train:manifest]      declared checkpoint: dense",
        REFUSE_LINE,
        "[fs:train:manifest]      run manifest (refused) -> /tmp/x/run_manifest.json",
    ]
)

# The torchrun banner. Note the closing rule: this is the line the pre-fix
# idiom kept, on every arm, in every pass.
P439_STDERR = "\n".join(
    [
        "W0914 11:27:41 api.py:1014] Sending process 1684444 closing signal SIGTERM",
        "E0914 11:27:41 api.py:988] failed (exitcode: 96) local_rank: 1",
        "Traceback (most recent call last):",
        '  File "<frozen runpy>", line 198, in _run_module_as_main',
        "    raise ChildFailedError(",
        "torch.distributed.elastic.multiprocessing.errors.ChildFailedError: ",
        "============================================================",
        "foundationscale.train.cli FAILED",
        "------------------------------------------------------------",
        "Root Cause (first observed failure):",
        "[0]:",
        "  time      : 2026-09-14_11:27:41",
        "  rank      : 1 (local_rank: 1)",
        "  exitcode  : 96 (pid: 1684445)",
        "  error_file: <N/A>",
        "============================================================",
    ]
)

PREFIX = "torchrun launcher exited 1 (the launcher's code, #171)"


def _pre_fix_reason(stdout: str, stderr: str) -> str:
    """The idiom t1_9 shipped before #441, reproduced exactly."""
    lines = stderr.strip().splitlines()
    if not lines:
        lines = stdout.strip().splitlines()
    return f"{PREFIX}; last line: {lines[-1][:160]}" if lines else PREFIX


def test_prefix_idiom_is_blind_to_the_refusal() -> None:
    """MUST FIRE: the shipped idiom reports the banner and loses the refusal.

    This is the control. If this leg ever fails, the fixture has drifted away
    from the bytes p439 actually produced and every other leg here is measuring
    something that never happened.
    """
    reason = _pre_fix_reason(P439_STDOUT, P439_STDERR)
    assert reason.endswith("=" * 60), reason
    assert "refuse" not in reason
    assert "OptimizerNames" not in reason


def test_diagnose_surfaces_the_declared_refusal() -> None:
    reason, _ = ad.diagnose(1, P439_STDOUT, P439_STDERR, prefix=PREFIX)
    assert "[fs:train:refuse]" in reason
    assert "OptimizerNames" in reason
    # The #171 framing survives: the record must still say the code is the
    # launcher's, not the trainer's declared verdict.
    assert reason.startswith(PREFIX)


def test_stdout_is_searched_even_when_stderr_is_full() -> None:
    """The whole defect in one leg.

    Every marker is printed to stdout; stderr is never empty under torchrun.
    A fallback of "use stdout only if stderr is empty" therefore never fires.
    """
    assert P439_STDERR.strip(), "fixture precondition: stderr must be non-empty"
    assert "[fs:train:" not in P439_STDERR, "fixture precondition: markers are on stdout"
    reason, _ = ad.diagnose(1, P439_STDOUT, P439_STDERR, prefix=PREFIX)
    assert "OptimizerNames" in reason


def test_both_stream_excerpts_are_persisted() -> None:
    _, excerpts = ad.diagnose(1, P439_STDOUT, P439_STDERR, prefix=PREFIX)
    assert set(excerpts) == {"stdout_tail", "stderr_tail"}
    assert "[fs:train:refuse]" in excerpts["stdout_tail"]
    assert "ChildFailedError" in excerpts["stderr_tail"]


def test_excerpts_are_bounded() -> None:
    flood = "\n".join(f"line {i}" for i in range(5000))
    _, excerpts = ad.diagnose(1, flood, flood, prefix=PREFIX)
    for key, value in excerpts.items():
        assert len(value.splitlines()) <= ad.EXCERPT_LINES, key
        assert len(value) <= ad.EXCERPT_CHARS + 3, key  # +3 for the "..." marker


def test_no_marker_falls_back_to_the_last_speaking_line() -> None:
    """A crash with no declared marker still has to name something.

    The furniture filter is what makes this useful: the last LINE of a torchrun
    banner is a rule, so "last line" and "last line that says something" are
    different answers, and only the second one is a diagnosis.
    """
    stderr = "RuntimeError: CUDA out of memory\n" + "=" * 60
    reason, _ = ad.diagnose(1, "", stderr, prefix=PREFIX)
    assert "CUDA out of memory" in reason
    assert "no [fs:train:*] marker" in reason


def test_empty_streams_are_named_as_such_not_silently_dropped() -> None:
    reason, excerpts = ad.diagnose(1, "", "", prefix=PREFIX)
    assert "both streams are empty" in reason
    assert "environment fault, not a verdict" in reason
    assert excerpts == {"stdout_tail": "", "stderr_tail": ""}


def test_a_signalled_arm_names_the_signal() -> None:
    """The OOM-killer case: no output at all, and that IS the diagnosis.

    subprocess reports a signalled child as -N. Without the signal name the
    record says only "both streams are empty", which reads as an instrument
    fault; with it, the record names the thing that happened.
    """
    reason, _ = ad.diagnose(-9, "", "", prefix=PREFIX)
    assert "[killed by SIGKILL]" in reason
    assert "both streams are empty" in reason


def test_the_signal_note_survives_a_declared_marker() -> None:
    reason, _ = ad.diagnose(-15, P439_STDOUT, P439_STDERR, prefix=PREFIX)
    assert "[killed by SIGTERM]" in reason
    assert "OptimizerNames" in reason


def test_a_real_exit_code_gets_no_signal_note() -> None:
    assert ad.signal_note(1) == ""
    assert ad.signal_note(96) == ""
    assert ad.signal_note(0) == ""


def test_an_unnameable_signal_is_still_reported_as_one() -> None:
    """Do not let an unknown signal number vanish into a bare ValueError."""
    assert ad.signal_note(-9999) == " [killed by signal 9999]"


@pytest.mark.parametrize(
    "line",
    ["", "   ", "=" * 60, "-" * 40, "***", "~~~~"],
)
def test_furniture_is_recognised(line: str) -> None:
    assert ad._is_furniture(line)


@pytest.mark.parametrize(
    "line",
    ["RuntimeError: boom", "  rank      : 1", "=== not a pure rule ==="],
)
def test_content_is_not_furniture(line: str) -> None:
    assert not ad._is_furniture(line)


def test_marker_vocabulary_matches_the_trainer() -> None:
    """The duplicated marker list must not drift from Step.

    arm_diagnosis deliberately hard-codes the marker strings so it stays
    importable without the package (a failing arm may be failing BECAUSE the
    package will not import). That duplication is only safe if something pins
    it, and this is that pin.
    """
    from foundationscale.train.loop import MARKERS

    # MARKERS, not vars(Step): loop.py derives it as THE marker denominator and
    # excludes dunders. Re-deriving here with a bare vars() scan admits __doc__
    # and __module__ -- both strings -- which a one-directional check cannot
    # feel, but which made the exhaustive test below fail on junk until this
    # was pointed at the denominator the trainer already publishes.
    declared = {f"[{value}]" for value in MARKERS}
    for marker in ad.DIAGNOSTIC_MARKERS:
        assert marker in declared, f"{marker} is not a Step the trainer emits"


# Every Step the trainer declares, that is NOT part of the refusal/abort
# vocabulary. Spelled out rather than derived, because the split is a judgment
# about meaning that no rule over the strings can make for us.
_PROGRESS_ONLY_MARKERS = frozenset(
    {
        "[fs:train:start]",
        "[fs:train:topology]",
        "[fs:train:profile]",
        "[fs:train:consistency]",
        "[fs:train:partition]",
        "[fs:train:validated]",
        "[fs:train:deps]",
        "[fs:train:data]",
        # Both are REPORTS, not verdicts, and each is here for a stated reason
        # rather than by default. host cannot fail a run at all -- it prints the
        # CPU budget and returns. fabric CAN end a run, but when it does the
        # trainer emits [fs:train:refuse] alongside it, so the death is named in
        # the vocabulary arm_diagnosis already reads; the fabric lines themselves
        # are the per-attempt evidence under that verdict.
        "[fs:train:host]",
        "[fs:train:fabric]",
        "[fs:train:trainer]",
        "[fs:train:adapter]",
        "[fs:train:run]",
        "[fs:train:saved]",
        "[fs:train:save_gate]",
        "[fs:train:objective_gate]",
        "[fs:train:manifest]",
        "[fs:train:adjudicate]",
        "[fs:train:done]",
    }
)


def test_every_trainer_step_is_classified() -> None:
    """The partition over Step must be exhaustive, so a NEW step cannot hide.

    The test above pins one direction only -- every marker is a real Step. The
    direction it cannot see is the dangerous one: a refusal Step added to the
    trainer and never added to DIAGNOSTIC_MARKERS. Nothing would fail, and an
    arm dying that way would reach the operator with no diagnosis at all, from
    the one module that exists to supply it. The same one-sided-detector shape
    that made the mutation instrument report 46 false hits until it was made to
    test both directions.

    Asserting the two buckets EXACTLY cover Step turns that silence into a
    failure here: a new step lands in neither set and this test names it. The
    fix is never to delete the line -- it is to decide which bucket the new
    step belongs to, which is the decision that was being skipped.
    """
    from foundationscale.train.loop import MARKERS

    declared = {f"[{value}]" for value in MARKERS}
    classified = set(ad.DIAGNOSTIC_MARKERS) | _PROGRESS_ONLY_MARKERS

    unclassified = declared - classified
    assert not unclassified, (
        f"{sorted(unclassified)} is emitted by the trainer but classified neither as a "
        "refusal marker (arm_diagnosis.DIAGNOSTIC_MARKERS) nor as progress "
        "(_PROGRESS_ONLY_MARKERS). Decide which it is: if an arm can DIE at this step, "
        "it belongs in DIAGNOSTIC_MARKERS or arm_diagnosis cannot name that death."
    )
    stale = classified - declared
    assert not stale, f"{sorted(stale)} is classified here but the trainer no longer emits it"
    assert not (set(ad.DIAGNOSTIC_MARKERS) & _PROGRESS_ONLY_MARKERS), (
        "a marker cannot be both a refusal and progress"
    )
