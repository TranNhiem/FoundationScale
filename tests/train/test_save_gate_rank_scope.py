"""Rank scoping for the save gate: a checkpoint is adjudicated by the rank that wrote it (#444).

FoundationScaleSaveGate.on_save used to adjudicate on every rank. Under
torchrun/DDP, HF Trainer fires on_save on EVERY rank, but only the writing rank
emits model shards; every other rank's view of checkpoint-N holds config.json,
generation_config.json and its own rng_state_<rank>.pth and nothing else.
Adjudicating that view refused the precision check as vacuous -- a comparison
over nothing must not pass -- and set should_training_stop=True on a rank that
owned no artifact.

Measured on a 2-GPU GB200 tray: 4 of 4 arms, rank 0 printed "PASS 4/4 gates"
and "[fs:train:done] PASS" while rank 1 exited 5 and torchrun flattened the
tray to launcher rc 1. The artifact was never absent; it was never that rank's
to inspect.

The MUST-FIRE leg reproduces the pre-fix behaviour by calling _adjudicate
directly over the exact bytes rank 1 saw, so the fix is attributable and not a
coincidence: if the abstention legs pass while the must-fire leg does not, the
fixture has drifted and the guard is being credited for something it did not do.
"""

from __future__ import annotations

import types
from pathlib import Path
from typing import NoReturn

from foundationscale.train.loop import (
    FoundationScaleSaveGate,
    Lifecycle,
    _agree_on_stop,
    _wrote_this_checkpoint,
)

_STEP = 7


def _rank1_view(tmp_path: Path, step: int) -> Path:
    """Build the exact directory rank 1 saw on the tray; return the output_dir.

    The gate composes ``output_dir / f"checkpoint-{step}"`` itself, so the
    parent is returned, not the checkpoint. The checkpoint holds ONLY the three
    files a non-writing rank holds under DDP: two configs and its own RNG
    state. No shards -- the shards were never this rank's to write.
    """
    ckpt = tmp_path / f"checkpoint-{step}"
    ckpt.mkdir()
    (ckpt / "config.json").write_text("{}")
    (ckpt / "generation_config.json").write_text("{}")
    (ckpt / "rng_state_1.pth").write_bytes(b"\xde\xad\xbe\xef" * 4)
    return tmp_path


def _args(output_dir: Path | str, **kw: object) -> types.SimpleNamespace:
    # An attribute must be ABSENT unless passed: that is what exercises the
    # None arm of _wrote_this_checkpoint, and a SimpleNamespace with a default
    # would silently answer a question the test means to leave open.
    return types.SimpleNamespace(output_dir=str(output_dir), **kw)


def _state(global_step: int, **kw: object) -> types.SimpleNamespace:
    return types.SimpleNamespace(global_step=global_step, **kw)


def _control() -> types.SimpleNamespace:
    return types.SimpleNamespace(should_training_stop=False)


def _unbuildable_context(ckpt_dir: Path) -> NoReturn:
    # Mimics "no recognizable checkpoint layout": the rank-1 view carries no
    # artifact family a context builder could decode.
    raise ValueError(f"no recognizable checkpoint layout under {ckpt_dir}")


def test_prefix_idiom_reds_a_non_writing_rank(tmp_path: Path) -> None:
    """Measure the pre-fix behaviour: the rank-1 view, adjudicated, REDs the run."""
    output_dir = _rank1_view(tmp_path, _STEP)
    gate = FoundationScaleSaveGate(context_builder=_unbuildable_context, declared_precision="bf16")
    # MUST FIRE: the old on_save adjudicated unconditionally, so calling
    # _adjudicate directly over the rank-1 view IS what rank 1 did on the
    # tray. If this leg ever stops firing, the fixture has drifted from the
    # bytes the tray produced and every other leg here is measuring something
    # that never happened.
    stop = gate._adjudicate(Lifecycle.FIRST_SAVE, output_dir / f"checkpoint-{_STEP}")
    assert stop is True, (
        "MUST FIRE: adjudicating the rank-1 view must stop the run -- this is the "
        "pre-fix behaviour, and without it the fix cannot be attributed"
    )
    agreement = gate.precision_agreement
    assert agreement is not None, "the first save must record a precision agreement"
    assert agreement.status == "refuse", (
        "a comparison over zero shards must refuse as vacuous, got "
        f"{agreement.status!r} -- a comparison over nothing must not pass"
    )


def test_a_non_writing_rank_abstains_instead_of_reding(tmp_path: Path) -> None:
    """Measure that a non-writing rank abstains: no block, no stop."""
    output_dir = _rank1_view(tmp_path, _STEP)
    gate = FoundationScaleSaveGate(context_builder=_unbuildable_context, declared_precision="bf16")
    control = _control()
    gate.on_save(_args(output_dir, should_save=False), _state(_STEP), control)
    assert gate.blocked is False, (
        "a rank that wrote nothing holds no artifact to adjudicate; it must not "
        "block on a view that was never its own"
    )
    assert control.should_training_stop is False, (
        "the abstaining rank must not stop the run -- rank 1 exiting 5 while "
        "rank 0 printed PASS is the exact failure being controlled"
    )


def test_the_abstention_is_declared_not_silent(tmp_path: Path) -> None:
    """Measure that the abstention leaves one declared record, not silence."""
    output_dir = _rank1_view(tmp_path, _STEP)
    gate = FoundationScaleSaveGate(context_builder=_unbuildable_context, declared_precision="bf16")
    gate.on_save(_args(output_dir, should_save=False), _state(_STEP), _control())
    assert len(gate.records) == 1, (
        f"an abstention must leave exactly one record, found {len(gate.records)}: {gate.records}"
    )
    record = gate.records[0]
    assert record["event"].endswith(".abstain"), (
        f"the record must be declared an abstention, got event={record['event']!r}"
    )
    assert record["verdicts"] == {}, (
        f"a rank that established nothing must record no verdicts, got {record['verdicts']}"
    )
    assert "writing rank" in record["reason"], (
        "the reason must name the writing rank as the owner of the verdict, got "
        f"{record['reason']!r} -- an abstention that leaves no record is "
        "indistinguishable from a gate that ran and liked what it saw"
    )


def test_a_non_writing_rank_does_not_touch_the_precision_verdict(tmp_path: Path) -> None:
    """Measure that abstention never reaches the precision check at all."""
    output_dir = _rank1_view(tmp_path, _STEP)
    gate = FoundationScaleSaveGate(context_builder=_unbuildable_context, declared_precision="bf16")
    gate.on_save(_args(output_dir, should_save=False), _state(_STEP), _control())
    assert gate.precision_agreement is None, (
        "the abstaining rank must not touch the precision verdict: the refusal "
        "over an empty view was the false RED this fix removes"
    )


def test_the_writing_rank_still_reds_a_bad_checkpoint(tmp_path: Path) -> None:
    """Regression guard: the writing rank still REDs the same bad checkpoint."""
    output_dir = _rank1_view(tmp_path, _STEP)
    gate = FoundationScaleSaveGate(context_builder=_unbuildable_context, declared_precision="bf16")
    control = _control()
    gate.on_save(_args(output_dir, should_save=True), _state(_STEP), control)
    assert gate.blocked is True, (
        "REGRESSION GUARD: the writing rank owns the artifact and must still RED "
        "a bad checkpoint -- the fix scopes adjudication, it does not disable it"
    )
    assert control.should_training_stop is True, (
        "a blocking verdict on the writing rank must set should_training_stop"
    )


def test_should_save_outranks_is_world_process_zero() -> None:
    """Measure that should_save wins over is_world_process_zero when they disagree."""
    wrote = _wrote_this_checkpoint(
        _args(".", should_save=True), _state(0, is_world_process_zero=False)
    )
    assert wrote is True, (
        "should_save=True must outrank is_world_process_zero=False -- it is the "
        "same predicate the Trainer itself consults before writing"
    )
    wrote = _wrote_this_checkpoint(
        _args(".", should_save=False), _state(0, is_world_process_zero=True)
    )
    assert wrote is False, (
        "should_save=False must outrank is_world_process_zero=True -- two oracles "
        "for one question would drift the first time save_on_each_node changed"
    )


def test_is_world_process_zero_is_the_fallback() -> None:
    """Measure the fallback: is_world_process_zero answers when should_save is absent."""
    wrote = _wrote_this_checkpoint(_args("."), _state(0, is_world_process_zero=True))
    assert wrote is True, (
        "with should_save absent, is_world_process_zero=True must answer the "
        "rank question for a Trainer old enough not to expose the property"
    )
    wrote = _wrote_this_checkpoint(_args("."), _state(0, is_world_process_zero=False))
    assert wrote is False, (
        "with should_save absent, is_world_process_zero=False must answer the "
        "rank question for a Trainer old enough not to expose the property"
    )


def test_an_unanswerable_rank_question_adjudicates(tmp_path: Path) -> None:
    """Measure that an unanswerable rank question adjudicates rather than refuses to look."""
    output_dir = _rank1_view(tmp_path, _STEP)
    wrote = _wrote_this_checkpoint(_args(output_dir), _state(_STEP))
    assert wrote is None, (
        "with neither attribute present the question cannot be answered and must "
        f"come back None, got {wrote!r}"
    )
    gate = FoundationScaleSaveGate(context_builder=_unbuildable_context, declared_precision="bf16")
    control = _control()
    gate.on_save(_args(output_dir), _state(_STEP), control)
    assert gate.blocked is True, (
        "None must adjudicate: refusing to look at the only checkpoint there is "
        "would be worse than the risk the guard removes, and single-process runs "
        "must keep working"
    )


def test_agree_on_stop_is_identity_without_a_process_group() -> None:
    """Measure that the stop agreement is the identity with no process group."""
    # Under pytest no process group is initialised, so the collective must be
    # skipped, not attempted: attempting it would raise, and a rank that
    # cannot reach its peers has not produced a verdict about the model.
    assert _agree_on_stop(True) is True, (
        "with no process group the local stop must pass through unchanged"
    )
    assert _agree_on_stop(False) is False, (
        "with no process group the local continue must pass through unchanged"
    )


def test_an_unbuildable_context_does_not_retract_a_precision_verdict(tmp_path: Path) -> None:
    """Measure that an unbuildable context cannot retract a refusal already reached."""
    output_dir = _rank1_view(tmp_path, _STEP)
    gate = FoundationScaleSaveGate(context_builder=_unbuildable_context, declared_precision="bf16")
    stop = gate._adjudicate(Lifecycle.FIRST_SAVE, output_dir / f"checkpoint-{_STEP}")
    assert stop is True, (
        "a context that cannot be built is not a retraction of a refusal already "
        "reached above it -- _adjudicate must return the precision verdict, "
        "not False"
    )


def test_on_save_always_returns_the_control_object(tmp_path: Path) -> None:
    """Measure that on_save returns the control object it was handed, on both paths."""
    output_dir = _rank1_view(tmp_path, _STEP)
    control = _control()
    abstaining = FoundationScaleSaveGate(
        context_builder=_unbuildable_context, declared_precision="bf16"
    )
    returned = abstaining.on_save(_args(output_dir, should_save=False), _state(_STEP), control)
    assert returned is control, (
        "the abstaining path must return the very control object passed in -- "
        "the Trainer keeps its own per-process control, not a copy"
    )
    writing_control = _control()
    writing = FoundationScaleSaveGate(
        context_builder=_unbuildable_context, declared_precision="bf16"
    )
    returned = writing.on_save(_args(output_dir, should_save=True), _state(_STEP), writing_control)
    assert returned is writing_control, (
        "the adjudicating path must return the very control object passed in -- "
        "the Trainer keeps its own per-process control, not a copy"
    )
