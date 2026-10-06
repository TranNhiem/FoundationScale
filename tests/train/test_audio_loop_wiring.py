"""Speech-plane wiring for the audio release: declaration conflicts,
dormant-modality towers, dangling placeholders, and the processor surface.

Four refus/announce/resolve decisions govern the audio arm, each decided before
the model or the corpus is touched so the refusal costs nothing and cannot be
confounded by an unloadable input.  The declaration conflict gate
(``_audio_declaration_conflict``) decides which audio shapes run at all; the
tower walker (``_dormant_modality_towers``) reports which modality towers carry
weights no declaration exercises; the placeholder scan
(``_unresolved_placeholder_notice``) names markers left dangling in the
text-only supervision; and ``resolve_audio_surface`` is the only path to a
processor surface -- no tokenizer fallback is permitted.

MUST_PASS / MUST_FIRE pairs state both halves of every decision: a guard that
only fires is as useless as one that never does, and a confirmation that only
ever echoes the clean shape never proves the guard can speak.  The fake omni
family and model below state the tower layout explicitly rather than depending
on a registry entry whose edit could silently invalidate the assertion.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from foundationscale.train.audio import resolve_audio_surface
from foundationscale.train.loop import (
    _audio_declaration_conflict,
    _dormant_modality_towers,
    _unresolved_placeholder_notice,
)

_AUDIO_ROW = "Q: <audio>\nDescribe the mechanics.\nA: The lever opens the vent."
_VIDEO_ROW = "Q: <video>\nWhich SOP step?\nA: Step 3."
_CLEAN_ROW = "Q: What year was the cave opened?\nA: 1974."

# An omni fake family: vision + audio towers in the same declaration order as
# gemma4, stated here so a registry edit cannot change what these tests measure.
_OMNI = SimpleNamespace(
    towers=[
        ("model.vision_tower", "image"),
        ("model.audio_tower", "audio"),
    ],
)

# An omni fake model: both towers PRESENT on the loaded module tree.
_OMNI_MODEL = SimpleNamespace(
    model=SimpleNamespace(vision_tower=object(), audio_tower=object()),
)


# ---------------------------------------------------------------------------
# 1. _audio_declaration_conflict
# ---------------------------------------------------------------------------


def test_undeclared_audio_never_conflicts_regardless_of_image_or_cp() -> None:
    """MUST_PASS. If this fails, a text-only run would be refused on an audio
    axis it never named -- silently disabling the simplest baseline."""
    for image_column, cp in [(None, 1), ("img", 1), (None, 8), ("img", 8)]:
        result = _audio_declaration_conflict(audio_column=None, image_column=image_column, cp=cp)
        assert result is None, f"audio=None with image={image_column!r} cp={cp} must not conflict"


def test_audio_alone_with_cp_1_is_clean() -> None:
    """MUST_PASS. The only audio shape this release supports must NOT be
    refused -- volume graphs need a faithful measure, not a paranoid one."""
    assert _audio_declaration_conflict(audio_column="audio", image_column=None, cp=1) is None


def test_audio_and_image_names_both_columns_and_the_policy() -> None:
    """MUST_FIRE. The two collators each own the whole batch; naming both
    columns is the difference between a diagnosis and a shrug."""
    refusal = _audio_declaration_conflict(audio_column="waveforms", image_column="pixels", cp=1)
    assert refusal is not None
    assert "'waveforms'" in refusal
    assert "'pixels'" in refusal
    assert "One non-text modality" in refusal


def test_audio_with_cp_gt_1_names_the_division_that_fails() -> None:
    """MUST_FIRE. Context parallelism cannot split a sequence that does not
    divide; naming both cp and 2*cp gives the operator the arithmetic."""
    refusal = _audio_declaration_conflict(audio_column="audio", image_column=None, cp=2)
    assert refusal is not None
    assert "cp=2" in refusal
    assert "2*cp=4" in refusal


def test_image_conflict_takes_precedence_over_cp_conflict() -> None:
    """MUST_FIRE against image, MUST_PASS that cp does NOT shadow it.  When
    both conditions hold the modality conflict is the deeper problem -- fixing
    the cp too still leaves the collators fighting over the batch.  The image
    text must win; the cp text must not appear."""
    refusal = _audio_declaration_conflict(audio_column="audio", image_column="image", cp=2)
    assert refusal is not None
    assert "One non-text modality" in refusal
    assert "2*cp" not in refusal


# ---------------------------------------------------------------------------
# 2. _dormant_modality_towers -- audio_declared on an omni fake family
# ---------------------------------------------------------------------------


def test_audio_alone_dormants_only_the_vision_tower() -> None:
    """MUST_FIRE for vision.  If this fails, a speech-only run on an omni model
    would either miss the frozen vision tower or also flag the exercised audio
    tower -- both wrong, and only the announcement would have told."""
    result = _dormant_modality_towers(
        _OMNI_MODEL, family=_OMNI, image_declared=False, audio_declared=True
    )
    assert result == ["model.vision_tower"]


def test_both_modalities_declared_exercises_everything() -> None:
    """MUST_PASS.  If this fails, a fully-declared run would announce phantom
    dormant towers and train operators to ignore the real announcement."""
    result = _dormant_modality_towers(
        _OMNI_MODEL, family=_OMNI, image_declared=True, audio_declared=True
    )
    assert result == []


def test_neither_declared_reports_both_in_declaration_order() -> None:
    """MUST_FIRE for both towers.  Unchanged behaviour: a text-only run on an
    omni model takes both towers dormant and names them in declaration order."""
    result = _dormant_modality_towers(
        _OMNI_MODEL, family=_OMNI, image_declared=False, audio_declared=False
    )
    assert result == ["model.vision_tower", "model.audio_tower"]


def test_default_audio_declared_false_keeps_old_behaviour() -> None:
    """MUST_PASS against the regression.  Callers that do not yet pass
    ``audio_declared`` must see the same output they always did -- audio towers
    dormant when the parameter is absent."""
    result = _dormant_modality_towers(_OMNI_MODEL, family=_OMNI, image_declared=False)
    assert result == ["model.vision_tower", "model.audio_tower"]


# ---------------------------------------------------------------------------
# 3. _unresolved_placeholder_notice -- audio placeholders in text
# ---------------------------------------------------------------------------


def test_declared_audio_resolves_its_own_placeholder() -> None:
    """MUST_PASS.  A declared audio column loads waveforms for '<audio>' so the
    marker is resolved -- reporting it would contradict the arm assembling the
    batch right beside it."""
    notice = _unresolved_placeholder_notice([_AUDIO_ROW], image_declared=False, audio_declared=True)
    assert notice is None


def test_undeclared_audio_stays_reportable() -> None:
    """MUST_FIRE.  If this fails, a text-only arm training on '<audio>' rows
    would silently supervise answers beside prompts the model never received --
    the #506 silence, now on the audio axis."""
    notice = _unresolved_placeholder_notice(
        [_AUDIO_ROW], image_declared=False, audio_declared=False
    )
    assert notice is not None
    assert "<audio>" in notice
    assert "audio" in notice


def test_image_and_audio_declared_leaves_only_video_unresolved() -> None:
    """MUST_FIRE for video only.  Each placeholder is resolved solely by its own
    column; declaring audio must not silence the video check or vice versa."""
    notice = _unresolved_placeholder_notice([_VIDEO_ROW], image_declared=True, audio_declared=True)
    assert notice is not None
    assert "<video>" in notice
    assert "<image>" not in notice
    assert "<audio>" not in notice


# ---------------------------------------------------------------------------
# 4. resolve_audio_surface -- the processor surface and its refusal
# ---------------------------------------------------------------------------


def test_resolve_audio_surface_refuses_96_when_processor_fails(monkeypatch, capsys) -> None:
    """MUST_FIRE (refusal).  A tokenizer fallback would encode the text and
    drop the waveform with no signal; the only honest answer when the processor
    is unloadable is SystemExit(96) with a stderr line naming the axis."""
    import transformers

    def _boom(model_id: str) -> None:
        raise RuntimeError("no processor available for " + model_id)

    monkeypatch.setattr(transformers.AutoProcessor, "from_pretrained", _boom)

    with pytest.raises(SystemExit) as exc_info:
        resolve_audio_surface("fake/speech-model")

    assert exc_info.value.code == 96
    captured = capsys.readouterr()
    assert "REFUSAL (exit 96):" in captured.err
    assert "FOUNDATIONSCALE_TRAIN_AUDIO_COLUMN" in captured.err
    assert "'fake/speech-model'" in captured.err


def test_resolve_audio_surface_loads_processor_surface(monkeypatch) -> None:
    """MUST_PASS (resolution).  A working processor is the only acceptable
    surface for a declared audio column -- kind must say 'processor' so
    downstream code cannot mistake it for a tokenizer."""
    import transformers

    fake_processor = SimpleNamespace()

    def _load(model_id: str) -> SimpleNamespace:
        return fake_processor

    monkeypatch.setattr(transformers.AutoProcessor, "from_pretrained", _load)

    surface = resolve_audio_surface("fake/speech-model")

    assert surface.kind == "processor"
    assert surface.supports_images is False
    assert "corpus carries audio" in surface.reason
    assert surface.surface is fake_processor
