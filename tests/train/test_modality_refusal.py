"""#490: a declared modality this plane cannot train must refuse, and an
ordinary column drop must stop being silent.

T1-22 and T1-23 both claimed "declaring <modality> REFUSES cleanly today" and
both were measured RED on a tray. The video arm carried a usable `text` column
AND a `video` field, trained on the text, exited 0, and named the video nowhere
-- not in the console, not in 30,671 characters of run manifest. The audio arm
did exit 96, but its message was "the thin path requires a 'text' column": a
corpus holding nothing but a timestamp earns those identical words, so the exit
code was right for a reason that had nothing to do with sound.

Two releases of a refusal that was never implemented, and no test anywhere to
notice -- because reaching the equivalent image branch (#410) costs a torch
import, a model, a tokenizer and a loadable corpus. That is why the decision and
the wording are pure functions now, and why these run under a bare interpreter.

THE CONTROLS ARE THE POINT, and there are three kinds:

  keyed on the declaration    a corpus that merely CONTAINS a column named
                              "video" must NOT refuse. Refusing on the name
                              would be name-sniffing: `label` is dropped from
                              every text corpus correctly, and a text field
                              called "video" is a legitimate corpus. This is
                              also what makes opting out free, which is the
                              whole reason no "train anyway" flag exists.
  silence is a decision       an undeclared column IS still named on the
                              console. Without this the fix could be "refuse
                              more", which trades a silent drop for a refused
                              run that used to work.
  the wording is the claim    T1-23's original exit code was correct and its
                              message was about the wrong thing, so an
                              assertion on the exit alone would have passed the
                              defect. The message must name the modality.
"""

from __future__ import annotations

import pytest

from foundationscale.train import loop

AUDIO_VAR = "FOUNDATIONSCALE_TRAIN_AUDIO_COLUMN"
VIDEO_VAR = "FOUNDATIONSCALE_TRAIN_VIDEO_COLUMN"


# --- the declaration detector ------------------------------------------------


@pytest.mark.parametrize(
    ("var", "modality"),
    [(AUDIO_VAR, "audio"), (VIDEO_VAR, "video")],
)
def test_declared_modality_is_detected(var: str, modality: str) -> None:
    """Declaring either axis is seen, and reported with the variable that did it."""
    got = loop._declared_untrainable_modality({var: "waveform"})
    assert got is not None, f"{var} was declared and nothing noticed -- the #490 defect"
    assert got == (modality, var, "waveform")


def test_nothing_declared_is_silent() -> None:
    """The control for the two above: an ordinary run must not trip the refusal."""
    assert loop._declared_untrainable_modality({}) is None
    assert loop._declared_untrainable_modality({"HOME": "/root", "LANG": "C"}) is None


def test_a_corpus_column_named_video_does_not_refuse() -> None:
    """Keyed on the DECLARATION, never on the data.

    This is the control that pins the design decision. A dataset whose columns
    happen to include "video" -- a URL string, a title, a boolean flag -- is a
    legitimate text corpus, and refusing it would be name-sniffing dressed up
    as rigour. It is also what makes opting out of the refusal free: drop the
    declaration, keep the column, train on the text.
    """
    notice = loop._dropped_column_notice(["text", "video", "audio", "label"])
    assert notice is not None and "video" in notice, "the drop must be named"
    assert loop._declared_untrainable_modality({}) is None, (
        "a column named 'video' in the data, with nothing declared, must not refuse"
    )


def test_an_empty_declaration_reads_as_undeclared() -> None:
    """An exported-but-blank variable meant nothing by it, matching #410's rule.

    Shells export empty variables constantly (`FOO=` in a sourced env file, an
    unset substitution in a launcher). Treating that as a declaration would
    refuse runs whose author never mentioned sound.
    """
    assert loop._declared_untrainable_modality({AUDIO_VAR: ""}) is None


def test_both_declared_reports_a_fixed_modality_first() -> None:
    """Order is arbitrary but FIXED, so the refusal text is reproducible.

    A refusal whose wording depends on dict iteration order is a refusal that
    cannot be asserted on, and this file asserts on wording for a reason.
    """
    got = loop._declared_untrainable_modality({AUDIO_VAR: "a", VIDEO_VAR: "v"})
    assert got is not None
    assert got[0] == "audio"


# --- the wording, which is half of what was wrong ----------------------------


@pytest.mark.parametrize("modality", ["audio", "video"])
def test_refusal_names_the_modality_and_the_way_out(modality: str) -> None:
    """T1-23's RED was a correct exit code attached to the wrong explanation."""
    var = AUDIO_VAR if modality == "audio" else VIDEO_VAR
    msg = loop._untrainable_modality_refusal(modality, var, "clip")
    assert modality in msg, "a refusal that does not name the modality is T1-23's defect again"
    assert var in msg, "the operator needs to know which variable to unset"
    assert "clip" in msg, "name the declared column, so a typo is visible as a typo"
    assert "text" in msg, "say what the plane CAN do, or the refusal reads as a crash"


@pytest.mark.parametrize(
    ("modality", "want"),
    [("audio", "an audio label"), ("video", "a video label")],
)
def test_the_article_agrees_with_the_modality(modality: str, want: str) -> None:
    """The first tray run printed "under a audio label".

    Trivial, and shipped anyway, because this string is the product surface of
    the refusal -- it is what an operator reads and what the row's evidence
    quotes. Pinning it here is what stops the next modality added to
    UNTRAINABLE_MODALITIES from reintroducing the blemish silently.
    """
    var = AUDIO_VAR if modality == "audio" else VIDEO_VAR
    assert want in loop._untrainable_modality_refusal(modality, var, "clip")


def test_refusal_is_not_the_missing_text_column_message() -> None:
    """The exact confound that made T1-23's original 96 unreadable.

    A corpus carrying nothing but a timestamp earned the words "the thin path
    requires a 'text' column". If the modality refusal reads the same way, the
    row is back where it started with a better exit code.
    """
    msg = loop._untrainable_modality_refusal("audio", AUDIO_VAR, "waveform")
    assert "requires a 'text' column" not in msg


# --- the drop, which must stop being silent ----------------------------------


def test_dropped_columns_are_named() -> None:
    """T1-22 found a video field dropped with nothing anywhere naming it."""
    notice = loop._dropped_column_notice(["text", "video", "label"])
    assert notice is not None
    assert "video" in notice and "label" in notice


def test_a_text_only_corpus_says_nothing() -> None:
    """The control: a line printed on every run is a line nobody reads.

    Without this, "name the dropped columns" could be satisfied by printing an
    empty list on every text corpus, which is noise in place of silence.
    """
    assert loop._dropped_column_notice(["text"]) is None


def test_the_kept_column_is_not_reported_as_dropped() -> None:
    """Off-by-one in the other direction: the column being trained on is kept."""
    notice = loop._dropped_column_notice(["text", "label"])
    assert notice is not None
    assert "['label']" in notice, "the dropped list must be exactly the dropped columns"
