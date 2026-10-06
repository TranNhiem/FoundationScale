"""The P1 audio data contract: accept sound only where every leg of it is measurable.

Declaration, verification, decode -- in that order. ``FOUNDATIONSCALE_TRAIN_AUDIO_COLUMN``
is how a run says "this corpus is about sound" (#490 turns that declaration into a
refusal where nothing can train it); this module is the "something" that can: what the
family and the processor must expose before a single row is read, how one row becomes
one verified waveform, and what the manifest may claim about the rows it has seen.

Three rules hold here:

  verify before decode      ``audio_support_refusal`` and ``max_audio_seconds`` run
                            against the DECLARATIONS alone -- no corpus, no model,
                            no GPU -- so a run that cannot accept audio refuses
                            before the first byte is read.
  never silently change    No resampling (see ``load_audio``), no silent truncation
  the measured input        (a row past the family's cap is ``too_long``, counted
                            and out). The mono mixdown is the one transform, and
                            both accepted carriers are frames x channels that the
                            file carrier mixes down on read anyway. P0 measured the
                            audio placeholder count equal to the tower's valid output
                            length per sample (147/121/312/248 on both sides,
                            2026-10-06); that equality is only meaningful over
                            samples nothing quietly rewrote.
  a refusal counted is     ``AudioCoverage`` is the audio rows' Coverage document:
  not a row lost           checked against expected, refusals ACCOUNTED per reason,
                            and zero rows checked is VACUOUS rather than a pass.

torch, numpy, soundfile and transformers are unreachable from here at import time.
The train package must import under a bare interpreter -- the verification plane runs
on boxes the training stack does not -- so numpy and soundfile enter inside the
functions that decode and never at module load.
"""

from __future__ import annotations

import math
import numbers
import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from io import BytesIO
from typing import TYPE_CHECKING, Any, Protocol

if TYPE_CHECKING:  # never executed at runtime: the import contract above forbids it
    import numpy as np

__all__ = [
    "AUDIO_COLUMN_ENV",
    "AUDIO_LOAD_REASONS",
    "AudioCoverage",
    "AudioFamily",
    "AudioLoadError",
    "AudioProcessor",
    "audio_placeholder_counts",
    "audio_support_refusal",
    "load_audio",
    "max_audio_seconds",
    "placeholder_mismatches",
]


# #490: the declaration that makes this module's answer relevant at all. It is
# named in every refusal because a refusal nobody can undo is just a stop.
AUDIO_COLUMN_ENV = "FOUNDATIONSCALE_TRAIN_AUDIO_COLUMN"


class AudioFamily(Protocol):
    """The family surface read here, spelled as attributes rather than as a class.

    ``FamilySpec`` is the real implementation and a test double is equally valid;
    the call site is duck-typed on purpose, because a family can also be one this
    package has never registered and THAT is the case the first refusal needs
    words for.
    """

    name: str
    tower_modalities: Any


class AudioProcessor(Protocol):
    """The processor surface read here -- a set of attributes, not a class.

    The live object is a transformers processor whose class moves with the library
    and whose attributes move with the family (Gemma-4's ``audio_ms_per_token`` sits
    on the processor in one release and on its config in the next). Everything is
    therefore read with ``getattr``, and a missing attribute is an outcome this
    module returns -- never a default it invents.
    """

    audio_token: str
    boa_token: str
    eoa_token: str
    audio_token_id: int
    feature_extractor: Any
    audio_seq_length: int
    audio_ms_per_token: int


def _family_label(family: Any) -> str:
    """``family 'gemma4'``, or generic words when there is no name to quote."""
    name = getattr(family, "name", None)
    return f"family {name!r}" if isinstance(name, str) and name else "the model family"


def _tower_modalities(family: Any) -> set[str]:
    """The modality names ``family`` connects to towers, whichever spelling it used.

    ``FamilySpec.tower_modalities`` is ``(prefix, modality)`` pairs; the caller's
    exercise sets speak bare modality strings. Membership of bare ``"audio"`` in
    the pairs is ALWAYS False -- it compares against a tuple -- so a literal ``in``
    would refuse every registered family and leave the accept path dead. Both
    spellings are read; anything unrecognised counts as no modality, which is the
    refuse direction.
    """
    modalities: set[str] = set()
    for entry in getattr(family, "tower_modalities", ()) or ():
        if isinstance(entry, str):
            modalities.add(entry)
        elif isinstance(entry, tuple) and len(entry) == 2 and isinstance(entry[1], str):
            modalities.add(entry[1])
    return modalities


def _as_int(value: Any) -> int | None:
    """``value`` as a real int, or None.

    bools are excluded and not merely odd: ``isinstance(True, int)`` is True in
    Python, so a config typo in ``audio_token_id`` would otherwise read as token 1
    and place every audio block in somebody else's vocabulary.
    """
    if isinstance(value, bool):
        return None
    if isinstance(value, numbers.Integral):
        return int(value)
    return None


def _positive_measure(value: Any) -> float | None:
    """A positive measured quantity, or None.

    bools are not measurements (True would read as 1.0 -- a 40 ms token as 1 ms) and
    neither are NaNs and infinities: an unbounded "cap" and a missing cap are the
    same UNMEASURED, and must not quietly produce different manifests.
    """
    if isinstance(value, bool) or not isinstance(value, numbers.Real):
        return None
    number = float(value)
    return number if number > 0 and math.isfinite(number) else None


def _audio_refusal(missing: str) -> str:
    """The one refusal text, in the train loop's #490 words: what, the cost, the way out.

    T1-23 shipped the right exit code attached to a message about the wrong thing,
    so the wording is part of the claim: the missing piece is named FIRST, then what
    accepting anyway would cost, then how to stop declaring audio at all. The closing
    line is the loop's verbatim, so both surfaces tell an operator the same story.
    """
    return (
        f"an audio column is declared via {AUDIO_COLUMN_ENV}, but {missing}: accepting it "
        "would decode, batch and train on sound this run cannot verify, and report success "
        f"under an audio label, which is the defect this refuses. Unset {AUDIO_COLUMN_ENV} "
        "to train text-only on the same corpus"
    )


def audio_support_refusal(family: Any, processor: Any) -> str | None:
    """None when a declared audio column may be accepted, else the one reason it may not.

    Checks in THIS order and reports the first that fails -- an arbitrary but fixed
    choice, so the text is a function of the state and not of argument order:

      1. the family is registered and connects a tower to ``"audio"``;
      2. a processor was loaded at all;
      3. the processor exposes non-``None`` ``audio_token``/``boa_token``/``eoa_token``
         and an int ``audio_token_id`` the placeholder expansion can count (P0 proved
         the placeholder count and the tower length agree per sample ON GEMMA-4; this
         plane measures them equal again rather than remembering);
      4. ``feature_extractor.sampling_rate`` is a positive int -- the rate every
         waveform is later required to match exactly.

    ``family`` and ``processor`` are ``Any`` deliberately: the live objects arrive
    typed as protocols elsewhere in the plane (see :class:`AudioFamily` and
    :class:`AudioProcessor` for the surface read) and every attribute is read with
    ``getattr``, because a missing attribute is an OUTCOME this function returns --
    an annotation that said the attribute was present would hide exactly the gap the
    refusal exists to name.
    """
    label = _family_label(family)
    if family is None:
        return _audio_refusal(
            "the model family is not registered, so whether it carries an audio tower "
            "has never been declared"
        )
    if "audio" not in _tower_modalities(family):
        declared = ", ".join(sorted(_tower_modalities(family))) or "none"
        return _audio_refusal(
            f"{label} declares no audio tower (declared tower modalities: {declared})"
        )
    if processor is None:
        return _audio_refusal(
            f"{label} has no audio processor, so its audio tokens and sampling rate cannot be read"
        )
    for attr in ("audio_token", "boa_token", "eoa_token"):
        if getattr(processor, attr, None) is None:
            return _audio_refusal(f"{label} processor exposes no {attr}")
    if _as_int(getattr(processor, "audio_token_id", None)) is None:
        return _audio_refusal(
            f"{label} processor exposes audio_token_id "
            f"{getattr(processor, 'audio_token_id', None)!r}, which is not an int"
        )
    extractor = getattr(processor, "feature_extractor", None)
    if extractor is None:
        return _audio_refusal(
            f"{label} processor exposes no feature_extractor, so its sampling_rate cannot be read"
        )
    sampling_rate = _as_int(getattr(extractor, "sampling_rate", None))
    if sampling_rate is None or sampling_rate <= 0:
        return _audio_refusal(
            f"{label} processor's feature_extractor reports no positive int sampling_rate "
            f"(got {getattr(extractor, 'sampling_rate', None)!r})"
        )
    return None


def max_audio_seconds(processor: Any) -> float | None:
    """``audio_seq_length * audio_ms_per_token / 1000``, or None when either is unreadable.

    Gemma-4's cap is 750 * 40 ms = 30.0 s. Note the asymmetry with the token
    attributes above: ``audio_ms_per_token`` may live on the processor or on its
    config depending on the release, and only
    ``getattr(processor, "audio_ms_per_token", None)`` is consulted -- a cap this
    function cannot READ is returned as None (UNMEASURED) and the CALLER decides what
    a row with no bound means. A guessed cap is worse than none, because it turns
    into a number in the manifest that no measurement produced.
    """
    seq_length = _positive_measure(getattr(processor, "audio_seq_length", None))
    ms_per_token = _positive_measure(getattr(processor, "audio_ms_per_token", None))
    if seq_length is None or ms_per_token is None:
        return None
    return seq_length * ms_per_token / 1000.0


# The load-time reason vocabulary, closed on purpose: the manifest merges refusals
# by this token, so a near-miss spelling ("TooLong", "too_long") would open a second
# invisible bucket and silently undercount.
AUDIO_LOAD_REASONS: tuple[str, ...] = (
    "unreadable",
    "empty",
    "sample_rate_mismatch",
    "too_long",
    "too_short",
    "unsupported_value",
)


class AudioLoadError(ValueError):
    """One row the loader cannot accept, carrying the row and a countable reason.

    Subclasses ValueError because an unreadable row is a data error, not an
    environment failure: ``.row_id`` names the row the manifest will count and
    ``.reason`` is the bucket it lands in (:data:`AUDIO_LOAD_REASONS` -- a token
    outside that vocabulary is refused at CONSTRUCTION, because a reason nobody can
    total is not a reason).
    """

    def __init__(self, row_id: str, reason: str) -> None:
        if reason not in AUDIO_LOAD_REASONS:
            raise ValueError(
                f"audio load reason {reason!r} is not one of {list(AUDIO_LOAD_REASONS)}; "
                "the manifest merges refusals by this token, so a near-miss spelling splits "
                "into a bucket nobody can total"
            )
        super().__init__(f"row {row_id!r}: audio rejected ({reason})")
        self.row_id = row_id
        self.reason = reason


def load_audio(
    value: Any,
    *,
    target_sr: int,
    row_id: str,
    min_seconds: float = 0.0,
    max_seconds: float | None,
) -> tuple[np.ndarray, float]:
    """One verified ``(float32 mono waveform, duration_s)`` for ``row_id``, or a counted refusal.

    Accepts what corpora actually carry: a filesystem path (``str``/``PathLike``),
    the decoded HF-datasets shape ``{"array", "sampling_rate"}``, and the stored
    shape ``{"path"}`` with optional ``bytes``. Anything else is ``unsupported_value``
    -- ``value`` is ``Any`` because the unsupported outcome is part of THIS contract,
    and a type that forbade the carrier this function must reject would make that
    rejection untestable.

    Not resampling is the point, not a gap. ``target_sr`` is the rate the feature
    extractor declares and the manifest records; the moment this function converts
    rates by itself, the waveform the tower sees stops being the waveform the manifest
    describes -- the seconds, the placeholder counts and the spectrum all move while
    the row looks untouched. Since the loader may not improve a row silently, a rate
    mismatch REFUSES (``sample_rate_mismatch``); resampling arrives later, as an
    explicit declaration.

    Not truncating is the same rule applied in time: a row past ``max_seconds`` is
    ``too_long``, because the processor would cap audio tokens and silently shorten
    the input, and a row under ``min_seconds`` is ``too_short``. Both are counted.
    Neither is edited.

    ``max_seconds=None`` is UNMEASURED, not unlimited-as-measured -- the caller has
    decided there is no cap (see :func:`max_audio_seconds`), and exactly the bounds
    given are the bounds applied.
    """
    samples, rate = _as_waveform_and_rate(value, row_id)
    wave = _mono_float32(samples, row_id)
    if rate != target_sr:
        # See the docstring: a resample here would change the measured input under
        # everyone's feet. The row leaves with its reason; it does not get rewritten.
        raise AudioLoadError(row_id, "sample_rate_mismatch")
    frames = int(wave.shape[0])
    if frames == 0:
        raise AudioLoadError(row_id, "empty")
    duration = frames / float(rate)
    if max_seconds is not None and duration > max_seconds:
        raise AudioLoadError(row_id, "too_long")
    if duration < min_seconds:
        raise AudioLoadError(row_id, "too_short")
    return wave, duration


def _as_waveform_and_rate(value: Any, row_id: str) -> tuple[Any, int]:
    """(samples, sampling_rate) out of any one accepted carrier, or a counted refusal.

    The three carriers are checked in THIS order because the decoded HF form also
    carries a ``"path"`` key -- a decoded row is array + path + sampling_rate, so
    testing ``"path"`` first would send the cleanest measurement in the corpus to the
    filesystem and lose it whenever the cache lives elsewhere. The stored form decodes
    from its ``"bytes"`` when it has any, for the same reason: its path names a cache
    that need not exist on the box doing the decode.
    """
    if isinstance(value, Mapping):
        if "array" in value:
            rate = _as_int(value.get("sampling_rate"))
            if rate is None or rate <= 0:
                raise AudioLoadError(row_id, "unsupported_value")
            return value["array"], rate
        if "path" in value:
            payload = value.get("bytes")
            if isinstance(payload, (bytes, bytearray, memoryview)):
                return _read_soundfile(BytesIO(bytes(payload)), row_id)
            return _read_soundfile(_as_str_path(value.get("path"), row_id), row_id)
        raise AudioLoadError(row_id, "unsupported_value")
    if isinstance(value, (str, os.PathLike)):
        return _read_soundfile(_as_str_path(value, row_id), row_id)
    # An int, a bare list, a handle nobody declared: one vocabulary of reasons keeps
    # the manifest's buckets countable, and "unloadable" is not a reason a retry fixes.
    raise AudioLoadError(row_id, "unsupported_value")


def _as_str_path(value: Any, row_id: str) -> str:
    """``value`` as a filesystem path to hand soundfile, or an unsupported row."""
    if isinstance(value, os.PathLike):
        value = os.fspath(value)
    if not isinstance(value, str) or not value:
        raise AudioLoadError(row_id, "unsupported_value")
    return value


def _read_soundfile(source: Any, row_id: str) -> tuple[Any, int]:
    """soundfile's ``(samples, rate)`` for ``source``, or an ``unreadable`` row.

    Any decode failure is one refused row and never a crash: a corpus carries a
    truncated header now and then, and the manifest counts it instead of losing the
    run -- a crash here would make one bad row unevidenced rather than counted.
    """
    import importlib

    # `import_module`, not `import soundfile as sf`: the import stays function-local
    # (the contract above) AND soundfile ships no py.typed, so a direct import makes
    # mypy's verdict depend on which machine had the package -- an environment
    # measurement where a code measurement was wanted. `types.ModuleType.__getattr__`
    # is Any in typeshed, so the decode is typed-unchecked rather than typed-wrong.
    sf = importlib.import_module("soundfile")
    try:
        samples, rate = sf.read(source)
    except Exception as exc:  # noqa: BLE001 - every decode failure is one row, not a run
        raise AudioLoadError(row_id, "unreadable") from exc
    return samples, rate


def _mono_float32(samples: Any, row_id: str) -> np.ndarray:
    """One float32 dimension out of whatever the carrier handed over.

    Shape, not provenance, decides here: 1-D is already mono, 2-D is mixed down over
    the LAST axis (soundfile and the decoded HF form are both frames x channels;
    averaging the wrong axis would silently halve the duration instead). Anything
    else refuses -- a shape this code cannot name is a shape it must not average.

    Integer PCM is scaled the way soundfile scales it on the file path (PCM to
    [-1, 1]); without that, the two accepted carriers would hand the tower different
    numbers for the same samples, differing harmlessly for one round trip and fatally
    never.
    """
    import numpy as np

    try:
        array = np.asarray(samples)
        if array.ndim == 2:
            array = array.mean(axis=-1)
        elif array.ndim != 1:
            raise AudioLoadError(row_id, "unsupported_value")
        if np.issubdtype(array.dtype, np.integer):
            info = np.iinfo(array.dtype)
            array = array.astype(np.float32) / float(max(abs(info.min), info.max))
        return array.astype(np.float32, copy=False)
    except AudioLoadError:
        raise
    except Exception as exc:  # noqa: BLE001 - numpy refuses more shapes than it names
        raise AudioLoadError(row_id, "unsupported_value") from exc


@dataclass
class AudioCoverage:
    """Rows checked against rows expected, refused-by-reason, and deliberately mutable.

    Mutable on purpose, and not a contradiction of the frozen dataclass doctrine:
    that doctrine protects DECLARATIONS from drifting mid-run, while this object IS
    mid-run -- a per-row accumulator the caller updates in place, so a rewritten
    binding cannot be mistaken for a counted row.

    Mirrors the gate Coverage doctrine: checked/expected, refusals ACCOUNTED toward
    the denominator (tolerant mode's whole contract -- a dropped row is absent from
    the batch and present in the total), and 0 rows checked blocks regardless of what
    else accumulated. A manifest proving nothing about the audio it saw is not a
    manifest.
    """

    rows_expected: int
    rows_checked: int = 0
    seconds_total: float = 0.0
    sampling_rate: int | None = None
    refused: dict[str, int] = field(default_factory=dict)

    def record_ok(self, duration_s: float) -> None:
        """One accepted row: counted, with the seconds it contributed to the manifest."""
        self.rows_checked += 1
        self.seconds_total += float(duration_s)

    def record_refused(self, reason: str) -> None:
        """One refused row: counted in its bucket, never dropped from the total."""
        self.refused[reason] = self.refused.get(reason, 0) + 1

    def verdict(self) -> str:
        """VACUOUS / UNDERCOVERED / OVERCOVERED / COVERED for what has been counted.

        0 rows CHECKED is VACUOUS before anything else, even when every row was
        accounted as refused: a batch of drops measures the decoder's refusals and
        nothing about the path the audio would have taken.
        """
        refused = sum(self.refused.values())
        if self.rows_checked == 0:
            return "VACUOUS"
        if self.rows_checked + refused < self.rows_expected:
            return "UNDERCOVERED"
        if self.rows_checked + refused > self.rows_expected:
            return "OVERCOVERED"
        return "COVERED"

    def as_manifest(self) -> dict[str, object]:
        """JSON-ready snapshot of the counts, keys stable as contract.

        The refused map is COPIED and the verdict COMPUTED once: the caller keeps
        counting after this call, and a manifest row that drifts as the loader
        continues is a manifest row that contradicts itself.
        """
        return {
            "rows_expected": self.rows_expected,
            "rows_checked": self.rows_checked,
            "rows_refused": sum(self.refused.values()),
            "seconds_total": self.seconds_total,
            "sampling_rate": self.sampling_rate,
            "refused": dict(self.refused),
            "verdict": self.verdict(),
        }


def audio_placeholder_counts(
    input_ids_rows: Sequence[Sequence[int]],
    audio_token_id: int,
) -> list[int]:
    """Per-row count of audio placeholders in already-tokenized ids. Pure python.

    No tensor types here on purpose: this runs on the row schema before any model
    exists (and under a bare interpreter), where counting must not require the
    training stack to have imported.
    """
    return [sum(1 for token in row if token == audio_token_id) for row in input_ids_rows]


def placeholder_mismatches(
    placeholder_counts: Sequence[int],
    tower_lengths: Sequence[int],
) -> list[tuple[int, int, int]]:
    """``(row index, placeholders, tower length)`` for every row where the two disagree.

    P0 measured these equal per sample on gemma-4 (147/121/312/248 on both sides);
    this is the check that keeps that equality a measurement instead of a memory. A
    mismatch in the LENGTHS of the two lists raises: a denominator mismatch is not a
    pass -- ``zip()`` would compare only the shared prefix and report agreement over
    rows that were never compared at all.
    """
    if len(placeholder_counts) != len(tower_lengths):
        raise ValueError(
            f"placeholder_counts and tower_lengths describe different numbers of rows "
            f"({len(placeholder_counts)} vs {len(tower_lengths)}); a denominator mismatch is "
            "not a pass -- zip() would compare only the shared prefix and report agreement"
        )
    return [
        (index, placeholders, length)
        for index, (placeholders, length) in enumerate(
            zip(placeholder_counts, tower_lengths, strict=True)
        )
        if placeholders != length
    ]
