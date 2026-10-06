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

torch, numpy, soundfile and transformers are unreachable from here at import time
(stdlib is not -- ``ctypes`` and ``multiprocessing`` load here, backing
SharedAudioCoverage's counters that must survive the DataLoader worker processes
that write them).
The train package must import under a bare interpreter -- the verification plane runs
on boxes the training stack does not -- so numpy and soundfile enter inside the
functions that decode and never at module load.
"""

from __future__ import annotations

import ctypes
import math
import multiprocessing
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
    "AUDIO_FEATURES_KEY",
    "AUDIO_LOAD_REASONS",
    "AudioCollator",
    "AudioCoverage",
    "AudioFamily",
    "AudioLoadError",
    "AudioProcessor",
    "SharedAudioCoverage",
    "audio_placeholder_counts",
    "audio_support_refusal",
    "load_audio",
    "max_audio_seconds",
    "placeholder_mismatches",
    "refuse_if_audio_features_dropped",
    "train_audio_collator_or_refuse",
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
    # Per-row placeholder verification against the processor's own token count:
    # rows where it was measured, and rows where the processor offers no count to
    # compare with. The second number is UNMEASURED, and it is reported, not folded
    # into the first.
    placeholder_rows_verified: int = 0
    placeholder_rows_unmeasured: int = 0

    def record_ok(self, duration_s: float) -> None:
        """One accepted row: counted, with the seconds it contributed to the manifest."""
        self.rows_checked += 1
        self.seconds_total += float(duration_s)

    def record_refused(self, reason: str) -> None:
        """One refused row: counted in its bucket, never dropped from the total."""
        self.refused[reason] = self.refused.get(reason, 0) + 1

    def add_expected(self, n: int = 1) -> None:
        """Count ``n`` more rows the run declares it will collate. Plain increments.

        The write half of the surface this class shares with
        :class:`SharedAudioCoverage`, where arithmetic cannot ride a property
        (``rows_expected += 1`` is exactly how counts stayed per-process). Here
        the method is a thin increment: one interface, two storage backends, so
        a caller that tracks coverage needs no branch over which class it holds.
        """
        self.rows_expected += n

    def add_placeholder_verified(self, n: int = 1) -> None:
        """Count ``n`` more rows whose placeholder count matched the processor's own."""
        self.placeholder_rows_verified += n

    def add_placeholder_unmeasured(self, n: int = 1) -> None:
        """Count ``n`` more rows with no processor count to compare placeholders with."""
        self.placeholder_rows_unmeasured += n

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

    def reset(self) -> None:
        """Zero every count, keeping the object (the collator holds a reference to it).

        For the construction-time survival probe: those rows are collated to prove
        input_features reaches the batch, not to train, and counting them would
        report two rows the run never trained on.
        """
        self.rows_expected = 0
        self.rows_checked = 0
        self.seconds_total = 0.0
        self.sampling_rate = None
        self.refused = {}
        self.placeholder_rows_verified = 0
        self.placeholder_rows_unmeasured = 0

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
            "placeholder_rows_verified": self.placeholder_rows_verified,
            "placeholder_rows_unmeasured": self.placeholder_rows_unmeasured,
            "verdict": self.verdict(),
        }


def _shared_coverage_context() -> Any:
    """The multiprocessing context SharedAudioCoverage makes its shared counters under.

    The DEFAULT context, because that is the one a DataLoader starts its workers with
    unless told otherwise (fork on Linux before Python 3.14, forkserver after, spawn
    on macOS), and shared ctypes and locks may only reach a process started under the
    same context. They travel as process START arguments -- the collate_fn a
    DataLoader hands each worker at creation -- the one channel multiprocessing allows
    for them. Measured: a fork-context lock handed to a spawn-started DataLoader worker
    raises "A SemLock created in a fork context is being shared with a process in a
    spawn context".
    """
    return multiprocessing.get_context()


def _coverage_verdict(rows_checked: int, rows_expected: int, refused_total: int) -> str:
    """VACUOUS / UNDERCOVERED / OVERCOVERED / COVERED over three totals. Pure.

    :meth:`AudioCoverage.verdict`'s four comparisons spelled as a function of
    three numbers, so :class:`SharedAudioCoverage` can compute a verdict from ONE
    locked snapshot instead of re-reading counters a worker may move between
    reads. AudioCoverage keeps its own inline body (other code owns that class);
    ``test_audio_shared_coverage`` pins one manifest equal to the other's, which
    is what makes "the same semantics" a measurement rather than a claim.
    """
    if rows_checked == 0:
        return "VACUOUS"
    if rows_checked + refused_total < rows_expected:
        return "UNDERCOVERED"
    if rows_checked + refused_total > rows_expected:
        return "OVERCOVERED"
    return "COVERED"


class SharedAudioCoverage:
    """:class:`AudioCoverage` whose counters live in multiprocessing shared memory. Same surface.

    MEASURED defect (GB200, 2026-10-06): with ``--dataloader-num-workers 4`` the
    collator runs in DataLoader WORKER processes, where a plain AudioCoverage is
    a per-process COPY -- ``rows_expected += 1`` in a worker updates only that
    worker's copy, the main process reads rows_checked=0, the speech gates
    report VACUOUS and the run ends RED. Zero workers passed only because there
    happened to be one copy to count in.

    So every counter is a ``multiprocessing`` shared-memory object made ONCE in
    the constructor, and every update takes ONE shared lock. Locking rule: the
    intervals callers observe are several fields wide (``record_ok`` writes
    ``rows_checked`` and ``seconds_total`` together) and without a single lock
    two children can interleave halves of two updates and publish counts no row
    produced. That lock is ``multiprocessing.Lock`` -- NOT re-entrant -- so no
    update may call another one from inside its ``with``.

    Why nothing must be shipped by hand: a DataLoader hands its ``collate_fn``,
    and therefore the closure holding this object, to its workers AT WORKER
    CREATION. Under ``fork`` the mapping is inherited verbatim; under ``spawn``
    the same objects ride the process start arguments -- the single channel
    multiprocessing lets shared ctypes and locks travel ("should only be shared
    between processes through inheritance" is what queue pickling answers).
    Both spellings ARE inheritance; a plain attribute is the one thing
    inheritance does not share, because it points at a box each process copied.

    Surface is AudioCoverage's as callers read it -- the manifest keys, the
    values and the verdict are the same contract (``test_audio_shared_coverage``
    pins one class's manifest equal to the other's) -- with the writable fields'
    arithmetic moved to ``add_expected``/``add_placeholder_verified``/
    ``add_placeholder_unmeasured``: a property cannot be ``+=``-ed, and that
    assignment is exactly the write that used to stay per-process.
    """

    def __init__(self, rows_expected: int = 0) -> None:
        ctx = _shared_coverage_context()
        # ONE lock for every update, always taken BEFORE any counter is touched
        # (see the class docstring: it is not re-entrant and the caller-visible
        # intervals are multi-field).
        self._lock: Any = ctx.Lock()
        self._rows_expected: Any = ctx.Value(ctypes.c_long, int(rows_expected))
        self._rows_checked: Any = ctx.Value(ctypes.c_long, 0)
        self._seconds_total: Any = ctx.Value(ctypes.c_double, 0.0)
        # A c_long cannot hold None; -1 IS the None sentinel and the property
        # below is its only reader -- a leaked sentinel would claim the run
        # trained at a sampling rate of -1, a number no measurement produced.
        self._sampling_rate: Any = ctx.Value(ctypes.c_long, -1)
        self._placeholder_rows_verified: Any = ctx.Value(ctypes.c_long, 0)
        self._placeholder_rows_unmeasured: Any = ctx.Value(ctypes.c_long, 0)
        # Refusals as one c_long Array INDEXED by AUDIO_LOAD_REASONS: the reason
        # vocabulary is closed, so a fixed shape covers every bucket
        # (multiprocessing offers no shared dict whose updates would be atomic
        # under this lock) and no near-miss spelling can open a second invisible
        # one -- record_refused raises for a reason outside the vocabulary, the
        # same check AudioLoadError applies at construction.
        self._refused: Any = ctx.Array(ctypes.c_long, len(AUDIO_LOAD_REASONS))
        self._refused_index: dict[str, int] = {
            reason: index for index, reason in enumerate(AUDIO_LOAD_REASONS)
        }

    # Single-field readers below take NO lock: their Value/Array slot is
    # self-synchronising, and only these readers may be called from inside a
    # lock-holding method (as_manifest does) without deadlocking on the one
    # non-re-entrant lock.

    @property
    def rows_expected(self) -> int:
        """Rows declared to exist -- read-only on purpose; add_expected() is the write."""
        return int(self._rows_expected.value)

    @property
    def rows_checked(self) -> int:
        """Rows measured and accepted (record_ok), as a plain int."""
        return int(self._rows_checked.value)

    @property
    def seconds_total(self) -> float:
        """Waveform seconds the checked rows contributed (record_ok), as a plain float."""
        return float(self._seconds_total.value)

    @property
    def sampling_rate(self) -> int | None:
        """The rate the run was measured at, or None -- the -1 sentinel never returns."""
        raw = int(self._sampling_rate.value)
        return None if raw < 0 else raw

    @property
    def refused(self) -> dict[str, int]:
        """Refusals per reason; uncounted buckets stay absent, as in AudioCoverage's map.

        A SNAPSHOT on purpose (the word is the contract): the caller keeps this
        and counts keep moving in the workers, so it must be a copy -- the
        manifest is copied and computed once for the same reason.
        """
        with self._lock:
            return {
                reason: int(self._refused[index])
                for reason, index in self._refused_index.items()
                if self._refused[index]
            }

    @property
    def placeholder_rows_verified(self) -> int:
        """Rows whose placeholder count was measured against the processor's own prediction."""
        return int(self._placeholder_rows_verified.value)

    @property
    def placeholder_rows_unmeasured(self) -> int:
        """Rows with no processor count to compare placeholders with (UNMEASURED, reported)."""
        return int(self._placeholder_rows_unmeasured.value)

    def record_ok(self, duration_s: float) -> None:
        """One accepted row: counted, with the seconds it contributed to the manifest.

        The pair is written under one lock so a reader never sees the row
        counted without its seconds, nor seconds without the row.
        """
        with self._lock:
            self._rows_checked.value = int(self._rows_checked.value) + 1
            self._seconds_total.value = float(self._seconds_total.value) + float(duration_s)

    def record_refused(self, reason: str) -> None:
        """One refused row: counted in its bucket, never dropped from the total.

        A reason outside AUDIO_LOAD_REASONS raises BEFORE anything is counted:
        the Array has one slot per vocabulary token and an unknown reason has no
        slot to count in, so accepting it would lose the row from the manifest.
        """
        index = self._refused_index.get(reason)
        if index is None:
            raise ValueError(
                f"audio load reason {reason!r} is not one of {list(AUDIO_LOAD_REASONS)}; "
                "the manifest merges refusals by this token, so a near-miss spelling "
                "splits into a bucket nobody can total"
            )
        with self._lock:
            self._refused[index] = int(self._refused[index]) + 1

    def add_expected(self, n: int = 1) -> None:
        """Count ``n`` more rows the run declares it will collate. Atomic increment.

        The write half of the surface this class shares with
        :class:`AudioCoverage` -- see that class's method for why the interface
        is arithmetic, and the class docstring for why it is not a property.
        """
        with self._lock:
            self._rows_expected.value = int(self._rows_expected.value) + int(n)

    def add_placeholder_verified(self, n: int = 1) -> None:
        """Count ``n`` more rows whose placeholder count matched the processor's own."""
        with self._lock:
            self._placeholder_rows_verified.value = int(
                self._placeholder_rows_verified.value
            ) + int(n)

    def add_placeholder_unmeasured(self, n: int = 1) -> None:
        """Count ``n`` more rows with no processor count to compare placeholders with."""
        with self._lock:
            self._placeholder_rows_unmeasured.value = int(
                self._placeholder_rows_unmeasured.value
            ) + int(n)

    def verdict(self) -> str:
        """VACUOUS / UNDERCOVERED / OVERCOVERED / COVERED for what has been counted.

        0 rows CHECKED is VACUOUS before anything else -- a batch of drops
        measures the decoder's refusals and nothing about the path the audio
        would have taken. One locked read so the three totals agree.
        """
        with self._lock:
            return _coverage_verdict(
                int(self._rows_checked.value),
                int(self._rows_expected.value),
                sum(int(self._refused[i]) for i in range(len(AUDIO_LOAD_REASONS))),
            )

    def reset(self) -> None:
        """Zero every count in place, keeping the OBJECT (workers hold a reference to it).

        AudioCoverage.reset's purpose -- a construction-time survival probe must
        not count -- with its one extra rule: a DataLoader's workers inherited
        THIS instance, so a fresh one in the parent would leave them tallying
        into the old one and the manifest would drift out of the object the
        gates read.
        """
        with self._lock:
            self._rows_expected.value = 0
            self._rows_checked.value = 0
            self._seconds_total.value = 0.0
            self._sampling_rate.value = -1
            self._placeholder_rows_verified.value = 0
            self._placeholder_rows_unmeasured.value = 0
            for index in range(len(AUDIO_LOAD_REASONS)):
                self._refused[index] = 0

    def as_manifest(self) -> dict[str, object]:
        """JSON-ready snapshot of the counts -- AudioCoverage.as_manifest's keys verbatim.

        ONE locked snapshot for every field (a worker may keep counting while the
        caller reads) and plain builtins in the document: a manifest row that
        drifts as the loader continues is a manifest row that contradicts
        itself, and a ctypes handle in JSON is not a measurement.
        """
        with self._lock:
            rows_expected = self.rows_expected
            rows_checked = self.rows_checked
            seconds_total = self.seconds_total
            refused = {
                reason: int(self._refused[index])
                for reason, index in self._refused_index.items()
                if self._refused[index]
            }
            rows_refused = sum(refused.values())
            return {
                "rows_expected": rows_expected,
                "rows_checked": rows_checked,
                "rows_refused": rows_refused,
                "seconds_total": seconds_total,
                "sampling_rate": self.sampling_rate,
                "refused": dict(refused),
                "placeholder_rows_verified": self.placeholder_rows_verified,
                "placeholder_rows_unmeasured": self.placeholder_rows_unmeasured,
                "verdict": _coverage_verdict(rows_checked, rows_expected, rows_refused),
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


# ---------------------------------------------------------------------------
# The TRAIN plane's audio carrier -- the pixel collator's direct twin
# (rl.prompt_surface.train_image_collator_or_refuse and
# rl.prompt_surface.refuse_if_pixel_column_dropped), widened to sound per
# #P1-SLICE2. Same doctrine: exit 96, never a fallback; per-row loader
# verification; the DECLARED modality key is checked on the batch the model
# would actually receive, because a declared axis that is not executed is a
# refusal, not a pass.
# ---------------------------------------------------------------------------

AUDIO_FEATURES_KEY = "input_features"
"""The key the audio tower consumes.

Named ONCE here so the collator and the drop probe must agree on the spelling
by import, not by re-typing. MEASURED on gemma-4 (P0, GB200,
transformers 5.5.0): apply_chat_template(return_dict=True) emits exactly this
key for the audio tower's log-mel features [B, T, 128], plus input_features_mask
[B, T].
"""


def _audio_refuse_exit_96(message: str) -> None:
    """The one loud refusal mechanism -- mirror of rl.prompt_surface._refuse_exit_96.

    Duplicated (rather than imported) to keep rl/prompt_surface.py untouched and
    to keep the "bare interpreter" import contract of this module simple. The
    MECHANISM is identical to the one refuse_if_pixel_column_dropped raises, per
    the shared contract: a stderr line prefixed "REFUSAL (exit 96):" then
    SystemExit(96). Never asserts, never returns, never falls back.
    """
    import sys  # stdlib; the module keeps its top-level imports minimal

    print(f"REFUSAL (exit 96): {message}", file=sys.stderr)
    raise SystemExit(96)


def resolve_audio_surface(model_id: str) -> Any:
    """The processor surface for a declared audio column; AutoProcessor is REQUIRED.

    Same rule as the image arm's ``rl.prompt_surface.resolve_prompt_surface`` --
    a processor that fails to load is a refusal (96), never a downgrade to a
    tokenizer, because a tokenizer would encode the text and drop the waveform
    with no signal. It is a separate function only so that the refusal names the
    axis that was actually declared: that function's messages say "corpus carries
    images", and on an audio run that wording would point the operator at the
    wrong column.
    """
    from foundationscale.rl.prompt_surface import PromptSurface  # noqa: PLC0415

    try:
        from transformers import AutoProcessor  # noqa: PLC0415
    except ImportError:
        _audio_refuse_exit_96(
            f"{AUDIO_COLUMN_ENV} is declared but transformers is absent; AutoProcessor "
            "cannot be imported and no tokenizer fallback is permitted (it would "
            "train on the text and drop the audio)"
        )
    try:
        processor = AutoProcessor.from_pretrained(model_id)
    except Exception as exc:  # noqa: BLE001 - any load failure is the same refusal
        _audio_refuse_exit_96(
            f"{AUDIO_COLUMN_ENV} is declared but AutoProcessor failed to load for "
            f"{model_id!r}: {exc!r}. Refusing rather than downgrading to a tokenizer, "
            "which would train on the text and drop the audio"
        )
    return PromptSurface(
        kind="processor",
        surface=processor,
        reason=(
            f"processor path: corpus carries audio; AutoProcessor loaded for "
            f"{model_id!r} ({type(processor).__name__})"
        ),
        supports_images=False,
    )


class AudioCollator(Protocol):
    """The TRAIN loop's audio collator surface: rows -> batch dict, with coverage.

    Exposed as a Protocol (not a class) so a caller can pass any duck-typed
    callable carrying ``coverage`` -- tests build their own fakes, and the
    closure returned by ``train_audio_collator_or_refuse`` satisfies the shape
    at runtime via ``setattr``. ``coverage`` is updated PER ROW as the collator
    sees rows, so a verifier can read the manifest of what this collator
    processed without reaching into the function that produced the batch.
    """

    coverage: AudioCoverage

    def __call__(
        self, rows: Sequence[Any]
    ) -> dict[str, Any]: ...  # pragma: no cover -- protocol shape only


def refuse_if_audio_features_dropped(
    batch_keys: Any, audio_column: str, feature_key: str = AUDIO_FEATURES_KEY
) -> None:
    """REFUSE (96) when a collator's output batch lost the audio feature column.

    The direct analog of ``rl.prompt_surface.refuse_if_pixel_column_dropped``
    (see its docstring: #371/#410/#422's class -- a declared axis that is not
    executed is a refusal). The mechanism is the SAME exit-96 mechanism: a
    stderr message naming the DECLARED column AND the dropped key so an
    operator can tell which axis vanished, then SystemExit(96). A batch that
    carries ``feature_key`` is allowed -- this probe is specifically about the
    silent drop of the declared modality, not a blanket end-to-end check.
    """
    keys = {str(k) for k in batch_keys}
    if feature_key not in keys:
        _audio_refuse_exit_96(
            f"audio column {audio_column!r} is DECLARED, but the batch the model would "
            f"actually receive has keys {sorted(keys)}: the audio feature column "
            f"{feature_key!r} was DROPPED between the dataset and the forward. "
            "That is the silent-drop defect (#371/#410, and #422's class -- a declared "
            "axis that is not executed is a refusal, not a pass). The run is refused; "
            "it never trains text-only under an audio label."
        )


def train_audio_collator_or_refuse(
    surface: Any,
    *,
    audio_column: str,
    max_length: int,
    text_fields: tuple[str, str] = ("text", "answer"),
) -> AudioCollator:
    """Build the audio TRAIN collator; refuse (96) at CONSTRUCTION if the surface cannot carry it.

    ``surface`` is an ``rl.prompt_surface.PromptSurface`` (typed ``Any`` here to
    keep this module importable under a bare interpreter AND to keep
    rl/prompt_surface.py untouched). The image collator's twin -- same loader
    refusal pattern, same label-mask-clone discipline, same "declared column"
    drop probe -- widened from one text cell to the P1 audio contract:

      * user prompt + assistant target come from the SAME row the image
        collator reads, via the pair ``text_fields=(prompt_field, answer_field)``.
        The user slot's default, ``text``, matches
        ``rl.prompt_surface.train_image_collator_or_refuse.text_column``
        default; the answer slot is added because audio SFT supervises an
        assistant target alone (P0: "only the assistant transcript is
        supervised: [28, 19, 47, 39] tokens; prompt and audio positions are
        -100"), which the image collator -- a causal-LM objective over
        input_ids -- does not need.
      * one row -> one (waveform, duration) via ``load_audio``; an
        ``AudioLoadError`` becomes an exit-96 refusal naming the row, the
        reason and the DECLARED column. Strict mode only: tolerating bad rows
        silently is the defect being fixed. No resampling, no truncation.
      * ``apply_chat_template`` on the FULL message list (user with one
        ``{"type": "audio"}`` block and one ``{"type": "text"}`` block, then an
        assistant turn), producing ``input_ids``, ``attention_mask``,
        ``input_features [B, T, 128]`` and ``input_features_mask [B, T]``
        in ONE call (P0 measured on gemma-4 + GB200).
      * labels mask (a) the prompt up to the assistant-turn boundary -- the
        boundary is recovered from a SECOND tokenizer-only
        ``apply_chat_template`` call on the prompt-only message list
        (``add_generation_prompt=True``), whose un-padded length is exactly the
        number of prompt tokens that precede the assistant target; (b) any
        position with ``attention_mask == 0`` (pad, both padding sides -- the
        prompt's first-attended position is located per-row via ``argmax`` on
        the attention mask, so left- and right-padded batches both behave);
        (c) every ``audio_token_id`` position, since a placeholder is a valid
        INPUT slot for tower output and never a valid generation target (the
        #450 lesson measured at 73.6% of label positions on one family).
      * PER-ROW, when the processor declares ``_compute_audio_num_tokens``, the
        count of ``audio_token_id`` in that row's ``input_ids`` must EQUAL the
        value the processor's own prediction function returns for that row's
        waveform length. A mismatch means the processor and the model pair are
        out of contract (P0 measured these equal on gemma-4 -- 147/121/312/248
        on both sides), and the run refuses at 96. When the function is
        ABSENT, the check is silently skipped -- not reported as pass -- for
        the same reason ``max_audio_seconds`` returns ``None`` (UNMEASURED)
        rather than a guessed cap.
      * an encoded batch wider than ``max_length`` REFUSES. NO truncation on
        this path: truncating audio placeholders drops measured sound between
        the loader and the forward -- the silent-drop defect one layer down.
        Same doctrine as
        ``rl.prompt_surface._refuse_if_image_batch_exceeds_declared_window``,
        but bounded by the caller's declared budget rather than a model's
        window.
      * a batch whose ``apply_chat_template`` output is MISSING
        ``input_features`` is refused on the way out via
        ``refuse_if_audio_features_dropped`` -- the audio twin of the pixel
        drop probe.

    The returned object is callable on rows -> batch dict and exposes
    ``.coverage`` updated per row -- a :class:`SharedAudioCoverage` (same
    surface as ``AudioCoverage``, counters in shared memory), so the counts a
    DataLoader's workers record are the counts the main process's gates read.
    """
    if surface.kind != "processor":
        _audio_refuse_exit_96(
            f"a train-time audio collator was requested for column "
            f"{audio_column!r} but the resolved surface is {surface.kind!r}: "
            "encoding audio through a tokenizer is the silent-drop defect one "
            "layer up (the waveform would never reach the audio tower and no "
            "signal would say so), so this refuses rather than collates"
        )
    processor = surface.surface
    # rows_expected grows with rows seen so the invariant
    #   rows_expected == rows_checked + sum(refused.values())
    # is preserved across batches -- keeping the manifest countable regardless
    # of how many times the collator is invoked -- and it grows through
    # SharedAudioCoverage's SHARED counters: a DataLoader runs THIS closure in
    # its worker processes, where a plain AudioCoverage is a per-worker copy and
    # the main process's gates would read rows_checked=0 (the measured
    # --dataloader-num-workers 4 failure, 2026-10-06).
    coverage = SharedAudioCoverage(rows_expected=0)
    user_field, answer_field = text_fields

    def collate(rows: Sequence[Any]) -> dict[str, Any]:
        full_convos: list[list[dict[str, Any]]] = []
        prompt_convos: list[list[dict[str, Any]]] = []
        waves: list[Any] = []
        durations: list[float] = []
        for index, raw_row in enumerate(rows):
            row = raw_row if isinstance(raw_row, dict) else vars(raw_row)
            row_id = f"train-row[{index}]"
            coverage.add_expected()
            value = row.get(audio_column)
            try:
                wave, duration = load_audio(
                    value,
                    target_sr=processor.feature_extractor.sampling_rate,
                    row_id=row_id,
                    max_seconds=max_audio_seconds(processor),
                )
            except AudioLoadError as exc:
                # Strict mode: one bad row is a REFUSED RUN, never a silently
                # shorter batch. The message names the row id, the reason from
                # AUDIO_LOAD_REASONS and the DECLARED column so the operator
                # can fix the corpus without guessing which axis was in play.
                _audio_refuse_exit_96(
                    f"audio column {audio_column!r} row {row_id!r} refused by the "
                    f"loader: reason={exc.reason!r}, source={value!r}. Strict mode "
                    "does not drop rows silently (that is the defect being fixed); "
                    "fix the corpus or the loader, or narrow the audio column"
                )
            user_text = str(row.get(user_field, ""))
            answer_text = str(row.get(answer_field, ""))
            # MEASURED (P0, GB200, transformers 5.5.0):
            #   processor.apply_chat_template(
            #       messages, tokenize=True, return_dict=True,
            #       return_tensors="pt", processor_kwargs={"padding": True})
            # accepts [{"type":"audio","audio": <float32 @ 16 kHz>},
            #          {"type":"text","text": ...}] on a user turn followed by
            # an assistant turn, and returns input_ids / attention_mask /
            # input_features [B,T,128] / input_features_mask [B,T] in ONE call.
            # No separate feature-extractor pass: a second call risks the
            # silent-drop class of bug this plane keeps finding.
            user_msg: dict[str, Any] = {
                "role": "user",
                "content": [
                    {"type": "audio", "audio": wave},
                    {"type": "text", "text": user_text},
                ],
            }
            answer_msg: dict[str, Any] = {
                "role": "assistant",
                "content": [{"type": "text", "text": answer_text}],
            }
            full_convos.append([user_msg, answer_msg])
            prompt_convos.append([user_msg])
            waves.append(wave)
            durations.append(duration)

        if not full_convos:
            return {}

        full_batch = processor.apply_chat_template(
            full_convos,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
            # transformers 5.5 routes processor arguments through
            # processor_kwargs; passed loose they still apply, after a warning
            # on every batch.
            processor_kwargs={"padding": True},
        )
        input_ids = full_batch["input_ids"]
        width = int(input_ids.shape[-1])
        if width > max_length:
            # NO truncation on the audio path: truncating a batch that carries
            # audio placeholders drops measured sound from the training input,
            # the silent-drop defect one layer down (same doctrine as
            # prompt_surface._refuse_if_image_batch_exceeds_declared_window,
            # but bounded here by the caller's declared max_length budget).
            # Refuse, naming BOTH numbers, never corrupt.
            _audio_refuse_exit_96(
                f"audio column {audio_column!r}: batch encoded to {width} tokens "
                f"wider than the declared max_length={max_length}. The audio path "
                "does not truncate -- truncating audio placeholders drops measured "
                "sound from the training input, the silent-drop defect. Reduce the "
                "audio length per row, the text fields, or the max_length budget; "
                "this refuses rather than corrupts."
            )

        audio_token_id = getattr(processor, "audio_token_id", None)
        if audio_token_id is None:
            _audio_refuse_exit_96(
                f"audio column {audio_column!r} put audio in the batch, but the "
                "processor declares no audio_token_id. Without it the placeholder "
                "positions cannot be counted (for verification) or masked out of "
                "the labels (for supervision), and training would optimise the "
                "model to emit placeholder tokens as text -- the objective would "
                "be mostly not the task. Refusing rather than training against "
                "an objective we cannot verify."
            )

        # PER-ROW placeholder verification. Gemma-4's
        # _compute_audio_num_tokens(len(wave)) is the processor's OWN prediction
        # of how many audio placeholder tokens the template inserted for this
        # waveform. If the ACTUAL count in input_ids disagrees, the processor
        # and the model pair are out of contract and the batch cannot be
        # trusted to line up with the tower's valid output length (P0 measured
        # these equal on gemma-4: 147/121/312/248 on both sides).
        # When the method is ABSENT the row is counted as UNMEASURED on the
        # coverage record -- not skipped silently and not reported as a pass.
        expected_count_fn: Any = getattr(processor, "_compute_audio_num_tokens", None)
        callable_expected = callable(expected_count_fn)
        for index, wave in enumerate(waves):
            row_id = f"train-row[{index}]"
            actual = int((input_ids[index] == audio_token_id).sum())
            if not callable_expected:
                coverage.add_placeholder_unmeasured()
                continue
            # Signature measured on transformers 5.5.0 (processing_gemma4.py):
            # _compute_audio_num_tokens(audio_waveform, sampling_rate) -> int.
            expected = int(expected_count_fn(wave, int(processor.feature_extractor.sampling_rate)))
            if actual != expected:
                _audio_refuse_exit_96(
                    f"audio column {audio_column!r} row {row_id!r}: "
                    f"apply_chat_template produced {actual} audio placeholders, but "
                    "the processor's own _compute_audio_num_tokens(<"
                    f"{len(wave)} samples>) expects {expected}. The processor and the model "
                    "pair are out of contract (P0 measured these equal on gemma-4); "
                    "refusing to train against a sequence the family does not "
                    "recognise."
                )
            coverage.add_placeholder_verified()

        # Prompt boundary. Row i's prompt is `prompt_batch[i]`'s un-padded
        # length -- everything after it in `input_ids[i]` is the assistant
        # target. Two apply_chat_template calls per batch (prompt-only with
        # add_generation_prompt=True and full with the default) instead of one
        # because the single-call alternative has no portable way to recover
        # the boundary index without guessing token-ids or offsets. The prompt
        # and full renderings share a prefix by the chat-template contract
        # (a template that renders the user turn differently once an assistant
        # turn follows would metastasise labels silently -- an assumption this
        # plan refuses to make without a check).
        prompt_batch = processor.apply_chat_template(
            prompt_convos,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
            # transformers 5.5 routes processor arguments through
            # processor_kwargs; passed loose they still apply, after a warning
            # on every batch.
            processor_kwargs={"padding": True},
            add_generation_prompt=True,
        )
        prompt_attention = prompt_batch.get("attention_mask")
        if prompt_attention is None:
            _audio_refuse_exit_96(
                f"audio column {audio_column!r}: the prompt-only tokenize produced "
                "no attention_mask, so the row's prompt length cannot be recovered "
                "and the label mask would mark the wrong tokens as supervised. "
                "Refusing rather than supervising the prompt by accident."
            )
        prompt_lens = [int(prompt_attention[i].sum()) for i in range(len(prompt_attention))]

        # Labels: clone so an in-place causal-shift cannot corrupt input_ids,
        # then mask (a) the prompt (whose slice includes the audio placeholders
        # and the assistant-turn opener), (b) any position with attention_mask
        # == 0 (pad; deliberately mask-based rather than pad-id equality since
        # gemma-4's pad id is 0 and 0 is a legal content id -- the #450 lesson),
        # (c) every audio_token_id, since a placeholder is a valid INPUT slot
        # but never a valid TARGET.
        labels = input_ids.clone()
        attention_mask = full_batch.get("attention_mask")
        if attention_mask is None:
            _audio_refuse_exit_96(
                f"audio column {audio_column!r}: the full tokenize produced no "
                "attention_mask, so padding cannot be masked out of the labels and "
                "the model would be trained to EMIT pad tokens. Refusing rather "
                "than training against an objective that is not the task."
            )
        for index, prompt_len in enumerate(prompt_lens):
            # The prompt slice starts at the first ATTENDED position so this is
            # correct for right-padded and left-padded batches alike (HF's
            # training default is right, but a left-padded caller is not a
            # silent-corruption case).
            first_attended = int(attention_mask[index].argmax())
            labels[index, first_attended : first_attended + prompt_len] = -100
        labels[attention_mask == 0] = -100
        labels[labels == audio_token_id] = -100
        full_batch["labels"] = labels

        # Last gate before returning: if the processor quietly dropped
        # input_features (a text-only encode sneaking in, a stub forgetting the
        # key), the run must not train text-only under an audio label. Same
        # idea as refuse_if_pixel_column_dropped on the image plane.
        refuse_if_audio_features_dropped(full_batch.keys(), audio_column)

        for duration in durations:
            coverage.record_ok(duration)
        return dict(full_batch)

    # Functions cannot nominally carry an attribute per PEP 544; at runtime,
    # assigning .coverage on the callable and duck typing
    # satisfies AudioCollator. The `# type: ignore` markers say so to the
    # type-checker without polluting the AudioCollator protocol with callable
    # implementation details.
    collate.coverage = coverage  # type: ignore[attr-defined]
    return collate  # type: ignore[return-value]


def audio_full_train_modules(family: Any) -> list[str]:
    """The family's measured full-train modules that belong to its AUDIO towers.

    Read from ``FamilySpec.adapter_full_train`` (see its docstring for why this is a
    measured, per-family declaration). An empty result means the family has not
    measured it, and the caller refuses an adapter run that declares audio.
    """
    if family is None:
        return []
    audio_towers = [prefix for prefix, modality in family.towers if modality == "audio"]
    return [
        module
        for module in getattr(family, "adapter_full_train", ())
        if any(module == t or module.startswith(t + ".") for t in audio_towers)
    ]
