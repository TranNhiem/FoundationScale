"""The worker-side audio census for the NeMo (Canary) lane: which rows may the trainer load?

The question this module answers is "is this audio what the row claims?" -- 16 kHz mono, between
1 and ``max_duration`` seconds, non-empty text, and no row reusing a recording a previous row
already claimed. The reader-side gate (``manifest_row_problems``, run by ``read_speech_manifest``)
asked a DIFFERENT question ("is the row well formed?") and answered it from the JSON alone.
Keeping the two halves apart is what lets the control plane validate manifests for campaigns it
cannot play back, and what keeps the refusal vocabulary honest.

The refusal ladder is the one ``validation_campaigns/speech_canary/nemo_finetune.convert`` used,
in the SAME ORDER (see :data:`REFUSAL_ORDER`): ``duplicate_audio_path``,
``sample_rate_mismatch``, ``not_mono``, ``duration_out_of_range`` (``1.0 <= duration <=
max_duration``), ``empty_text``. Order is NOT cosmetic -- a row with several faults is counted
under the first reason only, and any other order would silently reclassify existing runs'
refusal counts. A row that is both duplicated AND 8 kHz is a ``duplicate_audio_path`` and
nothing else.

Kept rows carry the MEASURED ``duration`` (from ``info``), not the manifest's ``duration``
field: the NeMo AED manifest's ``duration`` column is what NeMo's batcher reads, and the
original ``convert`` wrote ``info.duration`` there. That this may differ from the manifest value
is not a bug -- it is the measurement the audio stack is uniquely positioned to take.

``info`` is injected (DI) so this module runs in plain CI against fakes; the default opens the
file via ``soundfile.info`` and lives in :func:`default_info` AS A FUNCTION BODY, so ``import
soundfile`` stays lazy and the module imports in a plain CI environment.

``rows`` is expected to already be reader-validated (post ``read_speech_manifest``). The empty
text check remains here as defence-in-depth and because the refusal ladder must match the
original ``convert``'s exactly; ``read_speech_manifest`` will already have caught a
whitespace-only ``answer`` as its own ``empty_answer`` problem and refused the row before it
reaches us.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import Any

__all__ = [
    "InfoFn",
    "MIN_DURATION",
    "REFUSAL_ORDER",
    "REQUIRED_SAMPLE_RATE",
    "census",
    "default_info",
]


REQUIRED_SAMPLE_RATE = 16000
"""Only 16 kHz mono enters the NeMo lane. The HF eval refuses to resample and so does this one
-- a silent resample misaligns the target and no metric downstream would say so."""

MIN_DURATION = 1.0
"""Clips shorter than one second carry too little signal to be worth the trainer's batch slot."""

REFUSAL_ORDER: tuple[str, ...] = (
    "duplicate_audio_path",
    "sample_rate_mismatch",
    "not_mono",
    "duration_out_of_range",
    "empty_text",
)
"""The refusal ladder, in the order the original ``convert`` checked it. Order is part of the
contract: a row with several faults is counted under the first reason only, so rearranging the
ladder would silently reclassify every existing run's refusal buckets."""

InfoFn = Callable[[str], tuple[int, int, float]]
"""``info(path) -> (samplerate, channels, duration)``. Injected so the census runs against
fakes in CI; the default is :func:`default_info` and is the only place ``soundfile`` appears."""


def default_info(path: str) -> tuple[int, int, float]:
    """Open ``path`` with ``soundfile.info``; return ``(samplerate, channels, duration)``.

    This is the ONLY place in :mod:`census` that touches a wave file -- and it is a function
    body specifically so ``import soundfile`` stays lazy. The module imports in a plain CI
    environment with no soundfile installed.
    """
    import soundfile  # type: ignore[import-untyped]

    i = soundfile.info(path)
    return (i.samplerate, i.channels, i.duration)


def census(
    rows: Sequence[Mapping[str, Any]],
    *,
    max_duration: float,
    info: InfoFn | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Split ``rows`` into what the NeMo trainer may load, plus a coverage dict.

    ``rows`` is expected to already be reader-validated (post ``read_speech_manifest``). Every
    refusal is COUNTED under its stable reason code (see :data:`REFUSAL_ORDER`) and the count
    lands in ``coverage["refused"]``; no row is ever silently dropped.

    Returns ``(kept_rows, coverage)``.

    * ``kept_rows[i]`` is a fresh ``dict`` of the i-th kept row with ``duration`` REPLACED by the
      measured ``info.duration`` -- see the module docstring for why. That row is what
      :func:`to_nemo_aed_row` is called on, so the NeMo manifest's ``duration`` column is the
      MEASURED value, identical to what the original ``convert`` wrote.
    * ``coverage`` has EXACTLY the four keys the original produced: ``rows_expected`` (rows
      seen), ``rows_checked`` (rows the trainer will load), ``rows_refused``, ``refused``
      (reason -> count; only reasons that fired). Those four are the entirety of the campaign's
      ``coverage.json`` and the input to ``AudioRowCoverageGate`` on the adjudication side.
    """
    if info is None:
        info = default_info
    refused: dict[str, int] = {}
    kept_rows: list[dict[str, Any]] = []
    seen_paths: set[str] = set()
    for r in rows:
        samplerate, channels, duration = info(r["audio"])
        reason: str | None = None
        # A repeated path means two transcripts share one recording: every row still "loads",
        # so row coverage cannot see it. Measured: a builder bug left 6,000 rows on 820 files.
        if r["audio"] in seen_paths:
            reason = "duplicate_audio_path"
        elif samplerate != REQUIRED_SAMPLE_RATE:
            reason = "sample_rate_mismatch"
        elif channels != 1:
            reason = "not_mono"
        elif not (MIN_DURATION <= duration <= max_duration):
            reason = "duration_out_of_range"
        elif not str(r.get("answer", "")).strip():
            reason = "empty_text"
        if reason is not None:
            refused[reason] = refused.get(reason, 0) + 1
            continue
        seen_paths.add(r["audio"])
        kept_row = dict(r)
        kept_row["duration"] = (
            duration  # measured from the audio, not the manifest -- see mod docstring
        )
        kept_rows.append(kept_row)
    coverage: dict[str, Any] = {
        "rows_expected": len(rows),
        "rows_checked": len(kept_rows),
        "rows_refused": sum(refused.values()),
        "refused": refused,
    }
    return kept_rows, coverage
