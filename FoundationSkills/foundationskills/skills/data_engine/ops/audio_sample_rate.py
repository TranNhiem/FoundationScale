"""``audio_sample_rate`` op: MEASURE each speech row's audio and keep only the declared sample rate (A4.3).

WHY (measured): the FoundationScale speech manifest schema carries no sample-rate field and FS deliberately never reads
audio (``foundationscale.upstream.contracts``), so nothing enforced the 16 kHz the ASR recipes assume. The Earnings-22
long-form set illustrates the risk: its metadata says 16 kHz, but 4 calls were 24 kHz, 1 was 22.05 kHz stereo and
1 was 16 kHz stereo (validation_campaigns/speech_longform/EVIDENCE.md:8-12).

For every record the op reads ``record["audio"]`` with ``soundfile.info`` (header only, no decode):
- ``rate == sample_rate`` (default 16000): kept unchanged;
- any other rate: dropped as ``sample_rate_not_<rate>``;
- missing or unreadable file: dropped as ``audio_unreadable``;
- no ``audio`` field: dropped as ``audio_missing_field``.

``stats.extra`` records what was MEASURED, never assumed: ``sample_rate`` (the required rate), ``sample_rate_hist`` and
``channels_hist`` over every readable file, ``measured`` (files read), ``kept_seconds`` (sum of measured frames / rate
over kept rows) and ``duration_mismatch`` (kept rows whose manifest ``duration`` differs from the measured one by more
than ``duration_tolerance_s``). The readiness check DE-RDY-012 reads these.

Refusal (``OpUnavailable`` -> DE-IN-008): ``soundfile`` is not importable here.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable, Iterator

from foundationskills.skills.data_engine.ops.base import FunctionOp, OpStats, OpUnavailable, counted, register_op

CONFIG_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,  # unknown keys refused: a silently ignored key measures nothing
    "properties": {
        "sample_rate": {"type": "integer", "minimum": 1, "default": 16000},
        "duration_tolerance_s": {"type": "number", "minimum": 0, "default": 0.5},
    },
}


def preflight(cfg: dict) -> list[str]:  # noqa: ARG001 - the signature every runtime op exports
    """Cheap offline check; [] means the op can measure audio here."""
    try:
        import soundfile  # noqa: F401, PLC0415
    except ImportError:
        return ["soundfile is not importable (pip install soundfile): audio sample rates cannot be measured"]
    return []


def _audio_sample_rate(records: Iterable[dict], cfg: dict, stats: OpStats) -> Iterator[dict]:
    try:
        import soundfile  # noqa: PLC0415
    except ImportError as exc:
        raise OpUnavailable("audio_sample_rate: soundfile is not importable; sample rates cannot be measured") from exc
    want = int(cfg.get("sample_rate", 16000))
    tolerance = float(cfg.get("duration_tolerance_s", 0.5))
    rates: dict[str, int] = {}
    channels: dict[str, int] = {}
    measured = 0
    kept_seconds = 0.0
    mismatch = 0
    for record in counted(records, stats):
        audio = record.get("audio") if isinstance(record, dict) else None
        if not isinstance(audio, str) or not audio:
            stats.drop("audio_missing_field")
            continue
        try:
            info = soundfile.info(str(Path(audio)))
        except Exception:  # noqa: BLE001 - any unreadable file is a measured drop, never a silent keep
            stats.drop("audio_unreadable")
            continue
        measured += 1
        rates[str(info.samplerate)] = rates.get(str(info.samplerate), 0) + 1
        channels[str(info.channels)] = channels.get(str(info.channels), 0) + 1
        if int(info.samplerate) != want:
            stats.drop(f"sample_rate_not_{int(info.samplerate)}")
            continue
        seconds = float(info.frames) / float(info.samplerate) if info.samplerate else 0.0
        kept_seconds += seconds
        declared = record.get("duration")
        if isinstance(declared, (int, float)) and not isinstance(declared, bool) and abs(seconds - declared) > tolerance:
            mismatch += 1
        stats.records_out += 1
        yield record
    stats.backend = "soundfile"
    stats.extra.update(sample_rate=want, sample_rate_hist=rates, channels_hist=channels, measured=measured,
                       kept_seconds=round(kept_seconds, 3), duration_mismatch=mismatch)


register_op(FunctionOp("audio_sample_rate", _audio_sample_rate, CONFIG_SCHEMA))
