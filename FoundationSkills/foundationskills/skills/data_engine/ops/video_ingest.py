"""``video_ingest`` op: video records -> multimodal text/message records.

- Records carry a video path in ``rec["video"]`` (or ``rec["meta"]["video"]``).
  Records without one pass through untouched (counted in
  ``stats.extra["passthrough_non_video"]``) unless ``require_video`` drops them
  as ``no_video``: a mixed corpus must not be silently discarded just because
  the target modality wants video.
- Probing runs ``ffprobe -show_entries format=duration -of json``. A missing
  file is ``video_missing``, an unusable/absent duration is ``probe_failed``
  and a clip longer than ``max_duration_s`` is ``too_long`` (one 2 h lecture
  would otherwise eat the whole frame budget and hide everything else).
- Frames are sampled on the mid-interval grid ``t_k = (k + 0.5) * duration / K``
  with ``K = min(max_frames, max(1, floor(duration * fps)))`` -- never an exact
  endpoint, so a truncated final GOP cannot silently yield zero frames. Each
  frame is extracted with ffmpeg into
  ``<frames_dir>/<sha256(abs path + mtime_ns + size)[:16]>/frame_<k>.jpg``: the
  key hashes the source *fingerprint* so a re-encoded clip is never confused
  with a cached one, while re-runs over the same file reuse the frames
  (``frames_cached``). Zero prepared frames -> ``no_frames``.
- ASR (``asr: whisper``) transcribes the mono 16 kHz float32 audio decoded by
  ffmpeg through ``transformers``' ASR pipeline, wrapped in the injectable
  ``_ASR_FACTORY(cfg)`` hook (tests monkeypatch it with fakes) so no model
  weights are needed at test time; the model is loaded once per op call. A clip
  without an audio stream is *not* an error (``stats.extra["no_audio"]``, empty
  transcript), but a clip whose audio stream exists and fails to decode drops as
  ``audio_extract_failed``; a failing ASR run drops the record as ``asr_failed`` and records
  the first error line (max 5) instead of pretending the video is silent.
  Segments whose text zlib-compresses by more than ``max_compression_ratio``
  (default 2.4, Whisper's own fallback threshold) are repetition-loop
  hallucinations on non-speech audio and are suppressed
  (``stats.extra["asr_segments_suppressed"]``).
- Alignment: a segment lands in the frame interval ``[t - half_gap,
  t + half_gap)`` containing its midpoint (the grid tiles ``[0, duration)``, so
  in the plain case every segment gets a frame; a midpoint outside these
  intervals -- only possible inside a chunk window -- falls back to the nearest
  frame instead of dropping transcript text).
- Output replaces the record with ``{"id", "images", "text"|"messages", "meta"}``
  where ``meta`` keeps the original meta and adds ``video``, ``duration_s``,
  ``frame_times_s``, ``asr`` and ``transcript``. ``chunk_seconds`` splits the
  clip into window records ``<id>#c<j>`` (windows without a frame emit
  nothing); the extra generated records are counted in
  ``stats.extra["chunks_emitted"]`` and the accounting invariant becomes
  ``records_in + chunks_extra == records_out + dropped`` (asserted at the end of
  every run): ``chunks_extra`` is the number of generated output windows beyond
  the first per processed record, including windows later dropped as
  ``empty_transcript``, so record-level and unit-level drops stay countable.
- ``emit: messages`` renders one user turn ("<image>" * K + prompt) plus an
  assistant turn with the transcript text; a messages record without any
  transcript text is unlearnable and dropped as ``empty_transcript``.
- Frames and ASR are the only heavy parts; :func:`preflight` checks binaries,
  the absolute frames directory and the local model cache strictly offline, so
  a missing dependency is refused up front instead of failing mid-run.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
import subprocess
import zlib
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator

from foundationskills.skills.data_engine.ops.base import (
    FunctionOp,
    OpStats,
    OpUnavailable,
    counted,
    register_op,
)


CONFIG_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,  # unknown keys refused: a silently ignored key measures nothing
    "required": ["frames_dir"],
    "properties": {
        "frames_dir": {"type": "string", "minLength": 1},
        "ffmpeg": {"type": "string", "minLength": 1},
        "ffprobe": {"type": "string", "minLength": 1},
        "fps": {"type": "number", "exclusiveMinimum": 0.0},
        "max_frames": {"type": "integer", "minimum": 1},
        "frame_width": {"type": "integer", "minimum": 2},
        "max_duration_s": {"type": "number", "exclusiveMinimum": 0.0},
        "require_video": {"type": "boolean"},
        "asr": {"enum": ["none", "whisper"]},
        "asr_model": {"type": "string", "minLength": 1},
        "asr_device": {"enum": ["auto", "cpu", "cuda"]},
        "language": {"type": ["string", "null"]},
        "local_files_only": {"type": "boolean"},
        "chunk_seconds": {"type": "number", "exclusiveMinimum": 0.0},
        "max_compression_ratio": {"type": "number", "exclusiveMinimum": 1.0},
        "emit": {"enum": ["text", "messages"]},
        "prompt": {"type": "string"},
    },
}

AsrFn = Callable[[Any, int], list[dict]]
AsrFactory = Callable[[dict], AsrFn]

_DEFAULT_PROMPT = "Describe what happens in this video."
_MAX_ASR_ERRORS = 5
# Whisper's own decode-fallback threshold: text that zlib squeezes more than this is
# a repetition loop ("Beeeee...", "Thank you. Thank you. ...") -- a hallucination on
# non-speech audio, not a transcript.
_DEFAULT_MAX_COMPRESSION_RATIO = 2.4


def _compression_ratio(text: str) -> float:
    raw = text.encode("utf-8")
    return len(raw) / len(zlib.compress(raw)) if raw else 0.0


# --------------------------------------------------------------------------
# binaries / device / model resolution
# --------------------------------------------------------------------------

def _find_binary(name: str) -> str | None:
    """Resolve ``name`` to an executable (PATH lookup, or an explicit path)."""
    if not name:
        return None
    if "/" in name or os.sep in name:
        candidate = Path(name).expanduser()
        return str(candidate) if candidate.is_file() and os.access(candidate, os.X_OK) else None
    return shutil.which(name)


def _require_binary(name: str, label: str) -> str:
    found = _find_binary(name)
    if found is None:
        raise OpUnavailable(f"video_ingest: {label} binary not resolvable/executable ({name!r})")
    return found


def _resolve_device(device: str) -> Any:
    if device == "cpu":
        return "cpu"
    if device == "cuda":
        return "cuda"
    try:
        import torch  # type: ignore

        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:  # noqa: BLE001 - no torch means no CUDA; CPU is the honest fallback
        return "cpu"


def _local_model(model: str, local_only: bool) -> tuple[str | None, str | None]:
    """(resolved model path, problem text). Local resolution never downloads."""
    if Path(model).is_dir():
        return str(Path(model)), None
    if not local_only:
        return model, None
    try:
        from huggingface_hub import snapshot_download  # type: ignore

        resolved = snapshot_download(model, local_files_only=True)
    except Exception as exc:  # noqa: BLE001 - cache misses come in several exception shapes
        problem = (
            f"asr model {model!r} is not in the local HuggingFace cache and local_files_only=True "
            f"({type(exc).__name__}: {exc})"
        )
        return None, problem
    return str(resolved), None


def _ffmpeg_version(ffmpeg: str) -> str:
    try:
        proc = subprocess.run([ffmpeg, "-version"], capture_output=True, text=True, timeout=30)
    except (OSError, subprocess.SubprocessError):
        return "unknown"
    lines = (proc.stdout or "").splitlines()
    return lines[0].strip() if lines else "unknown"


# --------------------------------------------------------------------------
# probe / frames / audio / ASR
# --------------------------------------------------------------------------

def _probe_duration(ffprobe: str, path: Path) -> float | None:
    """Container duration in seconds; None when ffprobe cannot report one."""
    try:
        proc = subprocess.run(
            [ffprobe, "-v", "error", "-show_entries", "format=duration", "-of", "json", str(path)],
            capture_output=True, text=True, timeout=120,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if proc.returncode != 0:
        return None
    try:
        payload = json.loads(proc.stdout or "{}")
    except json.JSONDecodeError:
        return None
    section = payload.get("format") if isinstance(payload, dict) else None
    raw = section.get("duration") if isinstance(section, dict) else None
    try:
        duration = float(raw)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    if not math.isfinite(duration) or duration <= 0.0:
        return None
    return duration


def _frame_times(duration: float, max_frames: int, fps: float) -> list[float]:
    """Mid-interval grid ``t_k = (k + 0.5) * duration / K``."""
    n = max(1, math.floor(duration * fps))
    n = min(max_frames, n)
    return [(k + 0.5) * duration / n for k in range(n)]


def _frame_cache_dir(frames_root: Path, path: Path) -> Path:
    """Cache key = source fingerprint (absolute path + mtime_ns + size)."""
    stat = path.stat()
    token = f"{path.resolve()}|{stat.st_mtime_ns}|{stat.st_size}"
    return frames_root / hashlib.sha256(token.encode("utf-8")).hexdigest()[:16]


def _prepare_frames(
    ffmpeg: str, path: Path, cache_dir: Path, times: list[float], width: int, stats: OpStats
) -> list[tuple[float, str]]:
    """Extract (or reuse) every planned frame; failures are simply absent frames.

    The file name carries the timestamp and width, so a run with another sampling
    plan never relabels a cached frame from a different instant.
    """
    cache_dir.mkdir(parents=True, exist_ok=True)
    prepared: list[tuple[float, str]] = []
    for t in times:
        target = cache_dir / f"frame_{t:010.3f}s_w{width}.jpg"
        if target.is_file() and target.stat().st_size > 0:
            stats.extra["frames_cached"] += 1
            prepared.append((t, str(target)))
            continue
        cmd = [
            ffmpeg, "-v", "error", "-ss", f"{t:.3f}", "-i", str(path),
            "-frames:v", "1", "-vf", f"scale={width}:-2", "-q:v", "3", "-y", str(target),
        ]
        try:
            proc = subprocess.run(cmd, capture_output=True, timeout=300)
        except (OSError, subprocess.SubprocessError):
            continue
        if proc.returncode == 0 and target.is_file() and target.stat().st_size > 0:
            stats.extra["frames_written"] += 1
            prepared.append((t, str(target)))
        elif target.is_file():
            target.unlink()  # never accidentally reuse a half-written frame
    return prepared


def _has_audio(ffprobe: str, path: Path) -> bool | None:
    """True/False from ffprobe's stream list; None when ffprobe itself fails."""
    cmd = [ffprobe, "-v", "error", "-select_streams", "a", "-show_entries", "stream=index", "-of", "csv=p=0", str(path)]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
    except (OSError, subprocess.SubprocessError):
        return None
    return bool(proc.stdout.strip()) if proc.returncode == 0 else None


def _extract_audio(ffmpeg: str, ffprobe: str, path: Path) -> tuple[bytes | None, str | None]:
    """Mono 16 kHz float32 PCM -> (pcm, None); (None, None) = no audio stream; (None, error) = extraction failed.

    A clip without an audio stream is legitimate; a broken audio stream is not, and
    must not hide under the ``no_audio`` counter as an empty transcript.
    """
    has_audio = _has_audio(ffprobe, path)
    if has_audio is False:
        return None, None
    cmd = [ffmpeg, "-v", "error", "-i", str(path), "-vn", "-ac", "1", "-ar", "16000", "-f", "f32le", "-"]
    try:
        proc = subprocess.run(cmd, capture_output=True, timeout=600)
    except (OSError, subprocess.SubprocessError) as exc:
        return None, f"{type(exc).__name__}: {exc}"
    if proc.returncode != 0 or not proc.stdout:
        err = proc.stderr.decode("utf-8", errors="replace").strip().splitlines()
        return None, f"ffmpeg audio exit {proc.returncode}: {err[0] if err else 'no PCM output'}"
    return proc.stdout, None


def _norm_segment(raw: Any) -> dict | None:
    if not isinstance(raw, dict):
        return None
    text = raw.get("text")
    text = text if isinstance(text, str) else ("" if text is None else str(text))
    try:
        start = float(raw.get("start", 0.0))
    except (TypeError, ValueError):
        start = 0.0
    try:
        end = float(raw.get("end", start))
    except (TypeError, ValueError):
        end = start
    if end < start:
        end = start
    return {"start": start, "end": end, "text": text.strip()}


def _segments_from_result(result: Any, sr: int, n_samples: int) -> list[dict]:
    """Flatten a transformers ASR result (``chunks`` with ``timestamp`` tuples) into segments."""
    total = (n_samples / sr) if sr else 0.0
    if isinstance(result, list):
        flat: list[dict] = []
        for item in result:
            flat.extend(_segments_from_result(item, sr, n_samples))
        return flat
    if isinstance(result, dict) and isinstance(result.get("chunks"), list):
        raw_chunks = result["chunks"]
    elif isinstance(result, dict):
        raw_chunks = [result]
    else:
        return []
    segments: list[dict] = []
    for chunk in raw_chunks:
        if not isinstance(chunk, dict):
            continue
        ts = chunk.get("timestamp")
        start = end = None
        if isinstance(ts, (tuple, list)) and len(ts) == 2:
            start, end = ts
        start_f = float(start) if isinstance(start, (int, float)) else 0.0
        end_f = float(end) if isinstance(end, (int, float)) else max(total, start_f)
        segment = _norm_segment({"start": start_f, "end": end_f, "text": chunk.get("text", "")})
        if segment is not None:
            segments.append(segment)
    return segments


def _build_whisper_asr(cfg: dict) -> AsrFn:
    """Default ``_ASR_FACTORY``: a transformers whisper pipeline (offline when asked)."""
    try:
        from transformers import pipeline as hf_pipeline  # type: ignore
    except Exception as exc:  # noqa: BLE001 - missing/broken transformers is a dependency problem
        raise OpUnavailable(f"video_ingest: 'transformers' is not importable for asr='whisper' "
                            f"({type(exc).__name__}: {exc})") from exc
    model = str(cfg.get("asr_model", "openai/whisper-small"))
    local_only = bool(cfg.get("local_files_only", True))
    resolved, problem = _local_model(model, local_only)
    if problem is not None:
        raise OpUnavailable(f"video_ingest: {problem}")
    device = _resolve_device(str(cfg.get("asr_device", "auto")))
    try:
        pipe = hf_pipeline(
            "automatic-speech-recognition",
            model=resolved or model,
            device=device,
            chunk_length_s=30,
            return_timestamps=True,
        )
    except Exception as exc:  # noqa: BLE001 - load failures must name the dependency, not vanish
        raise OpUnavailable(f"video_ingest: ASR model {model!r} failed to load "
                            f"({type(exc).__name__}: {exc})") from exc
    language = cfg.get("language") if isinstance(cfg.get("language"), str) else None
    if language:
        try:  # whisper prefix conditioning; harmless when the tokenizer refuses
            pipe.tokenizer.set_prefix_tokens(language=language)
        except Exception:  # noqa: BLE001
            pass

    def run(audio: Any, sr: int) -> list[dict]:
        result = pipe({"raw": audio, "sampling_rate": sr})
        return _segments_from_result(result, sr, len(audio))

    return run


_ASR_FACTORY: AsrFactory = _build_whisper_asr


# --------------------------------------------------------------------------
# record assembly
# --------------------------------------------------------------------------

def _video_path(rec: dict) -> str | None:
    video = rec.get("video")
    if isinstance(video, str) and video.strip():
        return video
    meta = rec.get("meta")
    if isinstance(meta, dict):
        meta_video = meta.get("video")
        if isinstance(meta_video, str) and meta_video.strip():
            return meta_video
    return None


def _midpoint(seg: dict) -> float:
    start = float(seg.get("start", 0.0))
    end = float(seg.get("end", start))
    return (start + end) / 2.0


def _attach_segments(segments: list[dict], frame_times: list[float], half_gap: float) -> dict[int, list[dict]]:
    """Segment -> frame slot by the ``[t - half_gap, t + half_gap)`` rule (nearest frame as fallback)."""
    buckets: dict[int, list[dict]] = {}
    for seg in segments:
        mid = _midpoint(seg)
        slot: int | None = None
        for i, t in enumerate(frame_times):
            if t - half_gap <= mid < t + half_gap:
                slot = i
                break
        if slot is None:
            slot = min(range(len(frame_times)), key=lambda i: (abs(frame_times[i] - mid), frame_times[i]))
        buckets.setdefault(slot, []).append(seg)
    return buckets


def _render_text(frame_pairs: list[tuple[float, str]], buckets: dict[int, list[dict]]) -> str:
    lines: list[str] = []
    for i, (t, _target) in enumerate(frame_pairs):
        lines.append(f"<frame {i + 1} @ {t:.1f}s>")
        for seg in buckets.get(i, []):
            if seg["text"]:
                lines.append(seg["text"])
    return "\n".join(lines)


def _video_ingest_op(records: Iterable[dict], cfg: dict, stats: OpStats) -> Iterator[dict]:
    frames_root = Path(str(cfg.get("frames_dir", "")))
    if not frames_root.is_absolute():
        raise ValueError(f"video_ingest: frames_dir must be an absolute path, got {str(frames_root)!r}")

    ffmpeg = _require_binary(str(cfg.get("ffmpeg", "ffmpeg")), "ffmpeg")
    ffprobe = _require_binary(str(cfg.get("ffprobe", "ffprobe")), "ffprobe")

    fps = float(cfg.get("fps", 0.5))
    max_frames = int(cfg.get("max_frames", 16))
    frame_width = int(cfg.get("frame_width", 448))
    max_duration_s = float(cfg.get("max_duration_s", 3600.0))
    require_video = bool(cfg.get("require_video", False))
    asr_kind = str(cfg.get("asr", "whisper"))
    asr_model = str(cfg.get("asr_model", "openai/whisper-small"))
    raw_language = cfg.get("language")
    language = raw_language if isinstance(raw_language, str) and raw_language else None
    chunk_seconds = cfg.get("chunk_seconds")
    chunk_seconds = float(chunk_seconds) if chunk_seconds is not None else None
    emit = str(cfg.get("emit", "text"))
    prompt = str(cfg.get("prompt", _DEFAULT_PROMPT))

    stats.backend = f"ffmpeg+{'no-asr' if asr_kind == 'none' else 'transformers-asr'}:{asr_model}"
    extra = stats.extra
    extra.setdefault("frames_written", 0)
    extra.setdefault("frames_cached", 0)
    extra.setdefault("total_duration_s", 0.0)
    extra.setdefault("no_audio", 0)
    extra.setdefault("asr_errors", [])
    extra.setdefault("passthrough_non_video", 0)
    extra.setdefault("chunks_emitted", 0)
    extra.setdefault("chunks_extra", 0)
    extra.setdefault("asr_segments_suppressed", 0)
    max_ratio = float(cfg.get("max_compression_ratio", _DEFAULT_MAX_COMPRESSION_RATIO))
    extra["ffmpeg_version"] = _ffmpeg_version(ffmpeg)

    asr_cache: list[AsrFn] = []

    def asr_fn() -> AsrFn:
        if not asr_cache:  # model weights are loaded once per op call, and only when needed
            asr_cache.append(_ASR_FACTORY(cfg))
        return asr_cache[0]

    for rec in counted(records, stats):
        if not isinstance(rec, dict):
            stats.drop("non_dict_record")
            continue
        video = _video_path(rec)
        if video is None:
            if require_video:
                stats.drop("no_video")
                continue
            extra["passthrough_non_video"] += 1
            stats.records_out += 1
            yield rec
            continue

        path = Path(video)
        if not path.is_file():
            stats.drop("video_missing")
            continue
        duration = _probe_duration(ffprobe, path)
        if duration is None:
            stats.drop("probe_failed")
            continue
        if duration > max_duration_s:
            stats.drop("too_long")
            continue

        planned = _frame_times(duration, max_frames, fps)
        frame_pairs = _prepare_frames(ffmpeg, path, _frame_cache_dir(frames_root, path), planned, frame_width, stats)
        if not frame_pairs:
            stats.drop("no_frames")
            continue
        extra["total_duration_s"] += duration

        segments: list[dict] = []
        if asr_kind != "none":
            audio_bytes, audio_error = _extract_audio(ffmpeg, ffprobe, path)
            if audio_error is not None:
                stats.drop("audio_extract_failed")
                if len(extra["asr_errors"]) < _MAX_ASR_ERRORS:
                    extra["asr_errors"].append(f"{path}: {audio_error}".splitlines()[0])
                continue
            if audio_bytes is None:
                extra["no_audio"] += 1
            else:
                try:
                    import numpy as np  # type: ignore
                except ImportError as exc:
                    raise OpUnavailable("video_ingest: numpy is required for ASR audio buffers") from exc
                audio = np.frombuffer(audio_bytes, dtype=np.float32).copy()  # writable: backends may window in place
                try:
                    raw_segments = list(asr_fn()(audio, 16000))
                except OpUnavailable:
                    raise
                except Exception as exc:  # noqa: BLE001 - one broken clip must not kill the run
                    stats.drop("asr_failed")
                    errors = extra["asr_errors"]
                    if len(errors) < _MAX_ASR_ERRORS:
                        errors.append(f"{path}: {type(exc).__name__}: {exc}".splitlines()[0])
                    continue
                segments = [s for s in (_norm_segment(x) for x in raw_segments) if s is not None]
                kept = [s for s in segments if _compression_ratio(s["text"]) <= max_ratio]
                extra["asr_segments_suppressed"] += len(segments) - len(kept)
                segments = kept

        rid_raw = rec.get("id")
        rid = str(rid_raw) if rid_raw is not None else f"rec-{stats.records_in}"

        if chunk_seconds is None:
            units: list[tuple[str, list[tuple[float, str]], list[dict]]] = [(rid, frame_pairs, segments)]
        else:
            units = []
            n_windows = max(1, math.ceil(duration / chunk_seconds))
            for j in range(n_windows):
                lo, hi = j * chunk_seconds, (j + 1) * chunk_seconds
                unit_frames = [(t, p) for (t, p) in frame_pairs if lo <= t < hi]
                if not unit_frames:  # windows without a frame emit nothing at all
                    continue
                unit_segs = [s for s in segments if lo <= _midpoint(s) < hi]
                units.append((f"{rid}#c{j}", unit_frames, unit_segs))

        half_gap = duration / (2.0 * max(1, len(planned)))
        meta_base = dict(rec.get("meta")) if isinstance(rec.get("meta"), dict) else {}
        for unit_id, unit_frames, unit_segs in units:
            images = [p for _t, p in unit_frames]
            out_meta = dict(meta_base)
            out_meta.update({
                "video": video,
                "duration_s": duration,
                "frame_times_s": [t for t, _p in unit_frames],
                "asr": {
                    "model": asr_model if asr_kind != "none" else "no-asr",
                    "language": language,
                    "segments": len(unit_segs),
                },
                "transcript": unit_segs,
            })
            if emit == "messages":
                transcript_text = "\n".join(s["text"] for s in unit_segs if s["text"])
                if not transcript_text.strip():
                    stats.drop("empty_transcript")
                    continue
                payload = {
                    "id": unit_id,
                    "images": images,
                    "messages": [
                        {"role": "user", "content": "<image>" * len(images) + prompt},
                        {"role": "assistant", "content": transcript_text},
                    ],
                    "meta": out_meta,
                }
            else:
                buckets = _attach_segments(unit_segs, [t for t, _p in unit_frames], half_gap)
                payload = {
                    "id": unit_id,
                    "images": images,
                    "text": _render_text(unit_frames, buckets),
                    "meta": out_meta,
                }
            stats.records_out += 1
            yield payload

        # chunking creates extra output windows beyond the single un-chunked record
        extra["chunks_emitted"] += max(0, len(units) - 1)
        extra["chunks_extra"] += max(0, len(units) - 1)

    chunks_extra = int(extra["chunks_extra"])
    dropped_total = sum(stats.dropped.values())
    if stats.records_in + chunks_extra != stats.records_out + dropped_total:  # not assert: survives python -O
        raise RuntimeError(
            f"video_ingest accounting broken: in={stats.records_in} chunks_extra={chunks_extra} "
            f"out={stats.records_out} dropped={dict(stats.dropped)}"
        )


def preflight(cfg: dict) -> list[str]:
    """Cheap offline checks (no downloads, no GPU); [] means the cfg is runnable here."""
    problems: list[str] = []
    for label, key in (("ffmpeg", "ffmpeg"), ("ffprobe", "ffprobe")):
        name = str(cfg.get(key, label))
        if _find_binary(name) is None:
            problems.append(f"{label} binary not resolvable/executable: {name!r}")
    frames_dir = str(cfg.get("frames_dir", ""))
    if not frames_dir:
        problems.append("frames_dir is required (absolute output directory for extracted frames)")
    elif not Path(frames_dir).is_absolute():
        problems.append(f"frames_dir must be an absolute path, got {frames_dir!r}")
    if str(cfg.get("asr", "whisper")) == "whisper":
        try:
            import transformers  # noqa: F401
        except Exception as exc:  # noqa: BLE001
            problems.append(f"asr='whisper' needs an importable 'transformers' ({type(exc).__name__}: {exc})")
        _resolved, problem = _local_model(
            str(cfg.get("asr_model", "openai/whisper-small")), bool(cfg.get("local_files_only", True))
        )
        if problem is not None:
            problems.append(problem)
    return problems


register_op(FunctionOp("video_ingest", _video_ingest_op, CONFIG_SCHEMA))
