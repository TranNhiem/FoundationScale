"""Video as a DECLARED frame budget: clip -> N cached frame images.

Video was refused across the training plane because nothing here sampled frames,
and the missing pieces are dataset decisions, not plumbing: how many frames, how
they are spaced, at what resolution. A default chosen in code would silently
redefine the corpus. This module makes those decisions an explicit, recorded
:class:`FrameBudget` and turns each clip into that many frames on disk, so the
already-measured image path (processor content blocks, ``pixel_values``) carries
video unchanged -- a clip becomes ``frames`` images, in time order.

Sampling is defined over TIME, not frame index, so every decoder backend draws the
same instants: frame ``i`` of ``n`` is the frame showing at
``(i + 0.5) * duration / n`` seconds (centred uniform). A clip shorter than the
budget in distinct frames repeats a frame rather than silently shrinking the
token count the budget declared.

Decoding tries, in order, ``torchcodec``, ``av`` (PyAV), ``cv2`` and the
``ffmpeg``/``ffprobe`` CLIs, and uses the first one present -- the cluster images
differ in which they ship. ``FOUNDATIONSCALE_VIDEO_BACKEND`` (comma-separated)
pins the order. Frames are cached as JPEG under
``<cache>/<clip-hash>/<budget-key>/`` with a ``frames.json`` record, so a clip is
decoded once per budget no matter how many epochs or ranks read it.

Imports are lazy: the module loads without torch, PIL or any decoder.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import tempfile
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

__all__ = [
    "BACKENDS",
    "SAMPLINGS",
    "FrameBudget",
    "VIDEO_CACHE_ENV",
    "VIDEO_FRAMES_COLUMN",
    "VIDEO_FRAMES_ENV",
    "VIDEO_MAX_SIDE_ENV",
    "VIDEO_SAMPLING_ENV",
    "VideoDecodeError",
    "available_backends",
    "budget_from_env",
    "fold_video_row",
    "frames_for_row",
    "sample_times",
    "video_frame_paths",
]

SAMPLINGS: tuple[str, ...] = ("uniform",)
BACKENDS: tuple[str, ...] = ("torchcodec", "av", "cv2", "ffmpeg")
_RECORD = "frames.json"

# The SFT plane's declaration surface: environment variables with no default, the
# same rule as the image and video column names. FRAMES is the opt-in; the other
# three only refine a budget that FRAMES declared.
VIDEO_FRAMES_ENV = "FOUNDATIONSCALE_TRAIN_VIDEO_FRAMES"
VIDEO_SAMPLING_ENV = "FOUNDATIONSCALE_TRAIN_VIDEO_SAMPLING"
VIDEO_MAX_SIDE_ENV = "FOUNDATIONSCALE_TRAIN_VIDEO_MAX_SIDE"
VIDEO_CACHE_ENV = "FOUNDATIONSCALE_TRAIN_VIDEO_CACHE_DIR"
# The image column a video-only corpus's frames are folded into.
VIDEO_FRAMES_COLUMN = "__fs_video_frames"
_VIDEO_MARKER = "<video>"


class VideoDecodeError(RuntimeError):
    """A clip could not be turned into its declared frames."""


@dataclass(frozen=True, slots=True)
class FrameBudget:
    """How a clip becomes images: ``frames`` instants, spaced by ``sampling``.

    ``max_side`` caps the longer image side (aspect kept); ``None`` keeps the
    decoded resolution, which the processor then resizes by its own rule.
    """

    frames: int
    sampling: str = "uniform"
    max_side: int | None = None

    def __post_init__(self) -> None:
        if self.frames < 1:
            raise ValueError(f"FrameBudget.frames must be >= 1, got {self.frames}")
        if self.sampling not in SAMPLINGS:
            raise ValueError(f"FrameBudget.sampling {self.sampling!r} is not one of {SAMPLINGS}")
        if self.max_side is not None and self.max_side < 16:
            raise ValueError(f"FrameBudget.max_side must be >= 16 or None, got {self.max_side}")

    @property
    def key(self) -> str:
        """The cache directory name: one budget, one set of frames."""
        return f"f{self.frames}-{self.sampling}-s{self.max_side or 0}"


def _check_segment_bounds(start: float | None, end: float | None) -> None:
    """Both ``start``/``end`` given, or neither -- never just one.

    Shared by :func:`sample_times` (which also knows the duration, so it can
    additionally refuse ``start`` at or past it) and :func:`video_frame_paths`
    (which calls this FIRST, before touching the cache or any backend, so an
    obviously bad declaration costs nothing).
    """
    if (start is None) != (end is None):
        raise VideoDecodeError(
            f"video segment needs both start and end seconds, or neither "
            f"(got start={start!r}, end={end!r})"
        )
    if start is not None and end is not None and end <= start:
        raise VideoDecodeError(f"segment end {end}s must be greater than start {start}s")


def sample_times(
    duration: float, frames: int, *, start: float | None = None, end: float | None = None
) -> list[float]:
    """Centred-uniform instants (seconds) for ``frames`` samples of a clip.

    With no ``start``/``end`` this is unchanged: ``frames`` instants spanning
    the whole ``[0, duration]`` clip. With both given, the SAME centred-uniform
    rule applies inside ``[start, end]`` instead -- the segment a row's
    conversation describes within a longer clip -- clamped to the decoded
    ``duration`` so a segment that slightly overruns a clip's measured length
    (a common corpus rounding artifact) still samples rather than refuses.
    Refuses (:class:`VideoDecodeError`) when only one of ``start``/``end`` is
    given, when ``end <= start``, or when ``start`` is at or past ``duration``
    -- a segment this module cannot locate in the clip at all, as opposed to
    one it can merely clamp.
    """
    if not duration > 0:
        raise VideoDecodeError(f"clip duration {duration!r} is not positive")
    _check_segment_bounds(start, end)
    if start is None:
        return [(index + 0.5) * duration / frames for index in range(frames)]
    assert end is not None  # _check_segment_bounds already paired them
    if start >= duration:
        raise VideoDecodeError(
            f"segment start {start}s is at or beyond the clip duration {duration}s"
        )
    clamped_start = max(0.0, start)
    clamped_end = min(duration, end)
    span = clamped_end - clamped_start
    return [clamped_start + (index + 0.5) * span / frames for index in range(frames)]


# -- decoder backends: each returns (frames as PIL images, backend duration) ------


def _pil_from_array(array: Any) -> Any:
    from PIL import Image

    return Image.fromarray(array)


def _decode_torchcodec(
    path: str, frames: int, *, start: float | None = None, end: float | None = None
) -> list[Any]:
    from torchcodec.decoders import VideoDecoder  # type: ignore[import-not-found]

    decoder = VideoDecoder(path)
    duration = float(decoder.metadata.duration_seconds or 0.0)
    instants = sample_times(duration, frames, start=start, end=end)
    batch = decoder.get_frames_played_at(seconds=instants)
    data = batch.data.permute(0, 2, 3, 1).cpu().numpy()  # N,C,H,W -> N,H,W,C
    return [_pil_from_array(frame) for frame in data]


def _decode_av(
    path: str, frames: int, *, start: float | None = None, end: float | None = None
) -> list[Any]:
    import av  # type: ignore[import-not-found]

    with av.open(path) as container:
        stream = container.streams.video[0]
        if stream.duration is not None and stream.time_base is not None:
            duration = float(stream.duration * stream.time_base)
        else:
            duration = float(container.duration or 0) / 1_000_000
        images = []
        for instant in sample_times(duration, frames, start=start, end=end):
            container.seek(int(instant / stream.time_base), stream=stream, backward=True)
            chosen = None
            for frame in container.decode(stream):
                chosen = frame
                if frame.time is not None and frame.time >= instant:
                    break
            if chosen is None:
                raise VideoDecodeError(f"{path}: no frame decoded at {instant:.3f}s")
            images.append(chosen.to_image())
    return images


def _decode_cv2(
    path: str, frames: int, *, start: float | None = None, end: float | None = None
) -> list[Any]:
    import cv2  # type: ignore[import-not-found]

    capture = cv2.VideoCapture(path)
    try:
        if not capture.isOpened():
            raise VideoDecodeError(f"{path}: cv2 could not open the clip")
        fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
        count = float(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0.0)
        duration = count / fps if fps > 0 else 0.0
        images = []
        for instant in sample_times(duration, frames, start=start, end=end):
            capture.set(cv2.CAP_PROP_POS_MSEC, instant * 1000.0)
            ok, bgr = capture.read()
            if not ok:
                raise VideoDecodeError(f"{path}: cv2 read failed at {instant:.3f}s")
            images.append(_pil_from_array(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)))
        return images
    finally:
        capture.release()


def _ffmpeg_bins() -> tuple[str, str] | None:
    ffmpeg = os.environ.get("FOUNDATIONSCALE_FFMPEG") or shutil.which("ffmpeg")
    ffprobe = os.environ.get("FOUNDATIONSCALE_FFPROBE") or shutil.which("ffprobe")
    return (ffmpeg, ffprobe) if ffmpeg and ffprobe else None


def _decode_ffmpeg(
    path: str, frames: int, *, start: float | None = None, end: float | None = None
) -> list[Any]:
    import io

    from PIL import Image

    bins = _ffmpeg_bins()
    if bins is None:
        raise VideoDecodeError("ffmpeg/ffprobe not found")
    ffmpeg, ffprobe = bins
    probe = subprocess.run(
        [ffprobe, "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", path],
        capture_output=True,
        text=True,
        check=False,
    )
    try:
        duration = float(probe.stdout.strip())
    except ValueError as exc:
        detail = probe.stderr.strip()
        raise VideoDecodeError(f"{path}: ffprobe gave no duration ({detail})") from exc
    images = []
    for instant in sample_times(duration, frames, start=start, end=end):
        grab = subprocess.run(
            [ffmpeg, "-v", "error", "-ss", f"{instant:.6f}", "-i", path, "-frames:v", "1"]
            + ["-f", "image2pipe", "-vcodec", "png", "-"],
            capture_output=True,
            check=False,
        )
        if grab.returncode != 0 or not grab.stdout:
            raise VideoDecodeError(f"{path}: ffmpeg grab failed at {instant:.3f}s")
        image = Image.open(io.BytesIO(grab.stdout))
        image.load()
        images.append(image.convert("RGB"))
    return images


_DECODERS: dict[str, Callable[..., list[Any]]] = {
    "torchcodec": _decode_torchcodec,
    "av": _decode_av,
    "cv2": _decode_cv2,
    "ffmpeg": _decode_ffmpeg,
}


def _importable(module: str) -> bool:
    import importlib.util

    return importlib.util.find_spec(module) is not None


def available_backends() -> list[str]:
    """The decoders present here, in the order they will be tried."""
    pinned = os.environ.get("FOUNDATIONSCALE_VIDEO_BACKEND", "")
    order = [name.strip() for name in pinned.split(",") if name.strip()] or list(BACKENDS)
    unknown = [name for name in order if name not in _DECODERS]
    if unknown:
        raise ValueError(f"FOUNDATIONSCALE_VIDEO_BACKEND names unknown backend(s) {unknown}")
    present = []
    for name in order:
        if name == "ffmpeg":
            if _ffmpeg_bins() is not None:
                present.append(name)
        elif _importable(name):
            present.append(name)
    return present


# -- the cached clip -> frame-paths step --------------------------------------------


def _clip_id(path: Path) -> str:
    stat = path.stat()
    raw = f"{path.resolve()}|{stat.st_size}|{stat.st_mtime_ns}"
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:20]


def _resize(image: Any, max_side: int | None) -> Any:
    image = image.convert("RGB")
    if max_side is not None and max(image.size) > max_side:
        image.thumbnail((max_side, max_side))
    return image


def _cache_key(budget: FrameBudget, start: float | None, end: float | None) -> str:
    """``budget.key``, or one further split per segment so two segments of the
    SAME clip (and the same budget) never share a cache directory."""
    if start is None:
        return budget.key
    assert end is not None  # callers already ran _check_segment_bounds
    return f"{budget.key}-seg{start:.3f}-{end:.3f}"


def video_frame_paths(
    path: str | os.PathLike[str],
    budget: FrameBudget,
    cache_dir: str | os.PathLike[str],
    *,
    backends: Sequence[str] | None = None,
    start: float | None = None,
    end: float | None = None,
) -> list[str]:
    """The ``budget.frames`` frame files for ``path``, decoding once if not cached.

    With no ``start``/``end`` this is UNCHANGED from before segments existed:
    same cache path, same decoder call shape (``decoder(path, frames)``, two
    positional arguments, nothing more) -- callers that never mention a
    segment see byte-identical behaviour. With both given, the frames are
    centred-uniform instants inside ``[start, end]`` (see
    :func:`sample_times`) and the cache key folds in the segment (see
    :func:`_cache_key`), so a second segment of the same file decodes its own
    frames rather than silently reusing the first segment's.

    Raises :class:`VideoDecodeError` naming every backend tried when none can
    produce the frames (or, before any backend is touched, for a segment
    declaration that cannot be one -- see :func:`_check_segment_bounds`), or
    ``FileNotFoundError`` for a missing clip -- the caller turns either into
    its refusal; nothing here substitutes a blank frame.
    """
    _check_segment_bounds(start, end)
    clip = Path(path)
    if not clip.is_file():
        raise FileNotFoundError(f"video {clip} does not exist")
    target = Path(cache_dir) / _clip_id(clip) / _cache_key(budget, start, end)
    record = target / _RECORD
    if record.is_file():
        cached = json.loads(record.read_text(encoding="utf-8"))
        files = [str(target / name) for name in cached["files"]]
        if len(files) == budget.frames and all(Path(f).is_file() for f in files):
            return files
    if target.exists():
        # A record that no longer matches its files is stale; decode afresh.
        shutil.rmtree(target, ignore_errors=True)
    order = list(backends) if backends is not None else available_backends()
    if not order:
        raise VideoDecodeError(
            f"{clip}: no video decoder available (tried {list(BACKENDS)}); install "
            "torchcodec, av or opencv, or put ffmpeg/ffprobe on PATH"
        )
    failures = []
    images: list[Any] | None = None
    used = ""
    for name in order:
        try:
            if start is None:
                images = _DECODERS[name](str(clip), budget.frames)
            else:
                images = _DECODERS[name](str(clip), budget.frames, start=start, end=end)
            used = name
            break
        except Exception as exc:  # noqa: BLE001 -- try the next backend, report all
            failures.append(f"{name}: {exc}")
    if images is None or len(images) != budget.frames:
        got = "none" if images is None else str(len(images))
        raise VideoDecodeError(
            f"{clip}: could not decode {budget.frames} frames (got {got}); "
            f"tried {failures or order}"
        )
    target.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=".frames-", dir=target.parent))
    names = [f"frame_{index:04d}.jpg" for index in range(budget.frames)]
    for name, image in zip(names, images, strict=True):
        _resize(image, budget.max_side).save(staging / name, quality=95)
    (staging / _RECORD).write_text(
        json.dumps(
            {
                "source": str(clip.resolve()),
                "budget": budget.key,
                "backend": used,
                "times": [round(t, 6) for t in _times_or_empty(images, budget)],
                "files": names,
                "start": start,
                "end": end,
            }
        ),
        encoding="utf-8",
    )
    try:
        staging.replace(target)
    except OSError:
        # Another rank published the same clip and budget first; keep theirs.
        shutil.rmtree(staging, ignore_errors=True)
        if not (target / _RECORD).is_file():
            raise
    return [str(target / name) for name in names]


def _times_or_empty(images: list[Any], budget: FrameBudget) -> list[float]:
    """The record keeps relative instants (duration-normalised), backend-independent."""
    return [(index + 0.5) / budget.frames for index in range(len(images))]


# -- the SFT plane's declaration and row fold ----------------------------------------


def budget_from_env(environ: Mapping[str, str]) -> FrameBudget | None:
    """The declared :class:`FrameBudget`, or ``None`` when no frame count is declared.

    Raises ``ValueError`` for a declaration that cannot be a budget (a non-integer,
    a count below 1, an unknown sampling) -- the caller refuses on it rather than
    training under a budget nobody asked for.
    """
    raw = (environ.get(VIDEO_FRAMES_ENV) or "").strip()
    if not raw:
        return None
    side = (environ.get(VIDEO_MAX_SIDE_ENV) or "").strip()
    try:
        frames = int(raw)
        max_side = int(side) if side else None
    except ValueError as exc:
        raise ValueError(
            f"{VIDEO_FRAMES_ENV}={raw!r} / {VIDEO_MAX_SIDE_ENV}={side!r} must be integers"
        ) from exc
    sampling = (environ.get(VIDEO_SAMPLING_ENV) or "uniform").strip()
    return FrameBudget(frames=frames, sampling=sampling, max_side=max_side)


def _as_list(value: Any) -> list[str]:
    if value is None or value == "":
        return []
    if isinstance(value, (str, os.PathLike)):
        return [str(value)]
    return [str(item) for item in value if item]


def _row_segment(row: Mapping[str, Any]) -> tuple[float | None, float | None]:
    """The optional ``start``/``end`` seconds a row declares, or ``(None, None)``.

    Absent on both sides is a valid, common row (the whole clip); absent on
    only one side is a malformed row, and :func:`video_frame_paths` is the one
    that refuses it (via :func:`_check_segment_bounds`) -- this helper only
    reads what is there, it does not validate it.
    """
    start = row.get("start")
    end = row.get("end")
    return (None if start is None else float(start)), (None if end is None else float(end))


def fold_video_row(
    row: Mapping[str, Any],
    *,
    video_column: str,
    image_column: str,
    budget: FrameBudget,
    cache_dir: str | os.PathLike[str],
    base_dir: str | os.PathLike[str],
) -> dict[str, Any]:
    """One corpus row with its clip(s) turned into frame images on ``image_column``.

    Frames are APPENDED after any images the row already carries, clip by clip in
    order; relative clip paths resolve against ``base_dir`` (the dataset's own
    directory). ``<video>`` markers are removed from the text, because the frames
    now enter as image content blocks and a marker left behind would be a
    placeholder nothing resolves. A row with no clip passes through with its
    images unchanged, so a mixed corpus keeps its text-only and image rows.

    An optional ``start``/``end`` (seconds) on the row scopes EVERY clip it
    lists to that one segment, rather than the whole file -- the shape a row
    describing a short action inside a longer clip takes. A row without
    either key is unaffected (the whole clip, exactly as before segments
    existed).
    """
    out = dict(row)
    start, end = _row_segment(row)
    frames: list[str] = []
    for clip in _as_list(row.get(video_column)):
        path = Path(clip)
        if not path.is_absolute():
            path = Path(base_dir) / path
        frames.extend(video_frame_paths(path, budget, cache_dir, start=start, end=end))
    out[image_column] = _as_list(row.get(image_column)) + frames
    if frames and isinstance(out.get("text"), str):
        out["text"] = out["text"].replace(_VIDEO_MARKER, "").strip()
    return out


def frames_for_row(
    row: Mapping[str, Any],
    *,
    video_column: str,
    budget: FrameBudget,
    cache_dir: str | os.PathLike[str],
    base_dir: str | os.PathLike[str],
) -> dict[str, Any]:
    """One row's clip -> NATIVE video frames + metadata, for the conversation path.

    Unlike :func:`fold_video_row` (which turns a clip into ``budget.frames``
    separate IMAGES on the image arm), this keeps the clip as ONE video: the
    returned ``{"frames": [...], "fps": ..., "frames_indices": [...],
    "duration": ...}`` is exactly the shape
    ``train/conversation.py``'s ``frames_for`` hook documents, passed to the
    processor as native ``videos=``/``video_metadata=`` input
    (``do_sample_frames=False``) rather than re-expanded as N image blocks.

    REQUIRES both ``start`` and ``end`` (seconds) on ``row`` -- the segment the
    row's conversation describes inside a longer clip, the shape
    ``prepare_data.py``'s video rows carry. There is no whole-clip arm here: a
    video row with neither key is a declaration gap, not an invitation to
    assume "the whole file" -- :func:`fold_video_row` (the plain image arm)
    remains the whole-clip path. Raises :class:`VideoDecodeError` for a row
    missing the segment, for the same reason :func:`video_frame_paths` refuses
    a declaration it cannot locate rather than guessing one.

    ``fps`` and ``frames_indices`` describe the SAMPLED clip alone --
    ``budget.frames`` frames at ``budget.frames / (end - start)`` fps, indices
    ``0..budget.frames - 1`` -- not the source video's native frame rate: the
    two processor families this plane trains read these fields only to place
    frames in TIME relative to EACH OTHER (the per-frame timestamps embedded
    in the rendered prompt), which this self-consistent framing gives them
    without this module probing a native fps it has no other use for.
    """
    clip_value = row.get(video_column)
    path = Path(_as_list(clip_value)[0])
    if not path.is_absolute():
        path = Path(base_dir) / path
    start, end = _row_segment(row)
    if start is None or end is None:
        raise VideoDecodeError(
            f"video {clip_value!r} (column {video_column!r}) has no 'start'/'end' segment "
            f"declared on its row (got start={start!r}, end={end!r}); the conversation "
            "path's native-video frames require the segment the row describes -- "
            "fold_video_row (the image arm) is the whole-clip path, not this one"
        )
    paths = video_frame_paths(path, budget, cache_dir, start=start, end=end)
    from PIL import Image  # noqa: PLC0415

    images = []
    for file_path in paths:
        image = Image.open(file_path)
        image.load()  # decode NOW: a lazy handle failing mid-batch is quieter
        images.append(image)
    span = end - start
    return {
        "frames": images,
        "fps": budget.frames / span,
        "frames_indices": list(range(budget.frames)),
        "duration": span,
    }
