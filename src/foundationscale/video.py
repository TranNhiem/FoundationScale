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
    "conversation_frames_for",
    "fold_video_row",
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


def sample_times(duration: float, frames: int) -> list[float]:
    """Centred-uniform instants (seconds) for ``frames`` samples of a clip."""
    if not duration > 0:
        raise VideoDecodeError(f"clip duration {duration!r} is not positive")
    return [(index + 0.5) * duration / frames for index in range(frames)]


# -- decoder backends: each returns (frames as PIL images, backend duration) ------


def _pil_from_array(array: Any) -> Any:
    from PIL import Image

    return Image.fromarray(array)


def _decode_torchcodec(path: str, frames: int) -> list[Any]:
    from torchcodec.decoders import VideoDecoder  # type: ignore[import-not-found]

    decoder = VideoDecoder(path)
    duration = float(decoder.metadata.duration_seconds or 0.0)
    batch = decoder.get_frames_played_at(seconds=sample_times(duration, frames))
    data = batch.data.permute(0, 2, 3, 1).cpu().numpy()  # N,C,H,W -> N,H,W,C
    return [_pil_from_array(frame) for frame in data]


def _decode_av(path: str, frames: int) -> list[Any]:
    import av  # type: ignore[import-not-found]

    with av.open(path) as container:
        stream = container.streams.video[0]
        if stream.duration is not None and stream.time_base is not None:
            duration = float(stream.duration * stream.time_base)
        else:
            duration = float(container.duration or 0) / 1_000_000
        images = []
        for instant in sample_times(duration, frames):
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


def _decode_cv2(path: str, frames: int) -> list[Any]:
    import cv2  # type: ignore[import-not-found]

    capture = cv2.VideoCapture(path)
    try:
        if not capture.isOpened():
            raise VideoDecodeError(f"{path}: cv2 could not open the clip")
        fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
        count = float(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0.0)
        duration = count / fps if fps > 0 else 0.0
        images = []
        for instant in sample_times(duration, frames):
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


def _decode_ffmpeg(path: str, frames: int) -> list[Any]:
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
    for instant in sample_times(duration, frames):
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


_DECODERS: dict[str, Callable[[str, int], list[Any]]] = {
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


def video_frame_paths(
    path: str | os.PathLike[str],
    budget: FrameBudget,
    cache_dir: str | os.PathLike[str],
    *,
    backends: Sequence[str] | None = None,
) -> list[str]:
    """The ``budget.frames`` frame files for ``path``, decoding once if not cached.

    Raises :class:`VideoDecodeError` naming every backend tried when none can
    produce the frames, or ``FileNotFoundError`` for a missing clip -- the caller
    turns either into its refusal; nothing here substitutes a blank frame.
    """
    clip = Path(path)
    if not clip.is_file():
        raise FileNotFoundError(f"video {clip} does not exist")
    target = Path(cache_dir) / _clip_id(clip) / budget.key
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
            images = _DECODERS[name](str(clip), budget.frames)
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


def _clip_path(clip: str, base_dir: str | os.PathLike[str]) -> Path:
    """``clip`` as a path; a relative one resolves against ``base_dir``."""
    path = Path(clip)
    return path if path.is_absolute() else Path(base_dir) / path


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
    """
    out = dict(row)
    frames: list[str] = []
    for clip in _as_list(row.get(video_column)):
        frames.extend(video_frame_paths(_clip_path(clip, base_dir), budget, cache_dir))
    out[image_column] = _as_list(row.get(image_column)) + frames
    if frames and isinstance(out.get("text"), str):
        out["text"] = out["text"].replace(_VIDEO_MARKER, "").strip()
    return out


def conversation_frames_for(
    budget: FrameBudget,
    cache_dir: str | os.PathLike[str],
    *,
    base_dir: str | os.PathLike[str],
    video_column: str,
    backends: Sequence[str] | None = None,
) -> Callable[[Mapping[str, Any]], dict[str, Any]]:
    """The ``frames_for`` callable the conversation arm needs, from a declared budget.

    The image arm folds clips into frame FILES on the image column
    (:func:`fold_video_row`); the conversation arm instead hands the processor a
    native ``videos=`` input, so it needs the frames themselves. Both go through
    the same :func:`video_frame_paths` cache and the same relative-path rule, so
    one clip yields the same instants on either arm.

    The callable returns ``{"frames": [...], "frames_indices": [...]}`` -- RGB
    images, exactly ``budget.frames`` of them. No ``fps`` or ``duration`` is
    reported: the cache records normalised instants, and inventing either would be
    a guess. It raises rather than substituting: ``FileNotFoundError`` for a
    missing clip, :class:`VideoDecodeError` when no decoder can read it, and
    ``ValueError`` for a row carrying more than one clip (the conversation arm
    supports one ``<video>`` per row). The caller turns each into its refusal.
    """

    def frames_for(row: Mapping[str, Any]) -> dict[str, Any]:
        from PIL import Image  # noqa: PLC0415

        clips = _as_list(row.get(video_column))
        if len(clips) != 1:
            raise ValueError(
                f"expected exactly one clip in column {video_column!r}, found {len(clips)}"
            )
        files = video_frame_paths(
            _clip_path(clips[0], base_dir), budget, cache_dir, backends=backends
        )
        frames = []
        for name in files:
            with Image.open(name) as image:
                frames.append(image.convert("RGB"))
        return {"frames": frames, "frames_indices": list(range(len(frames)))}

    return frames_for
