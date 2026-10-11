"""Video frames at exact indices, decoded once and checked against the episode's length.

A training sample needs the frames at the indices its chunk spec names, not whatever a
decoder happens to yield in order. :func:`decode_frames` decodes a video file's stream
once, refuses it unless the decoded frame count equals the episode length
``meta/episodes.jsonl`` declares, and returns the requested frames as uint8 RGB
``HxWx3`` arrays in request order, duplicates included.

Nothing is guessed and nothing is padded. A missing video file, a stream that decodes to
a different number of frames than its episode declares (a video shorter or longer than
its episode is a misaligned dataset: every frame would then be read at the wrong
timestamp) and an index outside ``[0, expected_length)`` are all
:class:`FrameDecodeError` naming the file, the counts and the index. There is no
clamping and no negative-index wrap: an out-of-range index is a caller bug that would
otherwise silently train on the wrong frame. Tail padding is ``vla/chunking.py``'s
``pad="edge"`` decision, carried with an ``action_valid`` mask -- this module never
invents a frame.

Decoding needs an optional decoder. ``av`` (PyAV) is NOT a declared dependency of the
core install -- the same precedent as ``foundationscale/video.py`` -- so it is imported
function-locally and resolved through :data:`DECODERS` and :func:`_decoder`, which tests
replace with a stand-in. When no decoder is importable the refusal names
``pip install av`` rather than silently trying another backend.

Decoded episodes are cached per process under ``(path, mtime_ns, size)``, so a rewritten
file is decoded again and at most :data:`CACHE_MAXSIZE` episodes are held. The cached
arrays are handed out as they are; a caller must not mutate a decoded frame.
"""

from __future__ import annotations

import operator
from collections.abc import Callable, Sequence
from functools import lru_cache
from pathlib import Path
from typing import TYPE_CHECKING, Any, TypeGuard

if TYPE_CHECKING:
    import numpy as np

__all__ = [
    "CACHE_MAXSIZE",
    "DEFAULT_DECODER",
    "DECODERS",
    "FrameDecodeError",
    "decode_frames",
]

DEFAULT_DECODER = "pyav"
CACHE_MAXSIZE = 8


class FrameDecodeError(RuntimeError):
    """A video file is missing, undecodable, or disagrees with the episode it belongs to."""


def decode_frames(
    path: Path,
    indices: Sequence[int],
    *,
    expected_length: int,
) -> list[np.ndarray]:
    """The frames of ``path`` at ``indices``, in request order, as uint8 RGB ``HxWx3``.

    ``expected_length`` is the episode length ``meta/episodes.jsonl`` declares, and the
    decoded stream must carry exactly that many frames: a video shorter or longer than
    its episode is a misaligned dataset, and every frame would then be read at the wrong
    timestamp. ``indices`` may repeat an index and is answered in the order given.

    Raises :class:`FrameDecodeError` when ``path`` is missing, no decoder is importable
    (the message names ``pip install av``), the stream decodes to a frame count other
    than ``expected_length`` (both counts are named), or an index lies outside
    ``[0, expected_length)`` (the file, the index and the range are named). An
    out-of-range index is refused rather than clamped or wrapped: tail padding is the
    chunk spec's ``pad="edge"`` decision, carried with an ``action_valid`` mask, and this
    module never invents a frame.
    """
    file = Path(path)
    if not file.is_file():
        raise FrameDecodeError(
            f"{file}: is missing; expected a video file holding the {expected_length} frames "
            "of its episode"
        )
    if not _is_index(expected_length) or expected_length < 1:
        raise FrameDecodeError(
            f"{file}: expected_length is {expected_length!r}; expected an integer >= 1 "
            "(the episode length from meta)"
        )
    wanted: list[int] = []
    for position, index in enumerate(indices):
        if not _is_index(index):
            raise FrameDecodeError(
                f"{file}: indices[{position}] is {index!r}; expected an integer frame index "
                f"in [0, {expected_length})"
            )
        frame_index = int(index)
        if not 0 <= frame_index < expected_length:
            raise FrameDecodeError(
                f"{file}: index {frame_index} (indices[{position}]) is outside "
                f"[0, {expected_length}); expected_length is {expected_length} "
                "(the episode length from meta)"
            )
        wanted.append(frame_index)

    stat = file.stat()
    try:
        frames = _cached_frames(str(file), stat.st_mtime_ns, stat.st_size)
    except ImportError as exc:
        raise FrameDecodeError(
            f"{file}: no decoder is importable ({exc}); install one with 'pip install av'"
        ) from exc

    decoded = len(frames)
    if decoded != expected_length:
        raise FrameDecodeError(
            f"{file}: decoded {decoded} frames; expected_length is {expected_length} "
            "(the episode length from meta); a video shorter or longer than its episode is a "
            "misaligned dataset, so every frame would be read at the wrong timestamp"
        )
    return [frames[index] for index in wanted]


# -- decoders, the resolver and the per-process cache ------------------------------------


def _decode_error_types(av: Any) -> tuple[type[BaseException], ...]:
    """The classes PyAV raises for a file it cannot open or decode.

    PyAV renamed ``AVError`` to ``FFmpegError``; both are looked up so a decode failure
    is refused as :class:`FrameDecodeError` on either version, alongside the
    ``OSError``/``ValueError``/``RuntimeError`` a malformed file can also raise.
    """
    found: list[type[BaseException]] = [OSError, ValueError, RuntimeError]
    for candidate in (
        getattr(getattr(av, "error", None), "FFmpegError", None),
        getattr(av, "AVError", None),
    ):
        if isinstance(candidate, type) and issubclass(candidate, BaseException):
            found.append(candidate)
    return tuple(found)


def _decode_all_pyav(path: Path) -> list[np.ndarray]:
    """Every frame of ``path``, in stream order, as a uint8 RGB ``HxWx3`` array.

    ``av`` is imported here, function-locally: PyAV is an optional decoder, not a
    dependency of the core install. A file PyAV cannot open, a container with no video
    stream, and a stream it cannot decode to the end are all :class:`FrameDecodeError`
    naming the file -- a half-decoded episode must never reach a sample.
    """
    import av  # type: ignore[import-not-found]  # noqa: PLC0415
    import numpy as np  # noqa: PLC0415

    errors = _decode_error_types(av)
    try:
        container = av.open(str(path))
    except errors as exc:
        raise FrameDecodeError(f"{path}: 'pyav' could not open it ({exc})") from exc

    frames: list[np.ndarray] = []
    with container:
        if len(container.streams.video) == 0:
            raise FrameDecodeError(f"{path}: declares no video stream; expected one to decode")
        try:
            for frame in container.decode(video=0):
                frames.append(
                    np.ascontiguousarray(frame.to_ndarray(format="rgb24"), dtype=np.uint8)
                )
        except errors as exc:
            raise FrameDecodeError(f"{path}: 'pyav' could not decode it ({exc})") from exc
    return frames


DECODERS: dict[str, Callable[[Path], list[np.ndarray]]] = {"pyav": _decode_all_pyav}


def _decoder() -> Callable[[Path], list[np.ndarray]]:
    """The decode-all-frames callable behind :data:`DEFAULT_DECODER`.

    Raises ``ImportError`` when no decoder is registered, so :func:`decode_frames` can
    refuse with the ``pip install av`` hint instead of guessing another backend. A table
    holding exactly one decoder resolves to it whatever its name, so a test stand-in can
    replace :data:`DECODERS` wholesale without knowing the production name.
    """
    decode_all = DECODERS.get(DEFAULT_DECODER)
    if decode_all is not None:
        return decode_all
    if len(DECODERS) == 1:
        return next(iter(DECODERS.values()))
    raise ImportError(
        f"no decoder {DEFAULT_DECODER!r} is registered; DECODERS holds {sorted(DECODERS)}"
    )


@lru_cache(maxsize=CACHE_MAXSIZE)
def _cached_frames(path: str, mtime_ns: int, size: int) -> tuple[np.ndarray, ...]:  # noqa: ARG001
    """Every decoded frame of one video file, cached by ``(path, mtime_ns, size)``.

    ``mtime_ns`` and ``size`` are unused in the body on purpose: they are part of the
    cache key only.

    The file's mtime and size are part of the key, so a rewritten file is decoded again
    instead of serving frames nobody wrote, and at most :data:`CACHE_MAXSIZE` episodes
    are held. A decode that raises is never cached.
    """
    decode_all = _decoder()
    frames = tuple(decode_all(Path(path)))
    # Every caller receives these same arrays, so they are frozen: a caller that
    # edits a frame in place (an augmentation, say) raises instead of silently
    # changing the episode for every later reader of the cache.
    for frame in frames:
        frame.setflags(write=False)
    return frames


def _is_index(value: Any) -> TypeGuard[int]:
    """``value`` is an integer index; a numpy integer counts, a bool or a float does not."""
    if isinstance(value, bool):
        return False
    try:
        operator.index(value)
    except TypeError:
        return False
    return True
