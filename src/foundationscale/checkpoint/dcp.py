"""Partial, read-only weight access for DCP and safetensors checkpoints.

Why this module exists
----------------------
The audit's Incident #1 was a MoE checkpoint saved under *local* expert names:
128 experts collapsed to 16 distinct tensors aliased 8 ways, and it passed
``rc=0``, resume, loss, tensor counts and dtype checks for two full training
runs. The tool built to catch it then shipped Incident #6: it answered
``all_identity: true`` because the expert tensors were absent, the set was
empty, and ``all([])`` is ``True``. Both failures share one shape: **a check
that never touched the data reported success.**

The fix deployed there was forensically verified on the audited estate against
a real 51.7 GB sharded checkpoint, and it is cheap: a DCP checkpoint's
``.metadata`` file is a plain pickle, and every stored chunk range is a
complete, self-describing ``torch.save`` ZIP archive. Slicing bytes
``[offset, offset+length)`` out of a ``.distcp`` shard and passing them to
``torch.load(io.BytesIO(...), weights_only=True)`` is exactly what torch's own
loader does (``_create_file_view`` + ``torch.load``) — and it *self-validates*:
an off-by-one offset raises ``UnpicklingError``, a truncated slice raises
``PytorchStreamReader ... failed finding central directory``. It cannot return
garbage. That makes per-chunk reads safe enough to run as a gate at
``FIRST_SAVE``/``SAVE``/``EXPORT`` without a process group, CUDA, or a rank.

Two defects in the reference probe are fixed here, not inherited:

1. ``read_full()`` returned a tensor that was silently zero wherever no chunk
   covered it — the vacuous-truth bug wearing a different hat. Here coverage
   is a *returned fact* (:class:`ReadResult`), and :meth:`DcpReader.read_full`
   raises :class:`IncompleteCoverageError` unless covered elements exactly
   equal the declared size.
2. The probe's ``cosine``/``rms`` accumulated in float32 and underflowed on
   large tensors — a tensor's cosine *with itself* printed 1.80. All
   reductions here accumulate in float64, and :func:`compare_keys` asserts
   that bitwise-identical inputs score cosine 1.0, so a broken accumulator
   fails loudly instead of laundering a bad number into a report.

Memory behaviour, stated with its scope (measured, not aspirational): parsing
``.metadata`` alone costs ~180 MB RSS. Reading 128 expert chunks (~4 MB each)
added 18 MB and 1.9 s (peak 198 MB). Those figures cover *chunk-level* reads.
Materializing one ``262144 x 2816`` bf16 embedding and calling ``.float()``
drove peak RSS to 11.5 GB. Memory is therefore a function of what you
request, and this module provides :func:`compare_keys`, which streams row
blocks and never materializes a whole large tensor in float32.
"""

from __future__ import annotations

import io
import json
import math
import os
from collections import OrderedDict
from collections.abc import Callable, Hashable, Sequence
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

__all__ = [
    "CheckpointError",
    "CheckpointFormatError",
    "TensorNotFoundError",
    "ChunkReadError",
    "IncompleteCoverageError",
    "Chunk",
    "ReadResult",
    "TensorComparison",
    "WeightSource",
    "DcpReader",
    "SafetensorsReader",
    "open_weights",
    "compare_keys",
    "DCP_METADATA_FILENAME",
    "SAFETENSORS_INDEX_FILENAME",
    "VERDICT_EXACT",
    "VERDICT_CLOSE",
    "VERDICT_DIFFER",
    "VERDICT_SHAPE_MISMATCH",
    "VERDICT_NON_FINITE",
    "VERDICT_NO_ELEMENTS",
    "VERDICT_DTYPE_MISMATCH",
]

DCP_METADATA_FILENAME = ".metadata"
SAFETENSORS_INDEX_FILENAME = "model.safetensors.index.json"

# The machine-consumed vocabulary of TensorComparison.verdict. Adding a value
# is a breaking change for downstream gates, which is exactly why NON_FINITE
# is its own verdict instead of an overloaded DIFFER: a consumer written
# before this fix must see a verdict it does not recognize (and fail closed
# on it), never silently treat a poisoned artifact as ordinary divergence.
# NO_ELEMENTS exists on the same rule. A shape that matches but declares zero
# elements could not reuse DIFFER (no divergence was observed) or
# SHAPE_MISMATCH (the geometry agrees), and EXACT/CLOSE are pass grades that
# must never be earned by examining nothing -- that is the founding all([])
# incident surfacing inside the comparator itself.
# DTYPE_MISMATCH is the third verdict on this rule, and it is the abstention's
# mirror: unlike NO_ELEMENTS the difference WAS observed -- in the declared
# encodings, before any read -- so abstaining would un-state a fact the
# function is holding in its hands. The geometry agrees, so it cannot borrow
# SHAPE_MISMATCH; nothing is streamed, so DIFFER's definition ("finite
# content outside both tolerances") has no basis; and EXACT across differing
# encodings is a requantization laundered into an identity claim, which is
# the defect class the export gates exist to catch. A pre-fix consumer meets
# a token it cannot map to any grade and fails closed on it.
VERDICT_EXACT = "EXACT"
VERDICT_CLOSE = "CLOSE"
VERDICT_DIFFER = "DIFFER"
VERDICT_SHAPE_MISMATCH = "SHAPE_MISMATCH"
VERDICT_NON_FINITE = "NON_FINITE"
VERDICT_NO_ELEMENTS = "NO_ELEMENTS"
VERDICT_DTYPE_MISMATCH = "DTYPE_MISMATCH"


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


class CheckpointError(Exception):
    """Base class for everything this module raises on checkpoint defects.

    Every error carries the ``path`` of the source and, where meaningful, the
    ``key`` of the tensor, because "some tensor failed somewhere" is the kind of
    message that sends an operator to re-run a 51 GB comparison by hand.
    """

    def __init__(self, message: str, *, path: str, key: str | None = None) -> None:
        self.path = os.fspath(path)
        self.key = key
        where = f"path={self.path!r}"
        if key is not None:
            where += f", key={key!r}"
        super().__init__(f"{message} ({where})")


class CheckpointFormatError(CheckpointError):
    """The artifact is malformed, ambiguous, empty, or of an unsupported format.

    "Empty" is a format error here on purpose: a source with zero tensor keys
    makes every downstream ``all(...)`` check vacuously true, which is the exact
    bug class this module exists to prevent.
    """


class TensorNotFoundError(CheckpointError):
    """The requested tensor key is not present in the source."""

    def __init__(self, message: str, *, path: str, key: str) -> None:
        super().__init__(message, path=path, key=key)


class ChunkReadError(CheckpointError):
    """A stored chunk could not be read or failed self-validation.

    Carries the chunk ``offsets`` and the original exception. A well-formed DCP
    range self-validates (see module docstring), so reaching this error means
    the artifact — or the offsets used to address it — is corrupt.
    """

    def __init__(
        self,
        message: str,
        *,
        path: str,
        key: str,
        offsets: Sequence[int] | None = None,
        original: BaseException | None = None,
    ) -> None:
        self.offsets = None if offsets is None else tuple(int(o) for o in offsets)
        self.original = original
        detail = message
        if self.offsets is not None:
            detail += f" [offsets={self.offsets}]"
        if original is not None:
            detail += f" [cause: {type(original).__name__}: {original}]"
        super().__init__(detail, path=path, key=key)


class IncompleteCoverageError(CheckpointError):
    """Stored chunks did not cover the declared extent of a read.

    Without this check the uncovered region reads back as zeros that look like
    data — the silent-zero variant of the vacuous-truth bug.
    """

    def __init__(
        self,
        message: str,
        *,
        path: str,
        key: str,
        elements_covered: int,
        elements_expected: int,
    ) -> None:
        self.elements_covered = elements_covered
        self.elements_expected = elements_expected
        super().__init__(
            f"{message} [covered={elements_covered}, expected={elements_expected}]",
            path=path,
            key=key,
        )


# ---------------------------------------------------------------------------
# Shared value types
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Chunk:
    """One stored shard fragment of a logical tensor."""

    offsets: tuple[int, ...]
    """Origin of this chunk within the global tensor, one integer per dim."""

    sizes: tuple[int, ...]
    """Extent of this chunk along each dim. ``offsets + sizes`` is its far corner."""


@dataclass(frozen=True)
class ReadResult:
    """A tensor read plus its coverage, as facts. Coverage is returned, never inferred.

    A caller that ignores the coverage fields and treats ``tensor`` as ground
    truth is re-implementing the silent-zero bug; gates should assert
    :attr:`complete` or, better, rely on the reader having already raised
    :class:`IncompleteCoverageError`.
    """

    key: str
    tensor: Any
    """The assembled ``torch.Tensor``. Dtype is the on-disk dtype; no upcasting."""

    chunks_read: int
    elements_covered: int
    elements_expected: int
    bytes_read: int

    @property
    def complete(self) -> bool:
        """True iff stored chunks covered every declared element exactly once."""
        return self.elements_covered == self.elements_expected


@dataclass(frozen=True)
class TensorComparison:
    """Streaming comparison of one key across two :class:`WeightSource` objects.

    All norms and the dot product were accumulated in float64 over row blocks,
    so ``cosine`` is meaningful on tensors far too large to upcast at once.
    ``verdict`` is one of ``EXACT``, ``CLOSE``, ``DIFFER``, ``NON_FINITE``,
    ``SHAPE_MISMATCH``, ``NO_ELEMENTS``, ``DTYPE_MISMATCH``. ``NO_ELEMENTS``
    is a stated abstention, not a grade: both sources agreed on a shape that
    declares zero elements, so nothing was read and neither identity nor
    divergence can be claimed. ``DTYPE_MISMATCH`` is the mirror of that
    abstention: a divergence (in the declared encodings) that WAS observed --
    at metadata level, before any streaming -- so it is a finding, never an
    abstention, and no numeric field on its record claims a measurement.
    ``bitwise_equal`` is a claim about the stored encodings, not merely the
    decoded values: across differing dtypes no bitwise-identity claim is
    possible, so the field is False no matter how closely the values agree.
    ``nonfinite_elements`` counts NaN/Inf on every
    verdict, including ``EXACT`` — bitwise identity is a statement about
    bytes, not about the finiteness of the weights those bytes encode.
    """

    key: str
    elements: int
    shape_a: tuple[int, ...]
    shape_b: tuple[int, ...]
    dtype_a: str
    dtype_b: str
    bitwise_equal: bool
    mismatched_elements: int
    max_abs_diff: float
    mean_abs_diff: float
    cosine: float | None
    """None iff no finite, non-zero direction exists to compare: an all-zero
    tensor on either side (0/0 is a statement about nothing) or NaN/Inf in
    the content (an unbounded or undefined norm has no angle)."""

    rms_a: float
    rms_b: float
    chunks_read: int
    bytes_read: int
    verdict: str
    nonfinite_elements: int = 0
    """NaN/Inf elements seen across both sources, summed over blocks. Surfaced
    even on ``EXACT``: identical ±inf bytes are parity, but the operator must
    see the poison count rather than have it laundered into a clean verdict."""

    def to_dict(self) -> dict[str, Any]:
        """Projection that survives ``json.dumps(..., allow_nan=False)``.

        ``math.inf`` and ``nan`` are honest answers here ("unbounded",
        "undefined"), but strict JSON consumers rightly reject the bare
        tokens Python's default encoder would emit for them, so every float
        field passes through :func:`_json_float`.
        """
        return {
            "key": self.key,
            "elements": self.elements,
            "shape_a": self.shape_a,
            "shape_b": self.shape_b,
            "dtype_a": self.dtype_a,
            "dtype_b": self.dtype_b,
            "bitwise_equal": self.bitwise_equal,
            "mismatched_elements": self.mismatched_elements,
            "max_abs_diff": _json_float(self.max_abs_diff),
            "mean_abs_diff": _json_float(self.mean_abs_diff),
            "cosine": None if self.cosine is None else _json_float(self.cosine),
            "rms_a": _json_float(self.rms_a),
            "rms_b": _json_float(self.rms_b),
            "chunks_read": self.chunks_read,
            "bytes_read": self.bytes_read,
            "verdict": self.verdict,
            "nonfinite_elements": self.nonfinite_elements,
        }


def _json_float(value: float) -> float | str:
    """A float that survives ``json.dumps(..., allow_nan=False)``.

    Non-finite values become their names (``"inf"``/``"-inf"``/``"nan"``):
    the alternative is a strict encoder crashing on the report of the very
    defect the comparison exists to surface.
    """
    if math.isnan(value):
        return "nan"
    if math.isinf(value):
        return "inf" if value > 0 else "-inf"
    return value


# ---------------------------------------------------------------------------
# The common surface
# ---------------------------------------------------------------------------


@runtime_checkable
class WeightSource(Protocol):
    """One interface over DCP checkpoints and safetensors exports.

    This is the interface that lets a ``FIRST_SAVE`` gate and an ``EXPORT``
    gate compare "the checkpoint the trainer wrote" against "the artifact the
    converter produced" without either side knowing the other's format. Both
    readers guarantee the invariant that matters: a source whose
    :meth:`tensor_keys` is empty cannot exist — construction raises, because
    downstream an empty key set becomes ``all([])``.
    """

    path: str

    def tensor_keys(self) -> tuple[str, ...]:
        """Tensor keys, sorted. Guaranteed non-empty for any live instance."""
        ...

    def nontensor_keys(self) -> tuple[str, ...]:
        """Non-tensor state-dict entries (e.g. DCP ``_extra_state`` byte blobs).

        Exposed rather than hidden because silently dropping 8,042 of 8,970
        metadata entries (the real ratio in the audited 26B checkpoint) is how
        "the checkpoint contains X" claims drift from the checkpoint.
        """
        ...

    def shape(self, key: str) -> tuple[int, ...]: ...

    def dtype(self, key: str) -> Any:
        """A ``torch.dtype``, whatever the on-disk format calls it."""
        ...

    def chunks(self, key: str) -> tuple[Chunk, ...]: ...

    def read_chunk(self, key: str, offsets: Sequence[int]) -> Any:
        """Read exactly one stored chunk, addressed by its offsets tuple."""
        ...

    def read_box(self, key: str, lo: Sequence[int], hi: Sequence[int]) -> ReadResult:
        """Assemble the sub-box ``[lo, hi)``, reading only overlapping storage."""
        ...

    def read_full(self, key: str) -> ReadResult:
        """Read the whole tensor, verifying complete stored coverage."""
        ...

    def close(self) -> None:
        """Release any cached handles. Idempotent."""
        ...


# ---------------------------------------------------------------------------
# DCP
# ---------------------------------------------------------------------------


class DcpReader:
    """Read-only partial access to a ``torch.distributed.checkpoint`` directory.

    Construction costs are the measured ones: parsing ``.metadata`` is a
    ``pickle.load`` (~180 MB RSS on the audited 51.7 GB checkpoint), and each
    chunk read afterwards costs only that chunk's bytes (~4 MB each for the
    audited experts; 128 of them added 18 MB). Requesting a *full* large
    tensor is a different operation with a different bill — see
    :func:`compare_keys` for comparisons that do not materialize one.

    Args:
        path: Directory containing ``.metadata`` and the shard files.

    Raises:
        CheckpointFormatError: If ``path`` is not a readable DCP directory,
            if the metadata cannot be parsed, or if it declares zero tensors.
            The last case is a hard error, not an empty reader: an empty
            reader is an ``all([])`` waiting to happen.
    """

    def __init__(self, path: str | os.PathLike[str]) -> None:
        self.path = os.fspath(path)
        if not Path(self.path).is_dir():
            raise CheckpointFormatError("not a directory", path=self.path)
        md_path = Path(self.path) / DCP_METADATA_FILENAME
        if not md_path.exists():
            raise CheckpointFormatError(
                f"no {DCP_METADATA_FILENAME!r}; this is not a DCP checkpoint",
                path=self.path,
            )
        try:
            # Lazy: the package must import without torch installed.
            from torch.distributed.checkpoint import FileSystemReader
        except ImportError as exc:
            raise CheckpointFormatError(
                "torch.distributed.checkpoint is unavailable; cannot read DCP",
                path=self.path,
            ) from exc
        try:
            metadata = FileSystemReader(self.path).read_metadata()
        except Exception as exc:  # pickle/CRC errors surface here, fail closed
            raise CheckpointFormatError(
                f"failed to parse DCP metadata ({type(exc).__name__}: {exc})",
                path=self.path,
            ) from exc

        try:
            from torch.distributed.checkpoint.metadata import TensorStorageMetadata
        except ImportError as exc:
            raise CheckpointFormatError(
                "cannot import DCP metadata types; torch install is incomplete",
                path=self.path,
            ) from exc

        # state_dict_metadata is NOT a tensor list: in the audited 26B
        # checkpoint only 928 of 8,970 entries were tensors; the rest were
        # BytesStorageMetadata _extra_state blobs. Filter by type, and keep the
        # remainder visible instead of silently dropping it.
        tensor_md: dict[str, Any] = {}
        nontensor: list[str] = []
        for fqn, entry in metadata.state_dict_metadata.items():
            if isinstance(entry, TensorStorageMetadata):
                tensor_md[fqn] = entry
            else:
                nontensor.append(fqn)
        if not tensor_md:
            raise CheckpointFormatError(
                f"metadata declares 0 tensor entries "
                f"({len(nontensor)} non-tensor entries only). An empty key set "
                f"turns every downstream all(...) into the vacuous-truth bug, "
                f"so this source is refused",
                path=self.path,
            )

        # MetadataIndex.offset is None for byte blobs; those are not chunks and
        # must not enter the offset index.
        storage: dict[str, dict[tuple[int, ...], Any]] = {}
        skipped_blobs = 0
        for index, sinfo in metadata.storage_data.items():
            # Not `fqn`: that name is already bound to `str` by the loop above, and
            # rebinding it to `Any | None` is a mypy error — but only when torch is
            # installed, because without it MetadataIndex is untyped and everything
            # collapses to Any. The type checker was running against a different
            # program than CI was.
            index_fqn = getattr(index, "fqn", None)
            offsets = getattr(index, "offset", None)
            if index_fqn is None or offsets is None:
                skipped_blobs += 1
                continue
            storage.setdefault(index_fqn, {})[tuple(int(o) for o in offsets)] = sinfo

        self._tensor_md = tensor_md
        self._nontensor_keys = tuple(sorted(nontensor))
        self._storage = storage
        self._skipped_blobs = skipped_blobs
        self._metadata = metadata

    # -- key surface -----------------------------------------------------------

    def tensor_keys(self) -> tuple[str, ...]:
        return tuple(sorted(self._tensor_md))

    def nontensor_keys(self) -> tuple[str, ...]:
        return self._nontensor_keys

    def _meta(self, key: str) -> Any:
        try:
            return self._tensor_md[key]
        except KeyError:
            sample = [k for k in self.tensor_keys() if key in k][:3]
            hint = f" similar keys: {sample}" if sample else ""
            raise TensorNotFoundError(
                f"tensor key not found ({len(self._tensor_md)} present).{hint}",
                path=self.path,
                key=key,
            ) from None

    def shape(self, key: str) -> tuple[int, ...]:
        return tuple(int(d) for d in self._meta(key).size)

    def dtype(self, key: str) -> Any:
        return self._meta(key).properties.dtype

    def chunks(self, key: str) -> tuple[Chunk, ...]:
        meta = self._meta(key)
        return tuple(
            sorted(
                (
                    Chunk(
                        offsets=tuple(int(o) for o in c.offsets),
                        sizes=tuple(int(s) for s in c.sizes),
                    )
                    for c in meta.chunks
                ),
                key=lambda ch: ch.offsets,
            )
        )

    # -- reads -------------------------------------------------------------------

    def read_chunk(self, key: str, offsets: Sequence[int]) -> Any:
        """Read exactly one stored chunk identified by its offsets tuple.

        The byte range is a complete ``torch.save`` archive and self-validates
        in ``torch.load``, so a corrupt range raises :class:`ChunkReadError`
        rather than returning plausible garbage.

        Args:
            key: Tensor key.
            offsets: The chunk's offsets within the global tensor, as reported
                by :meth:`chunks`.

        Returns:
            The stored ``torch.Tensor``, on CPU, in its on-disk dtype.

        Raises:
            ChunkReadError: No chunk is stored at ``offsets``, the range is out
                of bounds/truncated, or the archive fails validation.
        """
        self._meta(key)  # normalize "unknown key" to TensorNotFoundError
        off = tuple(int(o) for o in offsets)
        sinfo = self._storage.get(key, {}).get(off)
        if sinfo is None:
            known = sorted(self._storage.get(key, {}))[:8]
            raise ChunkReadError(
                f"no chunk stored at the requested offsets; stored offsets begin: {known}",
                path=self.path,
                key=key,
                offsets=off,
            )
        tensor, _ = self._read_blob(key, off, sinfo)
        return tensor

    def _read_blob(self, key: str, offsets: tuple[int, ...], sinfo: Any) -> tuple[Any, int]:
        """Read one storage record, returning (tensor, bytes_read)."""
        blob_path = Path(self.path) / sinfo.relative_path
        try:
            with blob_path.open("rb") as handle:
                handle.seek(sinfo.offset)
                buf = handle.read(sinfo.length)
        except OSError as exc:
            raise ChunkReadError(
                f"cannot read shard range [{sinfo.offset}, {sinfo.offset + sinfo.length})",
                path=self.path,
                key=key,
                offsets=offsets,
                original=exc,
            ) from exc
        if len(buf) != sinfo.length:
            raise ChunkReadError(
                f"short read: got {len(buf)} of {sinfo.length} bytes; shard is truncated",
                path=self.path,
                key=key,
                offsets=offsets,
            )
        # Cheap magic check first so a mis-addressed range says what it is
        # before torch.load's (also reliable, but less specific) rejection.
        if buf[:4] != b"PK\x03\x04":
            raise ChunkReadError(
                "range does not begin with the PK zip magic of a torch.save "
                "archive; the stored offsets are wrong or the shard is corrupt",
                path=self.path,
                key=key,
                offsets=offsets,
            )
        try:
            import torch

            tensor = torch.load(io.BytesIO(buf), map_location="cpu", weights_only=True)
        except Exception as exc:
            # UnpicklingError (bad offset) / PytorchStreamReader (truncated):
            # the self-validation the module docstring promises, surfaced as
            # our error type with the key attached.
            raise ChunkReadError(
                "stored chunk failed torch.save archive self-validation",
                path=self.path,
                key=key,
                offsets=offsets,
                original=exc,
            ) from exc
        if not torch.is_tensor(tensor):
            raise ChunkReadError(
                f"archive at this range holds a {type(tensor).__name__}, not a Tensor; "
                f"the offset index and the shard disagree",
                path=self.path,
                key=key,
                offsets=offsets,
            )
        return tensor, len(buf)

    def read_box(self, key: str, lo: Sequence[int], hi: Sequence[int]) -> ReadResult:
        """Assemble the sub-box ``[lo, hi)`` reading only overlapping chunks.

        Coverage is verified, not assumed: stored chunks for a valid DCP
        tensor are disjoint, so we check that no two written overlap regions
        intersect and that their total volume equals the box volume. Any other
        outcome raises :class:`IncompleteCoverageError` — the caller must never
        receive zeros that look like data.

        Raises:
            ValueError: If ``lo``/``hi`` are not a valid box inside the tensor.
            IncompleteCoverageError: If stored chunks fail to tile the box.
        """
        meta = self._meta(key)
        shape = tuple(int(d) for d in meta.size)
        nd = len(shape)
        lo_t, hi_t = _validate_box(shape, lo, hi, key=key)
        box_dims = [hi_t[i] - lo_t[i] for i in range(nd)]
        expected = math.prod(box_dims)

        import torch

        out = torch.zeros(box_dims, dtype=meta.properties.dtype)
        written: list[tuple[tuple[int, ...], tuple[int, ...]]] = []
        covered = 0
        bytes_read = 0
        chunks_read = 0

        for chunk in meta.chunks:
            co = tuple(int(o) for o in chunk.offsets)
            cs = tuple(int(s) for s in chunk.sizes)
            ov_lo = tuple(max(lo_t[i], co[i]) for i in range(nd))
            ov_hi = tuple(min(hi_t[i], co[i] + cs[i]) for i in range(nd))
            if any(ov_hi[i] <= ov_lo[i] for i in range(nd)):
                continue
            sinfo = self._storage.get(key, {}).get(co)
            if sinfo is None:
                raise IncompleteCoverageError(
                    "metadata declares a chunk that has no storage record",
                    path=self.path,
                    key=key,
                    elements_covered=covered,
                    elements_expected=expected,
                )
            tensor, nbytes = self._read_blob(key, co, sinfo)
            src = tuple(slice(ov_lo[i] - co[i], ov_hi[i] - co[i]) for i in range(nd))
            dst = tuple(slice(ov_lo[i] - lo_t[i], ov_hi[i] - lo_t[i]) for i in range(nd))
            out[dst] = tensor[src]
            written.append(((ov_lo), (ov_hi)))
            covered += math.prod(ov_hi[i] - ov_lo[i] for i in range(nd))
            bytes_read += nbytes
            chunks_read += 1

        for i in range(len(written)):
            for j in range(i + 1, len(written)):
                if _boxes_overlap(written[i], written[j]):
                    raise IncompleteCoverageError(
                        "stored chunks overlap; coverage volume would be "
                        "double-counted and holes hidden",
                        path=self.path,
                        key=key,
                        elements_covered=covered,
                        elements_expected=expected,
                    )
        if covered != expected:
            raise IncompleteCoverageError(
                "stored chunks do not cover the requested box; the uncovered "
                "region would otherwise read back as zeros indistinguishable "
                "from data",
                path=self.path,
                key=key,
                elements_covered=covered,
                elements_expected=expected,
            )
        return ReadResult(
            key=key,
            tensor=out,
            chunks_read=chunks_read,
            elements_covered=covered,
            elements_expected=expected,
            bytes_read=bytes_read,
        )

    def read_full(self, key: str) -> ReadResult:
        """Read the whole tensor, failing unless stored coverage is exact.

        This is the fix for the probe's silent-zero defect: the returned
        :class:`ReadResult` has ``elements_covered == numel(shape)`` *because
        the reader proved it*, not because the caller assumed it.
        """
        return self.read_box(key, [0] * len(self.shape(key)), list(self.shape(key)))

    def close(self) -> None:
        """No-op: shard files are opened per read and closed immediately."""


def _validate_box(
    shape: tuple[int, ...], lo: Sequence[int], hi: Sequence[int], *, key: str
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    nd = len(shape)
    if len(lo) != nd or len(hi) != nd:
        raise ValueError(
            f"box rank mismatch for key {key!r}: tensor is {nd}-D, "
            f"lo has {len(lo)}, hi has {len(hi)}"
        )
    lo_t = tuple(int(v) for v in lo)
    hi_t = tuple(int(v) for v in hi)
    for i in range(nd):
        if not (0 <= lo_t[i] <= hi_t[i] <= shape[i]):
            raise ValueError(
                f"box dim {i} invalid for key {key!r}: [{lo_t[i]}, {hi_t[i]}) "
                f"outside tensor extent {shape[i]}"
            )
    # A zero-extent dimension (lo == hi) assembles nothing yet reports complete
    # coverage — the vacuous-truth defect. Dims where the tensor itself is
    # empty (shape[i] == 0 forces lo == hi == 0) are exempt: full coverage of
    # zero declared elements is a true statement, not a skipped read.
    for i in range(nd):
        if lo_t[i] == hi_t[i] and shape[i] > 0:
            raise ValueError(
                f"zero-extent box for key {key!r} on dim {i}: lo={lo_t[i]} == "
                f"hi={hi_t[i]} while tensor dim {i} has extent {shape[i]}; the "
                f"read would touch no elements and still report complete coverage"
            )
    return lo_t, hi_t


def _boxes_overlap(
    a: tuple[tuple[int, ...], tuple[int, ...]],
    b: tuple[tuple[int, ...], tuple[int, ...]],
) -> bool:
    return all(min(a[1][i], b[1][i]) > max(a[0][i], b[0][i]) for i in range(len(a[0])))


# ---------------------------------------------------------------------------
# safetensors
# ---------------------------------------------------------------------------


class _HandleCache:
    """Bounded LRU cache of open shard handles.

    The reference probe kept one handle per shard forever. On a many-shard
    export that is a file-descriptor leak across a long verification run, so
    handles are evicted least-recently-used beyond ``capacity``.
    """

    def __init__(self, capacity: int, opener: Callable[[str], Any]) -> None:
        if capacity < 1:
            raise ValueError("handle cache capacity must be >= 1")
        self._capacity = capacity
        self._opener = opener
        self._items: OrderedDict[str, Any] = OrderedDict()

    def get(self, shard_path: str) -> Any:
        handle = self._items.pop(shard_path, None)
        if handle is None:
            handle = self._opener(shard_path)
        self._items[shard_path] = handle
        while len(self._items) > self._capacity:
            _, evicted = self._items.popitem(last=False)
            _release(evicted)
        return handle

    def close(self) -> None:
        while self._items:
            _, handle = self._items.popitem(last=False)
            _release(handle)


def _release(handle: Any) -> None:
    close = getattr(handle, "close", None)
    if callable(close):
        # releasing a busted handle must not mask real errors
        with suppress(Exception):
            close()


class SafetensorsReader:
    """Read-only partial access to an HF-style safetensors export.

    Presents exactly the :class:`WeightSource` surface, so gates compare a DCP
    checkpoint against its exported form through one code path. Each tensor
    lives contiguously in exactly one shard, which is why :meth:`chunks`
    always reports a single whole-tensor chunk and why incomplete coverage
    cannot occur silently — an unreadable slice raises rather than zero-fills.

    Args:
        path: Directory containing ``model.safetensors.index.json`` and/or
            ``*.safetensors`` shards, or a single ``.safetensors`` file.
        handle_cache_size: Maximum simultaneously open shard handles.

    Raises:
        CheckpointFormatError: If no safetensors shards are found, the index is
            malformed, or the export declares zero tensors (refused on the same
            vacuous-truth grounds as an empty DCP).
    """

    def __init__(self, path: str | os.PathLike[str], *, handle_cache_size: int = 16) -> None:
        try:
            # Lazy: the package must import without safetensors installed.
            from safetensors import safe_open
        except ImportError as exc:
            raise CheckpointFormatError(
                "the 'safetensors' package is unavailable",
                path=os.fspath(path),
            ) from exc
        self._safe_open = safe_open

        given = os.fspath(path)
        if Path(given).is_dir():
            self.path = given
            weight_map = self._load_or_build_index(given)
        elif Path(given).is_file() and given.endswith(".safetensors"):
            self.path = str(Path(given).parent) or "."
            fname = Path(given).name
            weight_map = {k: fname for k in self._shard_keys(str(Path(self.path) / fname))}
        else:
            raise CheckpointFormatError(
                "neither a safetensors directory nor a .safetensors file",
                path=given,
            )

        if not weight_map:
            raise CheckpointFormatError(
                "export declares 0 tensors; an empty key set turns every "
                "downstream all(...) into the vacuous-truth bug, so this "
                "source is refused",
                path=self.path,
            )
        self._weight_map = weight_map
        self._handles = _HandleCache(
            handle_cache_size, lambda p: self._safe_open(p, framework="pt")
        )

    def _shard_keys(self, shard_path: str) -> list[str]:
        try:
            with self._safe_open(shard_path, framework="pt") as handle:
                return list(handle.keys())
        except Exception as exc:
            raise CheckpointFormatError(
                f"cannot read shard index ({type(exc).__name__}: {exc})",
                path=shard_path,
            ) from exc

    def _load_or_build_index(self, directory: str) -> dict[str, str]:
        index_path = Path(directory) / SAFETENSORS_INDEX_FILENAME
        if index_path.exists():
            try:
                with index_path.open(encoding="utf-8") as handle:
                    raw_map: Any = json.load(handle)["weight_map"]
            except (OSError, json.JSONDecodeError, KeyError) as exc:
                raise CheckpointFormatError(
                    f"malformed {SAFETENSORS_INDEX_FILENAME} ({type(exc).__name__}: {exc})",
                    path=directory,
                ) from exc
            if not isinstance(raw_map, dict):
                raise CheckpointFormatError(
                    f"{SAFETENSORS_INDEX_FILENAME}: 'weight_map' is not an object",
                    path=directory,
                )
            return dict(raw_map)
        shards = sorted(
            n.name for n in Path(directory).iterdir() if n.name.endswith(".safetensors")
        )
        if not shards:
            raise CheckpointFormatError(
                "directory contains neither an index nor any .safetensors shards",
                path=directory,
            )
        weight_map: dict[str, str] = {}
        for shard in shards:
            for k in self._shard_keys(str(Path(directory) / shard)):
                if k in weight_map:
                    raise CheckpointFormatError(
                        "tensor key appears in two shards; the export is corrupt",
                        path=directory,
                        key=k,
                    )
                weight_map[k] = shard
        return weight_map

    # -- key surface -----------------------------------------------------------

    def tensor_keys(self) -> tuple[str, ...]:
        return tuple(sorted(self._weight_map))

    def nontensor_keys(self) -> tuple[str, ...]:
        # safetensors exports carry tensors only; anything else lives in config
        # JSON, which this reader deliberately does not claim to cover.
        return ()

    def _handle_and_key(self, key: str) -> tuple[Any, str]:
        shard = self._weight_map.get(key)
        if shard is None:
            sample = [k for k in self.tensor_keys() if key in k][:3]
            hint = f" similar keys: {sample}" if sample else ""
            raise TensorNotFoundError(
                f"tensor key not found ({len(self._weight_map)} present).{hint}",
                path=self.path,
                key=key,
            )
        shard_path = str(Path(self.path) / shard)
        if not Path(shard_path).exists():
            raise CheckpointFormatError(
                f"weight_map points at missing shard {shard!r}",
                path=self.path,
                key=key,
            )
        try:
            return self._handles.get(shard_path), shard_path
        except Exception as exc:
            raise CheckpointFormatError(
                f"cannot open shard {shard!r} ({type(exc).__name__}: {exc})",
                path=self.path,
                key=key,
            ) from exc

    def _slice(self, key: str) -> Any:
        handle, shard_path = self._handle_and_key(key)
        try:
            return handle.get_slice(key)
        except Exception as exc:
            raise CheckpointFormatError(
                f"weight_map names {key!r} but shard {shard_path!r} does not "
                f"contain it ({type(exc).__name__}: {exc})",
                path=self.path,
                key=key,
            ) from exc

    def shape(self, key: str) -> tuple[int, ...]:
        return tuple(int(d) for d in self._slice(key).get_shape())

    def dtype(self, key: str) -> Any:
        name = self._slice(key).get_dtype()
        return _safetensors_dtype(name, path=self.path, key=key)

    def chunks(self, key: str) -> tuple[Chunk, ...]:
        shape = self.shape(key)
        return (Chunk(offsets=(0,) * len(shape), sizes=shape),)

    # -- reads -------------------------------------------------------------------

    def read_chunk(self, key: str, offsets: Sequence[int]) -> Any:
        off = tuple(int(o) for o in offsets)
        if any(o != 0 for o in off) or len(off) != len(self.shape(key)):
            raise ChunkReadError(
                "safetensors stores each tensor contiguously in one shard; the "
                "only valid chunk offsets are all-zero",
                path=self.path,
                key=key,
                offsets=off,
            )
        handle, _ = self._handle_and_key(key)
        return handle.get_tensor(key)

    def read_box(self, key: str, lo: Sequence[int], hi: Sequence[int]) -> ReadResult:
        shape = self.shape(key)
        lo_t, hi_t = _validate_box(shape, lo, hi, key=key)
        sl = self._slice(key)
        idx = tuple(slice(lo_t[i], hi_t[i]) for i in range(len(shape)))
        tensor = sl[idx]
        numel = math.prod(hi_t[i] - lo_t[i] for i in range(len(shape)))
        return ReadResult(
            key=key,
            tensor=tensor,
            chunks_read=1,
            elements_covered=numel,
            elements_expected=numel,
            bytes_read=tensor.numel() * tensor.element_size(),
        )

    def read_full(self, key: str) -> ReadResult:
        shape = self.shape(key)
        return self.read_box(key, (0,) * len(shape), shape)

    def close(self) -> None:
        self._handles.close()

    def __enter__(self) -> SafetensorsReader:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()


def _safetensors_dtype(name: str, *, path: str, key: str) -> Any:
    import torch

    attr = {
        "F64": "float64",
        "F32": "float32",
        "F16": "float16",
        "BF16": "bfloat16",
        "I64": "int64",
        "U64": "uint64",
        "I32": "int32",
        "U32": "uint32",
        "I16": "int16",
        "U16": "uint16",
        "I8": "int8",
        "U8": "uint8",
        "BOOL": "bool",
        "F8_E4M3": "float8_e4m3fn",
        "F8_E5M2": "float8_e5m2",
    }.get(name)
    dtype = getattr(torch, attr, None) if attr is not None else None
    if dtype is None:
        raise CheckpointFormatError(
            f"unsupported safetensors dtype {name!r} in this torch build",
            path=path,
            key=key,
        )
    return dtype


# ---------------------------------------------------------------------------
# Sniffer
# ---------------------------------------------------------------------------


def open_weights(path: str | os.PathLike[str]) -> WeightSource:
    """Open a checkpoint directory or file as the right kind of source.

    Sniffs by layout, not by hope: DCP is a directory with ``.metadata``;
    safetensors is an index and/or ``*.safetensors`` shards; plain ``.bin``
    pickles are recognized and *refused*, because a whole-file ``torch.load``
    cannot be partially read or safely verified and silently loading one in a
    gate is how verification quietly stopped happening in the audited estate.

    Args:
        path: Checkpoint directory, or a single ``.safetensors`` file.

    Returns:
        A :class:`WeightSource` with a provably non-empty tensor key set.

    Raises:
        CheckpointFormatError: Path missing; both formats present (ambiguous —
            names what it found); only ``.bin`` pickles present; or nothing
            recognizable at all, in which case the error lists what was there.
    """
    p = os.fspath(path)
    if not Path(p).exists():
        raise CheckpointFormatError("path does not exist", path=p)

    if Path(p).is_file():
        if p.endswith(".safetensors"):
            return SafetensorsReader(p)
        if p.endswith(".bin"):
            raise CheckpointFormatError(
                "plain torch pickle (.bin) cannot be partially read or "
                "verified chunk-wise; pass a DCP directory or a safetensors export",
                path=p,
            )
        raise CheckpointFormatError(
            f"unrecognized checkpoint file (name: {Path(p).name!r})",
            path=p,
        )

    names = {n.name for n in Path(p).iterdir()}
    has_dcp = DCP_METADATA_FILENAME in names
    st_shards = sorted(n for n in names if n.endswith(".safetensors"))
    has_st = bool(st_shards) or SAFETENSORS_INDEX_FILENAME in names

    if has_dcp and has_st:
        raise CheckpointFormatError(
            f"ambiguous: both a DCP {DCP_METADATA_FILENAME!r} and safetensors "
            f"shards {st_shards[:3]} are present; open the intended one "
            f"explicitly with DcpReader or SafetensorsReader",
            path=p,
        )
    if has_dcp:
        return DcpReader(p)
    if has_st:
        return SafetensorsReader(p)

    bins = sorted(n for n in names if n.endswith(".bin"))
    if bins:
        raise CheckpointFormatError(
            f"directory holds only plain .bin pickles {bins[:4]}; these cannot "
            f"be partially read or verified chunk-wise — provide the DCP or "
            f"safetensors form of this artifact",
            path=p,
        )
    found = ", ".join(sorted(names)[:10]) or "<empty directory>"
    raise CheckpointFormatError(
        f"no recognizable checkpoint layout; found: {found}",
        path=p,
    )


# ---------------------------------------------------------------------------
# Numerically stable streaming comparison
# ---------------------------------------------------------------------------


def compare_keys(
    source_a: WeightSource,
    source_b: WeightSource,
    key: str,
    *,
    block_rows: int = 4096,
    close_max_abs_diff: float = 1e-2,
    close_min_cosine: float = 0.999,
) -> TensorComparison:
    """Compare ``key`` across two sources without materializing it whole.

    Streams blocks of ``block_rows`` rows, accumulating every sum, sum of
    squares and the dot product in **float64**. This is the direct fix for the
    probe defect in which float32 reductions underflowed and reported a
    tensor's cosine with itself as 1.80. As a guard against that bug ever
    returning, when the two tensors are bitwise identical and finite this
    function asserts the computed cosine is 1.0: if the accumulator is broken,
    the comparison raises instead of reporting a number. Identical ±inf pairs
    are bitwise equal but have no finite norm to self-check against, so they
    report ``EXACT`` with ``nonfinite_elements`` set — parity over bytes, with
    the poison surfaced rather than laundered into a tolerance verdict.

    Memory: roughly ``block_rows * row_size * (2 dtypes + float64 temporaries)``
    — about 0.5 GB for ``block_rows=4096`` on a 2816-wide tensor, versus 11.5
    GB peak for reading the full embedding and upcasting it.

    Args:
        source_a: Left-hand source (e.g. the DCP checkpoint).
        source_b: Right-hand source (e.g. the safetensors export).
        key: Tensor key, looked up in both sources.
        block_rows: Rows compared per read; tune for RSS ceilings.
        close_max_abs_diff: ``max_abs_diff`` ceiling for a ``CLOSE`` verdict.
        close_min_cosine: Cosine floor for a ``CLOSE`` verdict.

    Returns:
        A :class:`TensorComparison`. ``EXACT`` means bitwise identical -- same
        declared encoding AND same decoded values, elementwise;
        ``CLOSE`` means within both tolerances; ``DIFFER`` means finite
        content outside both tolerances; ``NON_FINITE`` means NaN or Inf was
        present without bitwise identity — no tolerance answer over poisoned
        content is honest, because IEEE comparisons swallow NaN
        (``max(0.0, nan) == 0.0``); ``DTYPE_MISMATCH`` means the sources
        declare different encodings for the same logical tensor: a positively
        observed metadata finding, adjudicated before any streaming, so the
        record carries the same "compared nothing, visibly so" expression as
        the two branches below (``elements=0`` naming the compared
        denominator, ``bitwise_equal=False``, unbounded diffs, ``cosine``
        None) and no value statistic claims a measurement;
        ``SHAPE_MISMATCH`` means the two sources
        disagree on the shape (zero elements compared, by design visibly so);
        ``NO_ELEMENTS`` means the shapes agree but declare zero elements, so
        the comparison abstains using exactly the mismatch branch's
        "compared nothing" expression (``bitwise_equal=False``, unbounded
        diffs, ``elements=0`` naming the denominator): an empty tensor is a
        legitimate artifact, but an unread nothing must abstain, never pass.

    Raises:
        TensorNotFoundError: If ``key`` is absent from either source.
        CheckpointFormatError: If either source declares a negative extent for
            ``key``. This raises rather than returning a verdict because such
            metadata describes no tensor at all, so both "identical" and
            "different" would be fabrications — see the branch comment.
        AssertionError: If the float64 accumulator fails its self-check —
            bitwise-equal inputs must produce cosine 1.0 and zero max
            difference; anything else means the numerics are lying.
    """
    if block_rows < 1:
        raise ValueError(f"block_rows must be >= 1, got {block_rows}")

    import torch

    if key not in set(source_a.tensor_keys()):
        raise TensorNotFoundError("key absent from left source", path=source_a.path, key=key)
    if key not in set(source_b.tensor_keys()):
        raise TensorNotFoundError("key absent from right source", path=source_b.path, key=key)

    shape_a = source_a.shape(key)
    shape_b = source_b.shape(key)
    dtype_a = source_a.dtype(key)
    dtype_b = source_b.dtype(key)

    # Extent validation runs ABOVE both the shape-mismatch branch and the
    # zero-element abstention, because a negative extent is not a disagreement
    # and not an emptiness -- it is metadata that cannot describe any tensor,
    # and every downstream branch would render it as one of those two lies.
    #
    # Read as identity: shape (-1,) on both sides passes the equality check,
    # yields total == -1 which slips past the `total == 0` abstention, makes
    # range(0, -1, block_rows) empty, and lets the `bitwise = True` initialiser
    # survive a loop that never ran -- EXACT over zero reads with elements=-1
    # as its denominator. That is the founding all([]) incident wearing a
    # different predicate, one floor below the abstention that claims to close
    # it. Read as divergence: (-1,) against (3,) would return SHAPE_MISMATCH,
    # asserting a difference between a real tensor and a description of
    # nothing. A claim of divergence never observed is exactly as wrong as a
    # claim of identity never observed, so neither branch may own this.
    #
    # It raises rather than abstains because, unlike an empty tensor, no valid
    # writer produces this: DcpReader.shape() is `tuple(int(d) for d in
    # meta.size)` over an unvalidated pickle, so a negative dim means the
    # metadata is corrupt and the artifact -- not this comparison -- is the
    # thing that failed.
    for _shape, _src in ((shape_a, source_a), (shape_b, source_b)):
        _bad = [i for i, d in enumerate(_shape) if d < 0]
        if _bad:
            raise CheckpointFormatError(
                f"tensor metadata declares negative extent(s) at "
                f"dim(s) {_bad} of shape {_shape!r} -- no tensor has a "
                f"negative dimension; the metadata is corrupt",
                path=_src.path,
                key=key,
            )

    # Encoding disagreement: a positively observed finding, owned by this
    # branch and no other. The pre-fix comparator fetched dtype_a and dtype_b
    # into the lines above, recorded both on the returned record, and never
    # compared them -- identity was then decided by torch.equal, which
    # type-promotes and compares DECODED VALUES. A bf16 export against its
    # f32 source whose content is exactly representable in both encodings
    # (any lossless round-trip; every small integral constant in learned
    # scales and router buffers) streamed both sides, promoted, agreed, and
    # returned EXACT with bitwise_equal=True -- two artifacts holding
    # different-width bits, reported as identical. The requantized twin case
    # was no better: a successful bf16 conversion reported honest-looking
    # value statistics (near-CLOSE numbers riding a DIFFER verdict) that
    # buried the operative fact that the encoding had changed.
    #
    # Why this branch, and not a neighbour, owns the case:
    #
    # Not NO_ELEMENTS. The abstention below exists because nothing was
    # examined and so nothing may be claimed. Here something WAS examined --
    # the two declared encodings, two lines above -- and they differ.
    # Abstaining over an observed difference is the mirrored lie doctrine 5
    # forbids, and it was live pre-fix: (0, 8)-bf16 against (0, 8)-f32 fell
    # into the total==0 branch and reported "nothing examined" about two
    # artifacts whose encodings demonstrably disagreed.
    #
    # Not SHAPE_MISMATCH. The geometry agrees: rank, extents and addressing
    # of the element grid are identical. What differs is the width and
    # interpretation of each storage slot -- same evidential class (metadata
    # positively observed), different axis, and reusing the geometry token
    # would mint one false claim to retire another.
    #
    # Not DIFFER. That verdict is defined over streamed content ("finite
    # content outside both tolerances"), and nothing is streamed here -- by
    # choice. Streaming cannot repair the operative claim (there is no
    # encoding in which these bits are the same tensor), the value
    # statistics it would mint are precisely the laundering vector (a good
    # requantization reads as near-agreement), and the one verdict string
    # cannot co-express "encodings differ" with "values close" without
    # inviting a downgrade of a finding into a pass. The caller who
    # DELIBERATELY wants cross-encoding value distance (a conversion-
    # fidelity audit) has read_box/read_full to compute it; what closes
    # here is the accidental laundering of an encoding change into EXACT.
    #
    # Not a raise, either: valid writers produce dtype mismatches (a
    # converter legitimately emitting bf16). Both artifacts are well-formed;
    # this is a finding -- data, like a shape mismatch -- not corruption
    # like a negative extent, which describes no tensor at all.
    #
    # The numeric fields therefore carry the module's one established
    # "compared nothing, visibly so" expression -- elements=0 naming the
    # compared denominator, chunks/bytes 0, unbounded diffs, cosine None --
    # identical to the SHAPE_MISMATCH branch and to parity's _metadata_entry
    # for this same finding, so every metadata adjudication presents one
    # record shape downstream. The fields that must be positively true are:
    # the two dtype strings (they ARE the finding), the shapes (recorded as
    # declared), and bitwise_equal=False. That False is enforced by ROUTING,
    # not by clamping a computed True: the branch returns above the
    # `bitwise = True` initialiser and above every torch.equal call, so the
    # promotion predicate that minted the false claim is unreachable across
    # encodings. Same-dtype comparisons never touch this branch, and for
    # them torch.equal promotion is a no-op -- bitwise_equal's meaning is
    # thereby narrowed to exactly what it says: same encoding, same values.
    #
    # Precedence: above shape and above zero-element, matching parity's
    # adjudication order (dtype mismatch, then shape mismatch, then
    # zero-element abstention). Two layers of one framework must never
    # disagree about which finding owns a key, or which door the operator
    # walked in decides which defect gets named. The predicate is dtype-
    # OBJECT inequality, not string equality: both readers return canonical
    # torch.dtype singletons (_safetensors_dtype maps each on-disk name to
    # exactly one, and raises on any name the torch build lacks; DCP reads
    # properties.dtype from metadata), aliases are the same object, and
    # distinct torch dtypes always denote distinct storage encodings -- so
    # `!=` here can neither misfire on a renamed-equal encoding nor pass a
    # genuinely different one.
    if dtype_a != dtype_b:
        return TensorComparison(
            key=key,
            elements=0,
            shape_a=shape_a,
            shape_b=shape_b,
            dtype_a=str(dtype_a),
            dtype_b=str(dtype_b),
            bitwise_equal=False,
            mismatched_elements=0,
            max_abs_diff=math.inf,
            mean_abs_diff=math.inf,
            cosine=None,
            rms_a=0.0,
            rms_b=0.0,
            chunks_read=0,
            bytes_read=0,
            verdict=VERDICT_DTYPE_MISMATCH,
            nonfinite_elements=0,
        )

    if shape_a != shape_b:
        return TensorComparison(
            key=key,
            elements=0,
            shape_a=shape_a,
            shape_b=shape_b,
            dtype_a=str(dtype_a),
            dtype_b=str(dtype_b),
            bitwise_equal=False,
            mismatched_elements=0,
            max_abs_diff=math.inf,
            mean_abs_diff=math.inf,
            cosine=None,
            rms_a=0.0,
            rms_b=0.0,
            chunks_read=0,
            bytes_read=0,
            verdict=VERDICT_SHAPE_MISMATCH,
            nonfinite_elements=0,
        )

    nd = len(shape_a)
    total = math.prod(shape_a)

    if total == 0:
        # Zero elements is never a pass: EXACT here would claim verified
        # bitwise identity over content that does not exist, minted from the
        # `bitwise = True` initialiser surviving a loop that never ran -- the
        # founding all([]) incident inside the flagship comparator. An empty
        # tensor is a legitimate artifact (zero-row padding embeddings,
        # unallocated buffers), so this abstains rather than raising, and it
        # mirrors the SHAPE_MISMATCH branch's established expression of "I
        # compared nothing": bitwise_equal=False, unbounded diffs, elements=0
        # naming the denominator. Returning before the loop also covers
        # shapes like (5, 0): rows > 0 would otherwise run the stream over
        # zero-element blocks, burn real reads, and report chunks_read > 0 --
        # laundering "work happened" over a comparison of nothing.
        return TensorComparison(
            key=key,
            elements=0,
            shape_a=shape_a,
            shape_b=shape_b,
            dtype_a=str(dtype_a),
            dtype_b=str(dtype_b),
            bitwise_equal=False,
            mismatched_elements=0,
            max_abs_diff=math.inf,
            mean_abs_diff=math.inf,
            cosine=None,
            rms_a=0.0,
            rms_b=0.0,
            chunks_read=0,
            bytes_read=0,
            verdict=VERDICT_NO_ELEMENTS,
            nonfinite_elements=0,
        )

    rows = shape_a[0] if nd else 1

    sum_a = sum_b = sumsq_a = sumsq_b = dot = sum_abs = 0.0
    max_abs = 0.0
    mismatched = 0
    nonfinite = 0
    bitwise = True
    chunks_read = bytes_read = 0

    for start in range(0, rows, block_rows):
        stop = min(start + block_rows, rows)
        if nd:
            lo = (start,) + (0,) * (nd - 1)
            hi = (stop,) + shape_a[1:]
        else:
            lo, hi = (), ()
        ra = source_a.read_box(key, lo, hi)
        rb = source_b.read_box(key, lo, hi)
        chunks_read += ra.chunks_read + rb.chunks_read
        bytes_read += ra.bytes_read + rb.bytes_read
        ta, tb = ra.tensor, rb.tensor
        if not bool(torch.equal(ta, tb)):
            bitwise = False
            mismatched += int((ta != tb).sum().item())
        fa = ta.to(torch.float64)
        fb = tb.to(torch.float64)
        # Count the poison before any statistic derived from it can swallow
        # it: IEEE sum propagates NaN but max hides it (max(0.0, nan) == 0.0),
        # so the verdict needs the non-finite tally as its own fact.
        nonfinite += int((~torch.isfinite(fa)).sum().item())
        nonfinite += int((~torch.isfinite(fb)).sum().item())
        diff = (fa - fb).abs()
        sum_a += float(fa.sum().item())
        sum_b += float(fb.sum().item())
        sumsq_a += float((fa * fa).sum().item())
        sumsq_b += float((fb * fb).sum().item())
        dot += float((fa * fb).sum().item())
        sum_abs += float(diff.sum().item())
        if diff.numel():
            max_abs = max(max_abs, float(diff.max().item()))

    rms_a = math.sqrt(sumsq_a / total) if total else 0.0
    rms_b = math.sqrt(sumsq_b / total) if total else 0.0
    denom = math.sqrt(sumsq_a * sumsq_b)
    cosine = dot / denom if denom > 0.0 else None
    mean_abs = sum_abs / total if total else 0.0

    if nonfinite:
        # No finite angle exists over contaminated norms, and because IEEE
        # orderings swallow NaN — max(0.0, nan) kept max_abs at 0.0 above — a
        # poisoned, non-identical comparison must report an unbounded maximum,
        # never a numerical 0.0 that reads as agreement. Bitwise-identical
        # ±inf keeps its computed diff stats: on equal bytes a maximum
        # difference of 0.0 is the truth, not a laundered number.
        cosine = None
        if not bitwise:
            max_abs = math.inf

    # Self-check against the 1.80-cosine incident: identical bits are a ground
    # truth the accumulator cannot legitimately contradict. Gated on finite
    # content: identical ±inf has no cosine to contradict, and the guard must
    # condemn broken arithmetic, not parity over poisoned bytes.
    if bitwise and total > 0 and not nonfinite:
        if sumsq_a > 0.0:
            assert cosine is not None and abs(cosine - 1.0) <= 1e-9, (
                f"compare_keys self-check failed on {key!r}: bitwise-identical "
                f"tensors scored cosine={cosine!r}; the reduction is numerically "
                f"broken (float32 underflow class). Do not trust this result."
            )
        assert max_abs == 0.0, (
            f"compare_keys self-check failed on {key!r}: bitwise-identical "
            f"tensors produced max_abs_diff={max_abs!r}"
        )

    if bitwise:
        verdict = VERDICT_EXACT
    elif nonfinite:
        verdict = VERDICT_NON_FINITE
    elif max_abs <= close_max_abs_diff and (cosine is None or cosine > close_min_cosine):
        verdict = VERDICT_CLOSE
    else:
        verdict = VERDICT_DIFFER

    return TensorComparison(
        key=key,
        elements=total,
        shape_a=shape_a,
        shape_b=shape_b,
        dtype_a=str(dtype_a),
        dtype_b=str(dtype_b),
        bitwise_equal=bitwise,
        mismatched_elements=mismatched,
        max_abs_diff=max_abs,
        mean_abs_diff=mean_abs,
        cosine=cosine,
        rms_a=rms_a,
        rms_b=rms_b,
        chunks_read=chunks_read,
        bytes_read=bytes_read,
        verdict=verdict,
        nonfinite_elements=nonfinite,
    )


__ = Hashable  # re-export guard against accidental trimming of typing imports
