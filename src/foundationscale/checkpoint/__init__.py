"""Checkpoint weight access and cross-format comparison.

Why this subpackage exists: the forensic audit's Incident #1 was a 128-expert
MoE checkpoint saved under local names so that only 16 distinct tensors hit
disk, aliased 8 ways — 87.5% wrong while passing every check that existed
(rc=0, resume, loss, tensor counts, dtypes). Incident #6 was its mirror: the
verifier built to catch it answered ``all([]) is True`` on the corrupt
artifact. Both are prevented here structurally:

* :func:`~foundationscale.checkpoint.dcp.open_weights` refuses any source with
  zero tensor keys, so no downstream comparison can ever iterate an empty set
  and call it a match.
* :meth:`~foundationscale.checkpoint.dcp.DcpReader.read_full` proves stored
  coverage or raises :class:`IncompleteCoverageError`, instead of returning
  zeros that look like data.
* :func:`~foundationscale.checkpoint.dcp.compare_keys` streams blocks and
  accumulates in float64, with a self-check assertion, so "the embedding
  matches" never depends on an 11.5 GB materialization or on numerics that
  once printed a self-cosine of 1.80.

Everything here is read-only and CPU-only; ``torch`` and ``safetensors`` are
imported lazily inside the functions that need them, so the package imports
cleanly on control-plane nodes without an ML stack.
"""

from __future__ import annotations

from .dcp import (
    CheckpointError,
    CheckpointFormatError,
    Chunk,
    ChunkReadError,
    DcpReader,
    IncompleteCoverageError,
    ReadResult,
    SafetensorsReader,
    TensorComparison,
    TensorNotFoundError,
    WeightSource,
    compare_keys,
    open_weights,
)
from .dcp_meta import (
    CheckpointMetadata,
    StoredTensorMeta,
    load_manifest,
    read_metadata,
)

__all__ = [
    "CheckpointError",
    "CheckpointFormatError",
    "CheckpointMetadata",
    "Chunk",
    "ChunkReadError",
    "DcpReader",
    "IncompleteCoverageError",
    "ReadResult",
    "SafetensorsReader",
    "StoredTensorMeta",
    "TensorComparison",
    "TensorNotFoundError",
    "WeightSource",
    "compare_keys",
    "load_manifest",
    "open_weights",
    "read_metadata",
]
