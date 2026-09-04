"""Coverage-as-a-fact tests for the checkpoint readers.

Why this file exists
--------------------
The audit's central defect class is the check that never touched the data but
reported success. Two incidents drive everything asserted here:

* The reference DCP probe's ``read_full`` silently returned zeros wherever no
  stored chunk covered the tensor — the silent-zero variant of the
  ``all([]) is True`` bug. These tests feed :class:`DcpReader` checkpoints with
  holes, overlaps, and orphaned chunk metadata, and assert it refuses loudly
  (``IncompleteCoverageError`` with exact covered/expected counts) instead of
  handing back plausible-looking zeros.
* Two chunks that overlap can still sum to the *right volume* while covering
  the *wrong region*. These tests separate the module's volume check from its
  coverage check by constructing exactly that case and asserting the pairwise
  overlap guard fires first.

Every test is written to be able to fail: for each ``pytest.raises`` there is a
sibling must-pass control (a healthy checkpoint that reads back byte-exact), so
a reader that refused *everything* would also turn the suite red.
"""

from __future__ import annotations

import io
import json
import pickle
import struct
from pathlib import Path
from typing import Any

import pytest

from foundationscale.checkpoint.dcp import (
    DCP_METADATA_FILENAME,
    SAFETENSORS_INDEX_FILENAME,
    CheckpointFormatError,
    Chunk,
    DcpReader,
    IncompleteCoverageError,
    SafetensorsReader,
    open_weights,
)

# ---------------------------------------------------------------------------
# Synthetic checkpoint construction — torch installed or the test skips, and
# only the test that needs it pays for the import.
# ---------------------------------------------------------------------------


@pytest.fixture
def torch_mod() -> Any:
    """The torch module, or a skip, scoped to the single test that asks for it."""
    return pytest.importorskip("torch")


def _archive_bytes(tensor: Any) -> bytes:
    """Serialize one tensor exactly as :meth:`DcpReader._read_blob` expects it.

    Keeping serialization in one place means both the fixture builder and the
    tests that assert ``bytes_read`` deterministically agree on chunk sizes.
    """
    torch = pytest.importorskip("torch")
    buf = io.BytesIO()
    torch.save(tensor, buf)
    return buf.getvalue()


def _write_dcp_checkpoint(
    root: Path,
    *,
    tensors: dict[str, tuple[tuple[int, ...], list[tuple[tuple[int, ...], Any]]]],
    nontensor: list[str] | None = None,
    orphan_chunks: dict[str, list[tuple[tuple[int, ...], tuple[int, ...]]]] | None = None,
    offsetless_blobs: int = 0,
) -> None:
    """Write a minimal but structurally real DCP checkpoint into ``root``.

    Args:
        root: Directory to populate with ``.metadata`` and one shard file.
        tensors: ``fqn -> (global shape, [(chunk offsets, chunk tensor), ...])``.
            Every listed chunk gets real storage.
        nontensor: Names stored as ``BytesStorageMetadata`` (the ``_extra_state``
            blobs that dominate real checkpoint metadata).
        orphan_chunks: ``fqn -> [(offsets, sizes), ...]`` declared in the tensor
            metadata but deliberately given *no* storage record.
        offsetless_blobs: Number of ``MetadataIndex(offset=None)`` storage entries
            to add; these model byte-blob records and must never enter the
            reader's offset index.
    """
    torch = pytest.importorskip("torch")
    from torch.distributed.checkpoint import filesystem as dcfs
    from torch.distributed.checkpoint import metadata as dcmeta

    shard_name = "shard_0.distcp"
    # `_StorageInfo` is private and has moved between torch releases: it lives in
    # `checkpoint.filesystem` on 2.13 and in `checkpoint.metadata` on some earlier ones.
    # Search both rather than pinning one, and fail with a message that says the API moved
    # instead of a bare AttributeError — the module under test never names this class, so a
    # miss here is a fixture problem and should read as one.
    storage_info_cls = None
    for mod in (dcfs, dcmeta):
        storage_info_cls = getattr(mod, "_StorageInfo", None)
        if storage_info_cls is not None:
            break
    if storage_info_cls is None:  # pragma: no cover - torch API drift
        pytest.skip("torch's private _StorageInfo moved again; fixture needs updating")

    state_dict_metadata: dict[str, Any] = {}
    storage_data: dict[Any, Any] = {}
    parts: list[bytes] = []
    cursor = 0

    for fqn, (shape, chunks) in tensors.items():
        chunk_mds: list[Any] = []
        dtype = chunks[0][1].dtype if chunks else torch.float32
        for offsets, tensor in chunks:
            data = _archive_bytes(tensor)
            parts.append(data)
            index = dcmeta.MetadataIndex(fqn=fqn, offset=torch.Size(offsets))
            storage_data[index] = storage_info_cls(
                relative_path=shard_name, offset=cursor, length=len(data)
            )
            cursor += len(data)
            chunk_mds.append(
                dcmeta.ChunkStorageMetadata(
                    offsets=torch.Size(offsets),
                    sizes=torch.Size(tuple(tensor.shape)),
                )
            )
        for offsets, sizes in (orphan_chunks or {}).get(fqn, []):
            chunk_mds.append(
                dcmeta.ChunkStorageMetadata(offsets=torch.Size(offsets), sizes=torch.Size(sizes))
            )
        state_dict_metadata[fqn] = dcmeta.TensorStorageMetadata(
            size=torch.Size(shape),
            properties=dcmeta.TensorProperties(dtype=dtype),
            chunks=chunk_mds,
        )

    for name in nontensor or []:
        state_dict_metadata[name] = dcmeta.BytesStorageMetadata()

    for i in range(offsetless_blobs):
        index = dcmeta.MetadataIndex(fqn=f"_blob_{i}", offset=None)
        storage_data[index] = storage_info_cls(relative_path=shard_name, offset=0, length=0)

    metadata = dcmeta.Metadata(state_dict_metadata=state_dict_metadata, storage_data=storage_data)
    (root / DCP_METADATA_FILENAME).write_bytes(pickle.dumps(metadata))
    (root / shard_name).write_bytes(b"".join(parts))


def _write_minimal_safetensors(path: Path, name: str = "w") -> None:
    """Write a valid single-tensor safetensors file with no third-party help.

    The format is an 8-byte little-endian header length, a JSON header, then the
    raw payload — nothing here needs the ``safetensors`` package to *write*.
    """
    header = json.dumps({name: {"dtype": "F32", "shape": [2], "data_offsets": [0, 8]}}).encode(
        "utf-8"
    )
    payload = struct.pack("<2f", 1.25, -0.5)
    path.write_bytes(struct.pack("<Q", len(header)) + header + payload)


TWO_CHUNK_TENSOR_KEY = "model.layers.0.mlp.weight"


def _write_two_chunk(tmp_path: Path, torch_mod: Any) -> Any:
    """A healthy 4x6 tensor tiled into two row chunks. Returns the full tensor."""
    full = torch_mod.arange(24, dtype=torch_mod.float32).reshape(4, 6)
    _write_dcp_checkpoint(
        tmp_path,
        tensors={
            TWO_CHUNK_TENSOR_KEY: (
                (4, 6),
                [((0, 0), full[:2].contiguous()), ((2, 0), full[2:].contiguous())],
            )
        },
    )
    return full


# ---------------------------------------------------------------------------
# Coverage is a returned fact — the healthy control every refusal is measured
# against. If the module ever started refusing everything, these go red first.
# ---------------------------------------------------------------------------


def test_read_full_reports_coverage_as_facts(tmp_path: Path, torch_mod: Any) -> None:
    """A fully tiled tensor must read back complete, counted and byte-exact."""
    full = _write_two_chunk(tmp_path, torch_mod)
    reader = DcpReader(tmp_path)

    result = reader.read_full(TWO_CHUNK_TENSOR_KEY)

    assert result.complete
    assert result.chunks_read == 2
    assert result.elements_covered == 24
    assert result.elements_expected == 24
    expected_bytes = len(_archive_bytes(full[:2].contiguous())) + len(
        _archive_bytes(full[2:].contiguous())
    )
    assert result.bytes_read == expected_bytes
    assert result.tensor.dtype == torch_mod.float32
    # The must-pass control: the bytes the reader assembled are the bytes that
    # were trained, not an empty or zero-filled stand-in.
    assert torch_mod.equal(result.tensor, full)


def test_read_box_assembles_partial_box_across_chunks(tmp_path: Path, torch_mod: Any) -> None:
    """A sub-box spanning two chunks assembles exact values with exact counts."""
    full = _write_two_chunk(tmp_path, torch_mod)
    reader = DcpReader(tmp_path)

    result = reader.read_box(TWO_CHUNK_TENSOR_KEY, (1, 0), (3, 6))

    assert result.complete
    assert result.chunks_read == 2
    assert result.elements_covered == 12
    assert result.elements_expected == 12
    # Rows 1 and 2 of the original: proves the src/dst slicing reads the right
    # bytes from each chunk rather than whole-chunk copies.
    assert torch_mod.equal(result.tensor, full[1:3])


# ---------------------------------------------------------------------------
# Incomplete coverage must raise, never zero-fill — the read_full incident.
# ---------------------------------------------------------------------------


def test_read_full_raises_on_gap_instead_of_zero_fill(tmp_path: Path, torch_mod: Any) -> None:
    """Rows [2:4) of 4 stored with a hole at [2:3): refuse, with exact counts.

    This is the probe incident reproduced: a reader that zero-fills the missing
    row returns plausible data and a quiet success. The refusal is the feature.
    """
    full = torch_mod.arange(24, dtype=torch_mod.float32).reshape(4, 6)
    _write_dcp_checkpoint(
        tmp_path,
        tensors={
            TWO_CHUNK_TENSOR_KEY: (
                (4, 6),
                [((0, 0), full[:2].contiguous()), ((3, 0), full[3:].contiguous())],
            )
        },
    )
    reader = DcpReader(tmp_path)

    with pytest.raises(IncompleteCoverageError) as excinfo:
        reader.read_full(TWO_CHUNK_TENSOR_KEY)

    err = excinfo.value
    assert err.elements_covered == 18  # rows 0-1 (12) + row 3 (6)
    assert err.elements_expected == 24
    assert err.key == TWO_CHUNK_TENSOR_KEY  # an operator must not grep for the fqn


def test_read_box_raises_when_sub_box_is_partially_covered(tmp_path: Path, torch_mod: Any) -> None:
    """A requested box larger than the stored coverage refuses with exact counts."""
    full = torch_mod.arange(24, dtype=torch_mod.float32).reshape(4, 6)
    _write_dcp_checkpoint(
        tmp_path,
        tensors={
            TWO_CHUNK_TENSOR_KEY: (
                (4, 6),
                [((0, 0), full[:2].contiguous()), ((3, 0), full[3:].contiguous())],
            )
        },
    )
    reader = DcpReader(tmp_path)

    with pytest.raises(IncompleteCoverageError) as excinfo:
        reader.read_box(TWO_CHUNK_TENSOR_KEY, (1, 0), (4, 6))

    err = excinfo.value
    assert err.elements_covered == 12  # overlap rows [1,2) and [3,4): 6 + 6
    assert err.elements_expected == 18  # the 3x6 box that was asked for


def test_overlapping_chunks_refuse_even_when_volumes_sum_correctly(
    tmp_path: Path, torch_mod: Any
) -> None:
    """The test that separates a volume check from a coverage check.

    Chunk A covers rows [0,3), chunk B covers row [1,2). Their volumes sum to
    exactly the tensor's volume, yet they double-store row 1 and leave rows
    [3:4) unstored. A reader that only compared volumes would pass this
    corrupted layout; the pairwise written-region overlap guard must fire first.
    """
    full = torch_mod.arange(24, dtype=torch_mod.float32).reshape(4, 6)
    _write_dcp_checkpoint(
        tmp_path,
        tensors={
            TWO_CHUNK_TENSOR_KEY: (
                (4, 6),
                [((0, 0), full[:3].contiguous()), ((1, 0), full[1:2].contiguous())],
            )
        },
    )
    reader = DcpReader(tmp_path)

    with pytest.raises(IncompleteCoverageError) as excinfo:
        reader.read_full(TWO_CHUNK_TENSOR_KEY)

    err = excinfo.value
    assert "overlap" in str(err)
    # The count the volume check would have seen: exactly full. If the overlap
    # guard were deleted, this layout would sail through covered == expected.
    assert err.elements_covered == 24
    assert err.elements_expected == 24


def test_metadata_chunk_without_storage_record_is_incomplete(
    tmp_path: Path, torch_mod: Any
) -> None:
    """A tensor metadata chunk with no matching storage_data entry must raise.

    Metadata says rows [2,4) exist; the offset index disagrees. Treating that as
    "just not covered" double-counts nothing — but it must be surfaced at the
    chunk read where the data would otherwise come back as zeros.
    """
    full = torch_mod.arange(24, dtype=torch_mod.float32).reshape(4, 6)
    _write_dcp_checkpoint(
        tmp_path,
        tensors={TWO_CHUNK_TENSOR_KEY: ((4, 6), [((0, 0), full[:2].contiguous())])},
        orphan_chunks={TWO_CHUNK_TENSOR_KEY: [((2, 0), (2, 6))]},
    )
    reader = DcpReader(tmp_path)

    with pytest.raises(IncompleteCoverageError) as excinfo:
        reader.read_full(TWO_CHUNK_TENSOR_KEY)

    err = excinfo.value
    assert "no storage record" in str(err)
    assert err.elements_covered == 12  # only the first chunk was ever read
    assert err.elements_expected == 24


# ---------------------------------------------------------------------------
# Vacuous sources are refused at construction — empty key sets cannot become
# `all([])`. The refusal's *reason* is asserted, not just that some error flew.
# ---------------------------------------------------------------------------


def test_dcp_reader_refuses_zero_tensor_metadata(tmp_path: Path, torch_mod: Any) -> None:
    """Metadata holding only ``_extra_state`` blobs is not a checkpoint.

    The audited estate's real ratio was 928 tensor entries to 8,042 byte blobs;
    zero tensors is the extreme of that, and the reader must name the
    vacuous-truth failure mode it is preventing.
    """
    _write_dcp_checkpoint(tmp_path, tensors={}, nontensor=["_extra_state.optimizer"])
    assert torch_mod is not None  # construction of the fixture needed torch already

    with pytest.raises(CheckpointFormatError) as excinfo:
        DcpReader(tmp_path)

    message = str(excinfo.value)
    assert "0 tensor entries" in message
    assert "vacuous" in message


def test_safetensors_reader_refuses_empty_weight_map(tmp_path: Path) -> None:
    """An index whose weight_map is {} must be refused as vacuous, not opened."""
    pytest.importorskip("safetensors")
    (tmp_path / SAFETENSORS_INDEX_FILENAME).write_text(
        json.dumps({"weight_map": {}}), encoding="utf-8"
    )

    with pytest.raises(CheckpointFormatError) as excinfo:
        SafetensorsReader(tmp_path)

    message = str(excinfo.value)
    assert "declares 0 tensors" in message
    assert "vacuous" in message


# ---------------------------------------------------------------------------
# state_dict_metadata filtering: tensors are exposed, byte blobs are visible
# but never counted as tensors, and offset-less blob records never enter the
# offset index.
# ---------------------------------------------------------------------------


def test_dcp_metadata_filters_nontensor_entries_and_offsetless_blobs(
    tmp_path: Path, torch_mod: Any
) -> None:
    """Tensor keys come only from TensorStorageMetadata; the rest stays visible."""
    weight = torch_mod.arange(6, dtype=torch_mod.float32).reshape(2, 3)
    _write_dcp_checkpoint(
        tmp_path,
        tensors={TWO_CHUNK_TENSOR_KEY: ((2, 3), [((0, 0), weight)])},
        nontensor=["_extra_state.optimizer"],
        offsetless_blobs=1,
    )

    reader = DcpReader(tmp_path)

    assert reader.tensor_keys() == (TWO_CHUNK_TENSOR_KEY,)
    assert "_extra_state.optimizer" not in reader.tensor_keys()
    # The 8,042-of-8,970 lesson: non-tensor entries are surfaced, not dropped.
    assert reader.nontensor_keys() == ("_extra_state.optimizer",)
    # The offset-less blob was excluded from the offset index but still counted.
    # (No public counter exists; the skip count is the only place the fact lives.)
    assert reader._skipped_blobs == 1  # noqa: SLF001 — the count IS the assertion
    # Must-pass control: real tensors survived the filtering and still read
    # back bit-exact. Without this, a filter that dropped *everything* would
    # pass the assertions above (they only check names).
    result = reader.read_full(TWO_CHUNK_TENSOR_KEY)
    assert result.complete
    assert torch_mod.equal(result.tensor, weight)


# ---------------------------------------------------------------------------
# open_weights layout sniffing: one format opens, ambiguity and unverifiable
# formats fail loudly with both culprits named.
# ---------------------------------------------------------------------------


def test_open_weights_ambiguous_layout_names_both_formats(tmp_path: Path) -> None:
    """DCP metadata + safetensors shards in one dir must refuse, naming both.

    Content does not matter here: the ambiguity check runs before either reader
    parses anything, which is exactly why this test needs no torch.
    """
    (tmp_path / DCP_METADATA_FILENAME).write_bytes(b"not a pickle")
    (tmp_path / "model-00001-of-00001.safetensors").write_bytes(b"junk")

    with pytest.raises(CheckpointFormatError) as excinfo:
        open_weights(tmp_path)

    message = str(excinfo.value)
    assert "ambiguous" in message
    assert "DCP" in message
    assert "safetensors" in message


def test_open_weights_detects_dcp_layout(tmp_path: Path, torch_mod: Any) -> None:
    """A directory with .metadata opens as a DcpReader with its keys intact."""
    _write_two_chunk(tmp_path, torch_mod)

    source = open_weights(tmp_path)

    assert isinstance(source, DcpReader)
    assert source.tensor_keys() == (TWO_CHUNK_TENSOR_KEY,)


def test_open_weights_detects_safetensors_layout(tmp_path: Path) -> None:
    """A shard without an index opens via key enumeration and reads real values."""
    pytest.importorskip("safetensors")
    pytest.importorskip("torch")  # get_tensor with framework="pt" needs it
    shard = tmp_path / "model.safetensors"
    _write_minimal_safetensors(shard)

    source = open_weights(tmp_path)
    assert isinstance(source, SafetensorsReader)
    assert source.tensor_keys() == ("w",)
    assert source.shape("w") == (2,)
    # Contiguous single-shard storage means one whole-tensor chunk, always.
    assert source.chunks("w") == (Chunk(offsets=(0,), sizes=(2,)),)

    # Positive control: the sniffed reader reads back the exact values written,
    # with complete coverage — the safetensors analogue of the DCP control.
    result = source.read_full("w")
    assert result.complete
    assert result.elements_covered == 2
    assert result.tensor.tolist() == [1.25, -0.5]
    source.close()

    # The single-file entry point sniffs identically.
    file_source = open_weights(shard)
    assert isinstance(file_source, SafetensorsReader)
    assert file_source.tensor_keys() == ("w",)
    file_source.close()


def test_open_weights_refuses_bin_file(tmp_path: Path) -> None:
    """A lone .bin pickle is refused with the *reason*: it cannot be verified."""
    target = tmp_path / "model.bin"
    target.write_bytes(b"junk")

    with pytest.raises(CheckpointFormatError) as excinfo:
        open_weights(target)

    message = str(excinfo.value)
    assert "plain torch pickle" in message
    assert "cannot be partially read" in message


def test_open_weights_refuses_directory_of_bin_pickles(tmp_path: Path) -> None:
    """A directory holding only .bin files refuses rather than falling silent."""
    (tmp_path / "pytorch_model.bin").write_bytes(b"junk")

    with pytest.raises(CheckpointFormatError) as excinfo:
        open_weights(tmp_path)

    assert "only plain .bin pickles" in str(excinfo.value)


def test_open_weights_unrecognized_layout_says_what_it_found(tmp_path: Path) -> None:
    """Nothing recognizable: the error must list the directory, so nobody has to
    re-run the sniffer by hand on a 51 GB artifact to learn what was there."""
    (tmp_path / "notes.txt").write_text("not a checkpoint", encoding="utf-8")

    with pytest.raises(CheckpointFormatError) as excinfo:
        open_weights(tmp_path)

    message = str(excinfo.value)
    assert "no recognizable checkpoint layout" in message
    assert "notes.txt" in message


# ---------------------------------------------------------------------------
# Defect watch: a zero-volume box currently returns a vacuous ReadResult.
# ---------------------------------------------------------------------------


def test_read_box_refuses_zero_volume_box(tmp_path: Path, torch_mod: Any) -> None:
    """Requesting lo == hi must not report 'complete' on nothing examined."""
    _write_two_chunk(tmp_path, torch_mod)
    reader = DcpReader(tmp_path)

    with pytest.raises(ValueError):
        reader.read_box(TWO_CHUNK_TENSOR_KEY, (1, 0), (1, 6))
