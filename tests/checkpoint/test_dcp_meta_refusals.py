"""Coverage of dcp_meta's refusal matrix and measurement edges.

Why this file exists
--------------------
per the suite-wide coverage map (0c46636): 37 of dcp_meta's lines execute in
no test -- the whole ``_sniff`` refusal matrix, most of ``_read_st_header``'s
malformed-shard arms, the index/weight_map disagreement arms, DCP storage-id
edge cases (orphan chunk, zero-chunk, blob records), and both ``load_manifest``
fall-through/strict arms. These lines decide whether a garbage artifact is
REFUSED readably or mis-measured silently, so leaving them dark is the
vacuous-truth hazard one level down: the refusal text itself could rot.

The DCP fixtures reuse ``_write_dcp_checkpoint`` (imported from the sibling
module the same way ``_dcp_fakes`` is shared) because re-writing a pickle
fabricator here would fork two incidents' worth of blob/vacuity rules.
"""

from __future__ import annotations

import json
import os
import struct
from pathlib import Path
from typing import Any

import pytest

from foundationscale.checkpoint import dcp_meta
from foundationscale.checkpoint.dcp import CheckpointFormatError
from foundationscale.checkpoint.dcp_meta import load_manifest, read_metadata

# --- helpers -----------------------------------------------------------------


def _write_shard(path: Path, entries: dict[str, object]) -> Path:
    """Minimal valid safetensors shard: header JSON + enough zero payload."""
    header: dict[str, object] = {}
    cursor = 0
    for key, entry in entries.items():
        if key == "__metadata__":
            header[key] = entry
            continue
        dtype, shape = entry  # type: ignore[misc]
        numel = 1
        for d in shape:  # type: ignore[union-attr]
            numel *= int(d)
        nbytes = numel * {"F32": 4, "BF16": 2, "I64": 8}[dtype]  # type: ignore[index]
        header[key] = {
            "dtype": dtype,
            "shape": list(shape),  # type: ignore[union-attr]
            "data_offsets": [cursor, cursor + nbytes],
        }
        cursor += nbytes
    blob = json.dumps(header).encode("utf-8")
    path.write_bytes(struct.pack("<Q", len(blob)) + blob + b"\x00" * cursor)
    return path


def _raw_shard(path: Path, header_json: bytes, payload: bytes = b"") -> Path:
    path.write_bytes(struct.pack("<Q", len(header_json)) + header_json + payload)
    return path


# --- _sniff / read_metadata dispatch (134-137, 142, 153, 167, 444-445) -------


def test_sniff_accepts_a_lone_safetensors_file_and_reads_it(tmp_path: Path) -> None:
    """A single .safetensors FILE classifies and reads without a directory scan."""
    shard = _write_shard(tmp_path / "model.safetensors", {"w": ("F32", (2, 2))})
    meta = read_metadata(str(shard))
    assert meta.format == "safetensors"
    assert set(meta.tensors) == {"w"}
    assert meta.tensors["w"].storage_id == f"model.safetensors:0:{16}"


def test_sniff_refuses_a_bin_file_with_the_reason_named(tmp_path: Path) -> None:
    """A .bin pickle cannot be verified chunk-wise; the message says so."""
    target = tmp_path / "model.bin"
    target.write_bytes(b"pickle-ish")
    with pytest.raises(CheckpointFormatError, match="plain torch pickle"):
        read_metadata(str(target))


def test_sniff_refuses_an_unrecognized_file_name(tmp_path: Path) -> None:
    target = tmp_path / "model.ckpt"
    target.write_bytes(b"???")
    with pytest.raises(CheckpointFormatError, match="unrecognized checkpoint file"):
        read_metadata(str(target))


def test_sniff_refuses_a_directory_holding_both_formats(tmp_path: Path) -> None:
    """DCP + safetensors in one dir is ambiguous, never guessed."""
    (tmp_path / ".metadata").write_bytes(b"pickle would do")
    _write_shard(tmp_path / "model.safetensors", {"w": ("F32", (1,))})
    with pytest.raises(CheckpointFormatError, match="ambiguous"):
        read_metadata(str(tmp_path))


def test_sniff_refuses_a_directory_of_only_bin_pickles(tmp_path: Path) -> None:
    (tmp_path / "pytorch_model.bin").write_bytes(b"x")
    with pytest.raises(CheckpointFormatError, match="plain .bin pickles"):
        read_metadata(str(tmp_path))


# --- _read_st_header malformed arms ------------------------------------------


def test_symlink_shard_is_refused_before_any_stat(tmp_path: Path) -> None:
    """#343: a link is not a saved shard; the target's bytes would be credited here."""
    real = _write_shard(tmp_path / "real.safetensors", {"w": ("F32", (1,))})
    link = tmp_path / "model.safetensors"
    link.symlink_to(real)
    with pytest.raises(CheckpointFormatError, match="symbolic link"):
        read_metadata(str(link))


def test_shard_too_short_for_the_header_prefix_is_refused(tmp_path: Path) -> None:
    target = tmp_path / "model.safetensors"
    target.write_bytes(b"\x01\x02")
    with pytest.raises(CheckpointFormatError, match="too short for the 8-byte"):
        read_metadata(str(target))


def test_declared_header_length_must_fit_the_shard(tmp_path: Path) -> None:
    target = tmp_path / "model.safetensors"
    target.write_bytes(struct.pack("<Q", 10**9) + b"{}")
    with pytest.raises(CheckpointFormatError, match="does not fit shard size"):
        read_metadata(str(target))


def test_header_json_must_decode(tmp_path: Path) -> None:
    target = _raw_shard(tmp_path / "model.safetensors", b"\xffnot json")
    with pytest.raises(CheckpointFormatError, match="not valid JSON"):
        read_metadata(str(target))


def test_header_must_be_a_json_object(tmp_path: Path) -> None:
    target = _raw_shard(tmp_path / "model.safetensors", b"[1, 2]")
    with pytest.raises(CheckpointFormatError, match="not a JSON object"):
        read_metadata(str(target))


def test_header_entry_must_be_an_object(tmp_path: Path) -> None:
    target = _raw_shard(tmp_path / "model.safetensors", json.dumps({"w": 7}).encode())
    with pytest.raises(CheckpointFormatError, match="is not an object"):
        read_metadata(str(target))


def test_unknown_dtype_name_is_refused(tmp_path: Path) -> None:
    target = _raw_shard(
        tmp_path / "model.safetensors",
        json.dumps({"w": {"dtype": "F999", "shape": [1], "data_offsets": [0, 4]}}).encode(),
        b"\x00" * 4,
    )
    with pytest.raises(CheckpointFormatError, match="unsupported safetensors dtype"):
        read_metadata(str(target))


def test_entry_without_shape_or_offsets_is_refused(tmp_path: Path) -> None:
    target = _raw_shard(
        tmp_path / "model.safetensors",
        json.dumps({"w": {"dtype": "F32", "shape": [1]}}).encode(),
    )
    with pytest.raises(CheckpointFormatError, match="shape/data_offsets"):
        read_metadata(str(target))


def test_non_integer_shape_is_refused(tmp_path: Path) -> None:
    target = _raw_shard(
        tmp_path / "model.safetensors",
        json.dumps({"w": {"dtype": "F32", "shape": ["x"], "data_offsets": [0, 4]}}).encode(),
        b"\x00" * 4,
    )
    with pytest.raises(CheckpointFormatError, match="non-integer"):
        read_metadata(str(target))


def test_backwards_data_offsets_are_refused(tmp_path: Path) -> None:
    target = _raw_shard(
        tmp_path / "model.safetensors",
        json.dumps({"w": {"dtype": "F32", "shape": [1], "data_offsets": [8, 2]}}).encode(),
        b"\x00" * 10,
    )
    with pytest.raises(CheckpointFormatError, match="run backwards"):
        read_metadata(str(target))


def test_a_short_read_is_named_not_silently_accepted(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Truncation between stat and read surfaces as a named short read."""
    target = tmp_path / "model.safetensors"
    # 8-byte prefix declares 50 header bytes; only 10 are actually on disk,
    # while lstat is made to claim the file is complete (the torn-write shape).
    target.write_bytes(struct.pack("<Q", 50) + b'{"w": {"dt')
    real_stat = os.lstat

    class _LiedStat:
        st_size = 10**9

    monkeypatch.setattr(dcp_meta.os, "lstat", lambda p: _LiedStat())
    # lstat lied that the shard is huge: the header read must not trust it.
    with pytest.raises(CheckpointFormatError, match="does not fit shard size|short read"):
        read_metadata(str(target))
    monkeypatch.undo()
    assert real_stat is os.lstat


# --- safetensors directory arms (index, weight_map, zero-key) ----------------


def test_zero_key_shard_is_refused_as_a_format_error(tmp_path: Path) -> None:
    """An export declaring 0 tensors is the vacuous-truth machine; refused."""
    _write_shard(tmp_path / "model.safetensors", {"__metadata__": {"note": "x"}})
    with pytest.raises(CheckpointFormatError, match="0 tensors"):
        read_metadata(str(tmp_path))


def test_malformed_index_json_is_refused(tmp_path: Path) -> None:
    _write_shard(tmp_path / "model.safetensors", {"w": ("F32", (1,))})
    (tmp_path / "model.safetensors.index.json").write_bytes(b"{oops")
    with pytest.raises(CheckpointFormatError, match="malformed"):
        read_metadata(str(tmp_path))


def test_index_whose_weight_map_is_not_an_object_is_refused(tmp_path: Path) -> None:
    _write_shard(tmp_path / "model.safetensors", {"w": ("F32", (1,))})
    (tmp_path / "model.safetensors.index.json").write_text(json.dumps({"weight_map": ["w"]}))
    with pytest.raises(CheckpointFormatError, match="not an object"):
        read_metadata(str(tmp_path))


def test_index_pointing_at_a_missing_shard_is_refused(tmp_path: Path) -> None:
    _write_shard(tmp_path / "model.safetensors", {"w": ("F32", (1,))})
    (tmp_path / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {"w": "ghost.safetensors"}})
    )
    with pytest.raises(CheckpointFormatError, match="missing shard"):
        read_metadata(str(tmp_path))


def test_index_naming_a_key_the_shard_does_not_hold_is_refused(tmp_path: Path) -> None:
    _write_shard(tmp_path / "model.safetensors", {"w": ("F32", (1,))})
    (tmp_path / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {"v": "model.safetensors"}})
    )
    with pytest.raises(CheckpointFormatError, match="does not contain"):
        read_metadata(str(tmp_path))


def test_index_pointing_at_a_symlinked_shard_is_refused(tmp_path: Path) -> None:
    real = _write_shard(tmp_path / "real.safetensors", {"w": ("F32", (1,))})
    link = tmp_path / "linked.safetensors"
    link.symlink_to(real)
    (tmp_path / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {"w": "linked.safetensors"}})
    )
    with pytest.raises(CheckpointFormatError, match="symbolic link"):
        read_metadata(str(tmp_path))


def test_one_tensor_in_two_shards_is_corrupt_not_ambiguous(tmp_path: Path) -> None:
    """Duplicate key = two byte spans for one name; refused like the reader's."""
    _write_shard(tmp_path / "a.safetensors", {"w": ("F32", (1,))})
    _write_shard(tmp_path / "b.safetensors", {"w": ("F32", (1,))})
    with pytest.raises(CheckpointFormatError, match="two shards"):
        read_metadata(str(tmp_path))


# --- DCP storage-id edges (229 orphan chunk, 233 zero-chunk, 259 blobs) ------


def _dcp_fabricator(tmp_path: Path) -> Any:
    from test_dcp_coverage import _write_dcp_checkpoint

    return _write_dcp_checkpoint


def test_dcp_orphan_chunk_yields_no_storage_id_not_an_invented_one(tmp_path: Path) -> None:
    """A declared chunk with no storage record: identity is honestly unknowable."""
    _write = _dcp_fabricator(tmp_path)
    _write(
        tmp_path,
        tensors={"w": ((2,), [((0,), __import__("torch").zeros(1))])},
        orphan_chunks={"w": [((1,), (1,))]},
    )
    meta = read_metadata(str(tmp_path))
    assert meta.tensors["w"].storage_id is None


def test_dcp_blob_records_get_byte_identity_and_flag(tmp_path: Path) -> None:
    """Offsetless byte blobs enter the identity pass, flagged extra-state."""
    _write = _dcp_fabricator(tmp_path)
    _write(
        tmp_path,
        tensors={"w": ((1,), [((0,), __import__("torch").zeros(1))])},
        nontensor=["_extra_state_blob"],
        offsetless_blobs=1,
    )
    meta = read_metadata(str(tmp_path))
    blob = meta.tensors["_extra_state_blob"]
    assert blob.is_extra_state is True
    assert blob.dtype == "bytes"


# --- load_manifest: shared-name fall-through and reserved-name strictness ----


def test_shared_name_with_non_manifest_json_is_skipped_to_absence(
    tmp_path: Path,
) -> None:
    """A foreign manifest.json (export tooling's) is not ours: skipped, not raised."""
    (tmp_path / "manifest.json").write_text(json.dumps({"tier": 1}))
    assert load_manifest(tmp_path) is None


def test_reserved_name_with_valid_shape_but_failed_invariants_raises(
    tmp_path: Path,
) -> None:
    """run_manifest.json carrying the 4 keys but failing validation is corrupt."""
    (tmp_path / "run_manifest.json").write_text(
        json.dumps({"run_id": 1, "code": "x", "environment": "y", "topology": "z"})
    )
    with pytest.raises(CheckpointFormatError, match="fails validation|corrupt"):
        load_manifest(tmp_path)


# --- tiny closers ------------------------------------------------------------


def test_sniff_refuses_a_path_that_does_not_exist(tmp_path: Path) -> None:
    with pytest.raises(CheckpointFormatError, match="does not exist"):
        read_metadata(str(tmp_path / "nowhere"))


def test_dcp_zero_chunk_tensor_has_no_byte_identity(tmp_path: Path) -> None:
    """No chunks on disk means no bytes, hence no identity -- None, not a hash."""
    _write = _dcp_fabricator(tmp_path)
    _write(tmp_path, tensors={"w": ((0,), [])})
    meta = read_metadata(str(tmp_path))
    assert meta.tensors["w"].storage_id is None


def test_lstat_failure_names_itself(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A shard whose size cannot be established is a format error, not a guess."""
    shard = _write_shard(tmp_path / "model.safetensors", {"w": ("F32", (1,))})

    def _boom(p: Any) -> Any:
        raise OSError("simulated stat failure")

    monkeypatch.setattr(dcp_meta.os, "lstat", _boom)
    with pytest.raises(CheckpointFormatError, match="cannot stat shard"):
        read_metadata(str(shard))


def test_header_read_failure_names_itself(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """An unreadable shard is a refusal, not a 0-byte misread."""
    shard = _write_shard(tmp_path / "model.safetensors", {"w": ("F32", (1,))})

    class _Unopenable:
        def __call__(self, *a: Any, **k: Any) -> Any:
            raise OSError("simulated open failure")

    monkeypatch.setattr(dcp_meta.Path, "open", _Unopenable())
    with pytest.raises(CheckpointFormatError, match="cannot read shard header"):
        read_metadata(str(shard))
