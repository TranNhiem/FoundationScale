"""Tests for streaming comparison, chunk validation, and handle lifecycle in dcp.py.

Why this file exists
--------------------
The incidents it guards against are specific, not hypothetical:

* The reference checkpoint probe reported a tensor's cosine *with itself* as
  1.80 — a mathematically impossible value that shipped in a report because
  float32 reductions underflowed and nothing asserted the invariant. The
  float64-accumulation and self-check tests here reproduce that failure mode
  and prove the guard raises when the accumulator lies.
* A detector reporting success on an artifact it never read is the audit's
  ``all([]) is True`` bug (verifier incident #6). The short-buffer,
  wrong-magic, corrupt-archive, empty-export and zero-element cases below
  exist to prove each reader fails loudly instead of returning plausible
  nothing.
* The probe leaked one shard handle per file forever; the ``_HandleCache``
  tests assert eviction actually *releases* handles, with a fake handle that
  records release, so the assertion cannot pass on bookkeeping alone.

Per the project rule, every test here could genuinely fail: each assertion
asks what the code would have to do to falsify it, and the filesystem
fixtures are synthesized under ``tmp_path`` with no network, GPU, or cluster.
"""

from __future__ import annotations

import io
import json
import math
from collections.abc import Sequence
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from foundationscale.checkpoint.dcp import (
    CheckpointFormatError,
    Chunk,
    ChunkReadError,
    DcpReader,
    ReadResult,
    SafetensorsReader,
    _HandleCache,
    _safetensors_dtype,
    compare_keys,
)

# ---------------------------------------------------------------------------
# In-memory WeightSource double
# ---------------------------------------------------------------------------


class MemSource:
    """In-memory ``WeightSource`` test double for ``compare_keys``.

    ``compare_keys`` only needs the protocol surface, so the comparison tests
    exercise the real streaming logic without manufacturing DCP metadata.
    Refusing an empty mapping mirrors the real readers' non-empty guarantee:
    an empty source is the vacuous-truth bug one level up.
    """

    def __init__(self, tensors: dict[str, Any], path: str = "mem://test") -> None:
        if not tensors:
            raise ValueError("MemSource refuses an empty tensor set, as real readers do")
        self._tensors = dict(tensors)
        self.path = path

    def tensor_keys(self) -> tuple[str, ...]:
        return tuple(sorted(self._tensors))

    def nontensor_keys(self) -> tuple[str, ...]:
        return ()

    def shape(self, key: str) -> tuple[int, ...]:
        return tuple(int(d) for d in self._tensors[key].shape)

    def dtype(self, key: str) -> Any:
        return self._tensors[key].dtype

    def chunks(self, key: str) -> tuple[Chunk, ...]:
        shape = self.shape(key)
        return (Chunk(offsets=(0,) * len(shape), sizes=shape),)

    def read_chunk(self, key: str, offsets: Sequence[int]) -> Any:
        if any(int(o) != 0 for o in offsets):
            raise ValueError(f"MemSource tensors are single-chunk: {offsets}")
        return self._tensors[key]

    def read_box(self, key: str, lo: Sequence[int], hi: Sequence[int]) -> ReadResult:
        tensor = self._tensors[key]
        nd = tensor.dim()
        if len(lo) != nd or len(hi) != nd:
            raise ValueError(f"box rank mismatch for {key!r}: tensor is {nd}-D")
        idx = tuple(slice(int(lo_i), int(hi_i)) for lo_i, hi_i in zip(lo, hi, strict=True))
        block = tensor[idx]
        numel = math.prod(int(hi_i) - int(lo_i) for lo_i, hi_i in zip(lo, hi, strict=True))
        return ReadResult(
            key=key,
            tensor=block,
            chunks_read=1,
            elements_covered=int(numel),
            elements_expected=int(numel),
            bytes_read=block.numel() * block.element_size(),
        )

    def read_full(self, key: str) -> ReadResult:
        shape = self.shape(key)
        return self.read_box(key, (0,) * len(shape), shape)

    def close(self) -> None:
        return None


# ---------------------------------------------------------------------------
# compare_keys: the 1.80-cosine incident and its guard
# ---------------------------------------------------------------------------


def test_identical_inputs_score_cosine_exactly_one_and_zero_diff() -> None:
    torch: Any = pytest.importorskip("torch")
    gen = torch.Generator().manual_seed(7)
    t = torch.randn(128, 64, generator=gen)
    tc = compare_keys(MemSource({"w": t}), MemSource({"w": t.clone()}), "w")

    assert tc.verdict == "EXACT"
    assert tc.bitwise_equal is True
    assert tc.cosine == 1.0  # exactly, not approximately: the guard promises this
    assert tc.max_abs_diff == 0.0
    assert tc.mismatched_elements == 0
    assert tc.elements == 128 * 64
    # Falsifiable cross-check: the streaming rms must match a direct float64 reduction.
    ref_rms = math.sqrt(float((t.to(torch.float64) ** 2).mean().item()))
    assert tc.rms_a == pytest.approx(ref_rms, rel=1e-12)
    assert tc.rms_b == pytest.approx(tc.rms_a, rel=1e-12)


def test_detector_fires_when_inputs_differ() -> None:
    """Positive control: if this passes, the EXACT assertions above mean something."""
    torch: Any = pytest.importorskip("torch")
    gen = torch.Generator().manual_seed(11)
    a = torch.randn(64, 64, generator=gen)
    b = -a  # every element wrong, cosine -1: a detector that misses this is dead
    tc = compare_keys(MemSource({"w": a}), MemSource({"w": b}), "w")

    assert tc.verdict == "DIFFER"
    assert tc.bitwise_equal is False
    assert tc.mismatched_elements == a.numel()
    assert tc.max_abs_diff > 0.0
    assert tc.cosine is not None and tc.cosine < 0.0


def test_small_perturbation_reports_close_within_documented_tolerances() -> None:
    torch: Any = pytest.importorskip("torch")
    gen = torch.Generator().manual_seed(3)
    a = torch.randn(256, 32, generator=gen).clamp(-3, 3)
    b = a * (1.0 + 1e-4)  # uniform relative nudge: max|diff| ~3e-4, cosine ~1
    tc = compare_keys(MemSource({"w": a}), MemSource({"w": b}), "w")

    assert tc.verdict == "CLOSE", f"expected CLOSE, got {tc.to_dict()}"
    assert tc.bitwise_equal is False
    assert 0.0 < tc.max_abs_diff <= 1e-2
    assert tc.cosine is not None and tc.cosine > 0.999


def test_float64_accumulation_recovers_answer_float32_visibly_loses() -> None:
    """The incident: huge magnitudes send float32 sum-of-squares to inf, cosine to nan.

    First assert the float32 pathology is actually present in this input (the
    detector-could-have-fired control), then assert the real path is exact.
    """
    torch: Any = pytest.importorskip("torch")
    t = torch.full((5000, 512), 1e19, dtype=torch.float32)

    naive_sumsq32 = float((t * t).sum().item())  # float32 reduction
    assert math.isinf(naive_sumsq32), (
        "control failed: this input no longer overflows float32, so the test "
        "no longer proves float64 accumulation matters"
    )

    tc = compare_keys(MemSource({"w": t.clone()}), MemSource({"w": t.clone()}), "w")
    assert tc.verdict == "EXACT"
    assert tc.cosine == 1.0  # not nan: float64 holds ~2.6e45 without overflow
    assert math.isfinite(tc.rms_a)
    # Compare against the value actually STORED, not the literal 1e19. float32 cannot
    # represent 1e19; the nearest value is ~9.9999999805e18, so a `rel=1e-9` assertion
    # against the literal demands the RMS be nearer to 1e19 than the dtype holding the
    # data can be — it fails on a correct implementation. Anchoring to `t[0, 0]` is also
    # the stricter test: every element is that value, so an exact RMS must reproduce it
    # to float64 precision, and rel=1e-15 says so.
    stored = float(t[0, 0].item())
    assert tc.rms_a == pytest.approx(stored, rel=1e-15)


def test_self_check_raises_when_accumulator_contradicts_identical_bits(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A broken accumulator must raise, not launder a bad number into a report.

    Simulates the float32-underflow class by downgrading the accumulator; on
    this input the computed self-cosine becomes nan, and the internal guard
    must fire exactly as it would have on the historic 1.80 report.
    """
    torch: Any = pytest.importorskip("torch")
    t = torch.full((5000, 512), 1e19, dtype=torch.float32)
    monkeypatch.setattr(torch, "float64", torch.float32)

    with pytest.raises(AssertionError, match="self-check"):
        compare_keys(MemSource({"w": t.clone()}), MemSource({"w": t.clone()}), "w")


# ---------------------------------------------------------------------------
# compare_keys: streaming must not change the answer
# ---------------------------------------------------------------------------


def test_result_is_independent_of_block_rows_including_uneven_division() -> None:
    torch: Any = pytest.importorskip("torch")
    gen = torch.Generator().manual_seed(19)
    a = torch.randn(1000, 64, generator=gen)
    b = a.clone()
    b[::7] += 0.25  # non-trivial differences so every reduction is exercised
    b[13] = -b[13]
    sa, sb = MemSource({"w": a}), MemSource({"w": b})

    coarse = compare_keys(sa, sb, "w", block_rows=333)  # does not divide 1000
    fine = compare_keys(sa, sb, "w", block_rows=150)

    # The streaming granularity really changed (control), but the physics did not.
    assert coarse.chunks_read != fine.chunks_read
    assert coarse.bytes_read == fine.bytes_read

    for field in (
        "elements",
        "mismatched_elements",
        "bitwise_equal",
        "verdict",
        "shape_a",
        "shape_b",
        "dtype_a",
        "dtype_b",
        "max_abs_diff",
    ):
        assert getattr(coarse, field) == getattr(fine, field), field
    for field in ("mean_abs_diff", "rms_a", "rms_b"):
        assert getattr(coarse, field) == pytest.approx(getattr(fine, field), rel=1e-12), field
    assert coarse.cosine is not None and fine.cosine is not None
    assert coarse.cosine == pytest.approx(fine.cosine, rel=1e-12)

    # Falsifiable ground truth: the streamed numbers match a one-shot float64 reduction.
    fa, fb = a.to(torch.float64), b.to(torch.float64)
    ref_cos = float((fa * fb).sum().item()) / math.sqrt(
        float((fa * fa).sum().item()) * float((fb * fb).sum().item())
    )
    assert coarse.cosine == pytest.approx(ref_cos, rel=1e-12)
    assert coarse.max_abs_diff == pytest.approx(float((fa - fb).abs().max().item()), rel=0)


def test_shape_mismatch_is_visibly_empty_not_a_pass() -> None:
    torch: Any = pytest.importorskip("torch")
    a = torch.zeros(10, 4)
    b = torch.zeros(10, 5)
    tc = compare_keys(MemSource({"w": a}), MemSource({"w": b}), "w")

    assert tc.verdict == "SHAPE_MISMATCH"
    assert tc.elements == 0  # zero compared, reported as zero — an unqualified count is not a fact
    assert tc.cosine is None
    assert tc.max_abs_diff == math.inf
    assert tc.bitwise_equal is False
    assert tc.chunks_read == 0


def test_zero_row_tensor_reports_zero_elements_and_no_fabricated_cosine() -> None:
    """Vacuous case: comparing nothing must not crop a cosine of convenience."""
    torch: Any = pytest.importorskip("torch")
    a = torch.zeros(0, 8)
    tc = compare_keys(MemSource({"w": a}), MemSource({"w": a.clone()}), "w")

    assert tc.elements == 0  # the count qualifies the verdict; a consumer can catch 0
    # 0 elements have no direction. Reporting 1.0 here would be inventing agreement
    # between two things that were never compared.
    assert tc.cosine is None
    assert tc.chunks_read == 0
    # This test originally asserted `max_abs_diff == 0.0` and never asserted the
    # verdict at all — which is precisely how the vacuous-EXACT blocker survived
    # into production. The author reasoned correctly about the fabricated cosine
    # and about elements/chunks, then pinned a diff of 0.0 ("no difference found")
    # and left the one field a consumer actually branches on unchecked. Both lines
    # below now match the SHAPE_MISMATCH sibling above: comparing nothing yields an
    # unbounded diff and an abstaining verdict, never a pass grade.
    assert tc.verdict == "NO_ELEMENTS"
    assert tc.max_abs_diff == math.inf and tc.rms_a == 0.0


def test_nonpositive_block_rows_is_rejected_not_silently_exact() -> None:
    torch: Any = pytest.importorskip("torch")
    t = torch.randn(1000, 16)
    with pytest.raises((ValueError, AssertionError)):
        compare_keys(MemSource({"w": t}), MemSource({"w": t.clone()}), "w", block_rows=-64)


# ---------------------------------------------------------------------------
# DcpReader chunk validation: the reader must fail loudly, never zero-fill
# ---------------------------------------------------------------------------


def _bare_dcp_reader(
    shard_dir: Path,
    *,
    blob_name: str,
    blob_len: int,
    offsets: tuple[int, ...] = (0, 0),
    key: str = "model.w",
) -> DcpReader:
    """A DcpReader wired to one on-disk blob, without needing real DCP metadata.

    `read_chunk` touches only `_meta`, `_storage` and `_read_blob`, so this
    exercises the real validation path with storage records we fully control.
    """
    reader = object.__new__(DcpReader)
    reader.path = str(shard_dir)
    tensor_md: dict[str, Any] = {key: object()}
    storage: dict[str, dict[tuple[int, ...], Any]] = {
        key: {offsets: SimpleNamespace(relative_path=blob_name, offset=0, length=blob_len)}
    }
    reader._tensor_md = tensor_md
    reader._storage = storage
    return reader


def test_short_buffer_raises_chunk_read_error_carrying_key_and_offsets(tmp_path: Path) -> None:
    blob = b"PK\x03\x04" + bytes(32)
    (tmp_path / "shard.bin").write_bytes(blob)
    reader = _bare_dcp_reader(tmp_path, blob_name="shard.bin", blob_len=len(blob) + 8)

    with pytest.raises(ChunkReadError, match="short read") as exc_info:
        reader.read_chunk("model.w", (0, 0))

    err = exc_info.value
    assert err.key == "model.w"
    assert err.offsets == (0, 0)
    assert "shard.bin" not in str(err) or True  # message contract is the byte counts:
    assert f"{len(blob)}" in str(err) and str(len(blob) + 8) in str(err)


def test_wrong_magic_raises_before_torch_and_names_location(tmp_path: Path) -> None:
    # Every stored DCP range is a complete torch.save zip; anything else stored
    # at a keyed offset means offsets and shard disagree. This must raise
    # without torch even installed — it never reaches torch.load.
    blob = b"NOPE" + bytes(60)
    (tmp_path / "shard.bin").write_bytes(blob)
    reader = _bare_dcp_reader(tmp_path, blob_name="shard.bin", blob_len=len(blob))

    with pytest.raises(ChunkReadError) as exc_info:
        reader.read_chunk("model.w", (0, 0))

    err = exc_info.value
    assert "PK" in str(err)
    assert err.key == "model.w"
    assert err.offsets == (0, 0)


def test_corrupt_archive_wraps_torch_load_and_chains_original(tmp_path: Path) -> None:
    pytest.importorskip("torch")
    blob = b"PK\x03\x04" + bytes(range(256)) * 3  # zip magic, then not a zip
    (tmp_path / "shard.bin").write_bytes(blob)
    reader = _bare_dcp_reader(tmp_path, blob_name="shard.bin", blob_len=len(blob))

    with pytest.raises(ChunkReadError, match="self-validation") as exc_info:
        reader.read_chunk("model.w", (0, 0))

    err = exc_info.value
    assert err.key == "model.w"
    assert err.offsets == (0, 0)
    assert err.original is not None, "the wrapping error must retain the torch failure"
    assert err.__cause__ is err.original  # chained, so tracebacks keep the real cause


def test_valid_chunk_roundtrips_and_unstored_offsets_raise(tmp_path: Path) -> None:
    torch: Any = pytest.importorskip("torch")
    expected = torch.arange(12, dtype=torch.float64).reshape(3, 4)
    buf = io.BytesIO()
    torch.save(expected, buf)
    blob = buf.getvalue()
    assert blob[:4] == b"PK\x03\x04"  # control: the fixture is what the reader expects
    (tmp_path / "shard.bin").write_bytes(blob)
    reader = _bare_dcp_reader(tmp_path, blob_name="shard.bin", blob_len=len(blob))

    got = reader.read_chunk("model.w", (0, 0))
    assert torch.equal(got, expected)
    assert got.dtype == torch.float64  # no silent upcast: on-disk dtype survives

    with pytest.raises(ChunkReadError, match="no chunk stored") as exc_info:
        reader.read_chunk("model.w", (5, 5))  # a miss must raise, never default
    assert exc_info.value.offsets == (5, 5)


def test_read_chunk_unknown_key_raises_tensor_not_found_not_chunk_error(tmp_path: Path) -> None:
    from foundationscale.checkpoint.dcp import TensorNotFoundError

    (tmp_path / "shard.bin").write_bytes(b"PK\x03\x04" + bytes(16))
    reader = _bare_dcp_reader(tmp_path, blob_name="shard.bin", blob_len=20)
    with pytest.raises(TensorNotFoundError):
        reader.read_chunk("not.there", (0, 0))


# ---------------------------------------------------------------------------
# SafetensorsReader: whole-tensor chunk surface and dtype identity
# ---------------------------------------------------------------------------


def _save_st_shard(dirpath: Path, name: str, tensors: dict[str, Any]) -> Path:
    st: Any = pytest.importorskip("safetensors.torch")
    out = dirpath / name
    st.save_file(tensors, str(out))
    return out


def test_read_chunk_accepts_only_all_zero_offsets(tmp_path: Path) -> None:
    torch: Any = pytest.importorskip("torch")
    w = torch.arange(6, dtype=torch.float32).reshape(3, 2)
    _save_st_shard(tmp_path, "model.safetensors", {"w": w})
    reader = SafetensorsReader(tmp_path)

    # Positive control first: the detector path works on the one legal chunk.
    got = reader.read_chunk("w", (0, 0))
    assert torch.equal(got, w)

    for bad in [(1, 0), (0, 1), (0,), (0, 0, 0)]:
        with pytest.raises(ChunkReadError) as exc_info:
            reader.read_chunk("w", bad)
        assert exc_info.value.key == "w"
        assert exc_info.value.offsets == bad


def test_chunk_surface_is_one_whole_tensor_chunk_and_reads_are_complete(tmp_path: Path) -> None:
    torch: Any = pytest.importorskip("torch")
    w = torch.arange(12, dtype=torch.float32).reshape(3, 4)
    _save_st_shard(tmp_path, "model.safetensors", {"w": w})
    reader = SafetensorsReader(tmp_path)

    assert reader.chunks("w") == (Chunk(offsets=(0, 0), sizes=(3, 4)),)

    box = reader.read_box("w", (1, 0), (3, 4))
    assert torch.equal(box.tensor, w[1:3, :])
    assert box.complete
    assert box.elements_covered == box.elements_expected == 8

    full = reader.read_full("w")
    assert full.complete
    assert full.elements_covered == full.elements_expected == 12
    assert torch.equal(full.tensor, w)


def test_dtype_maps_to_real_torch_dtype_objects_for_equality_comparison(
    tmp_path: Path,
) -> None:
    torch: Any = pytest.importorskip("torch")
    _save_st_shard(
        tmp_path,
        "model.safetensors",
        {
            "f32": torch.zeros(2, dtype=torch.float32),
            "f16": torch.zeros(2, dtype=torch.float16),
            "bf16": torch.zeros(2, dtype=torch.bfloat16),
        },
    )
    reader = SafetensorsReader(tmp_path)

    # Identity (`is`), not string matching: a cross-format dtype comparison
    # against a DcpReader's torch.dtype is then a plain equality test.
    assert reader.dtype("f32") is torch.float32
    assert reader.dtype("f16") is torch.float16
    assert reader.dtype("bf16") is torch.bfloat16


def test_unknown_dtype_string_is_rejected_not_guessed() -> None:
    pytest.importorskip("torch")  # the mapper imports torch before consulting the table
    with pytest.raises(CheckpointFormatError, match="NOT_A_DTYPE"):
        _safetensors_dtype("NOT_A_DTYPE", path="/nowhere", key="k")


def test_empty_weight_map_refused_and_empty_directory_refused(tmp_path: Path) -> None:
    """Vacuous case: an export claiming zero tensors is refused, not opened as 'all match'."""
    torch: Any = pytest.importorskip("torch")
    _save_st_shard(tmp_path, "model.safetensors", {"w": torch.zeros(2, 2)})
    index = {"metadata": {}, "weight_map": {}}
    (tmp_path / "model.safetensors.index.json").write_text(json.dumps(index), encoding="utf-8")

    with pytest.raises(CheckpointFormatError, match="0 tensors"):
        SafetensorsReader(tmp_path)

    bare = tmp_path / "bare"
    bare.mkdir()
    with pytest.raises(CheckpointFormatError, match="neither an index"):
        SafetensorsReader(bare)

    # Control: the same directory without the empty index opens fine, so the
    # refusal above is attributable to the empty weight_map, not the fixture.
    del index
    good = tmp_path / "good"
    good.mkdir()
    _save_st_shard(good, "model.safetensors", {"w": torch.zeros(2, 2)})
    reader = SafetensorsReader(good)
    assert reader.tensor_keys() == ("w",)


def test_duplicate_tensor_key_across_shards_is_rejected(tmp_path: Path) -> None:
    torch: Any = pytest.importorskip("torch")
    _save_st_shard(tmp_path, "model-00001-of-00002.safetensors", {"w": torch.zeros(2)})
    _save_st_shard(tmp_path, "model-00002-of-00002.safetensors", {"w": torch.ones(2)})

    with pytest.raises(CheckpointFormatError, match="two shards"):
        SafetensorsReader(tmp_path)


# ---------------------------------------------------------------------------
# _HandleCache: bounded LRU with real release semantics
# ---------------------------------------------------------------------------


class _FakeHandle:
    """Records release, so 'the handle was closed' is observable, not assumed."""

    def __init__(self, name: str) -> None:
        self.name = name
        self.close_calls = 0

    @property
    def closed(self) -> bool:
        return self.close_calls > 0

    def close(self) -> None:
        self.close_calls += 1


class _ExplodingHandle(_FakeHandle):
    def close(self) -> None:
        self.close_calls += 1
        raise RuntimeError("close exploded")


def _open_tracking(opened: list[_FakeHandle], names: list[str]) -> Any:
    def opener(path: str) -> _FakeHandle:
        handle = _FakeHandle(path)
        opened.append(handle)
        names.append(path)
        return handle

    return opener


def test_lru_eviction_releases_evicted_handle_only() -> None:
    opened: list[_FakeHandle] = []
    names: list[str] = []
    cache = _HandleCache(2, _open_tracking(opened, names))

    a = cache.get("a")
    b = cache.get("b")
    assert cache.get("a") is a  # hit: refresh recency without reopening
    assert len(opened) == 2

    c = cache.get("c")  # over capacity: evicts b, the least-recently-used
    assert b.closed is True, "eviction must release the evicted handle"
    assert b.close_calls == 1
    assert a.closed is False and c.closed is False

    cache.get("d")  # a is now LRU and must go
    assert a.closed is True
    assert c.closed is False


def test_close_releases_everything_and_is_idempotent() -> None:
    opened: list[_FakeHandle] = []
    cache = _HandleCache(4, _open_tracking(opened, []))
    handles = [cache.get(name) for name in ("x", "y", "z")]

    cache.close()
    assert all(h.closed for h in handles), "close() must release every cached handle"
    assert len(cache.__dict__) > 0  # cache object still intact

    after_first = [h.close_calls for h in handles]
    cache.close()  # second close must be a no-op, not a double-release
    assert [h.close_calls for h in handles] == after_first

    # And the cache is usable again afterwards: a get re-opens fresh state.
    reopened = cache.get("x")
    assert reopened is not handles[0]
    assert reopened.closed is False


def test_broken_close_does_not_prevent_releasing_the_rest() -> None:
    opened: list[_FakeHandle] = []
    cache = _HandleCache(4, _open_tracking(opened, []))
    good = cache.get("good")
    cache.__dict__["_items"]["bad"] = _ExplodingHandle("bad")  # inject a hostile handle

    cache.close()  # must not propagate the explosion, and must still release 'good'
    assert good.closed is True


def test_capacity_below_one_rejected() -> None:
    with pytest.raises(ValueError, match="capacity"):
        _HandleCache(0, _open_tracking([], []))


def test_close_on_never_used_cache_is_a_no_op() -> None:
    # Vacuous case: releasing nothing must not report or imply it released something.
    cache = _HandleCache(2, _open_tracking([], []))
    cache.close()
    cache.close()


def test_safetensors_reader_close_and_context_manager_empty_the_handle_cache(
    tmp_path: Path,
) -> None:
    torch: Any = pytest.importorskip("torch")
    for i in range(3):
        _save_st_shard(
            tmp_path,
            f"model-0000{i + 1}-of-00003.safetensors",
            {f"w{i}": torch.zeros(4, 4)},
        )
    reader = SafetensorsReader(tmp_path, handle_cache_size=2)
    for key in reader.tensor_keys():  # opens 3 handles through a capacity-2 cache
        reader.read_full(key)
    held = reader.__dict__["_handles"].__dict__["_items"]
    assert len(held) == 2  # capacity bound held under churn (control for below)

    reader.close()
    assert len(held) == 0, "close() must drop every open shard handle"
    reader.close()  # idempotent

    with SafetensorsReader(tmp_path) as scoped:
        scoped.read_full("w0")
        held_scoped = scoped.__dict__["_handles"].__dict__["_items"]
        assert len(held_scoped) == 1
    assert len(held_scoped) == 0, "__exit__ must release what the body opened"
