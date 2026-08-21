"""Bridge tests: real bytes on disk, through ``from_path``, into blocking verdicts.

Why this file exists
--------------------
The nine gates currently pass 32 controls, every one over a synthetic fixture built
by hand in ``fixtures.py``. That proves the gates work against inputs the framework
imagined for itself. It does not prove that a checkpoint a trainer actually wrote —
a ``.metadata`` pickle, shard files, byte ranges — reaches those gates at all.
``CheckpointGateContext.from_path`` is the only code that carries a real artifact
across that gap, and until recently it called functions that did not exist.

So every context in this file is produced by :meth:`CheckpointGateContext.from_path`
from a checkpoint written to ``tmp_path`` by a real ``torch.save`` writer (the same
construction ``tests/test_dcp_coverage.py`` uses). The single most important test is
``test_aliased_expert_bytes_produce_equal_storage_ids_and_block_distinctness``: it
reproduces Incident #1 *on disk* (many expert FQNs, one stored byte range) and
asserts the chain bytes → ``storage_id`` → expert-distinctness gate ends in a
blocking FAIL.

Two seams are deliberately stubbed and must not be confused with fixture cheating:

* ``load_manifest`` is monkeypatched where a test needs a manifest that *declares*
  model facts (expert counts, byte volumes). The provenance ``RunManifest`` has no
  such fields, so no manifest the real reader could return would carry them either;
  the checkpoint half of every test remains fully real. See the module NOTES.
* Assertions about ``fsckpt.read_metadata`` are written against the frozen API
  contract, never against a guessed implementation.
"""

from __future__ import annotations

import json
import os
import pickle
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest
from test_dcp_coverage import _archive_bytes, _write_dcp_checkpoint

from foundationscale import checkpoint as fsckpt
from foundationscale.checkpoint.dcp import (
    CheckpointError,
    ChunkReadError,
    DcpReader,
    SafetensorsReader,
)
from foundationscale.gates.checkpoint_gates import CheckpointGateContext
from foundationscale.gates.core import REGISTRY, Verdict

_CHECKPOINT_GATE_IDS = (
    "checkpoint.expert_distinctness",
    "checkpoint.expert_bytes",
    "checkpoint.save_complete",
    "checkpoint.first_save",
)

_EXPERT_LAYERS = (0, 1)
_NUM_EXPERTS = 8
_EXPERT_SHAPE = (8, 3, 5)
_PROJ_BYTES = 8 * 3 * 5 * 4  # shape x float32
# 2 layers x 2 projections (fc1, fc2) x 480 B.
EXPECTED_EXPERT_BYTES = 2 * 2 * _PROJ_BYTES


@pytest.fixture
def torch_mod() -> Any:
    """The torch module, or a skip, paying for the import only where it is needed."""
    return pytest.importorskip("torch")


# ---------------------------------------------------------------------------
# Manifest plumbing
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _ManifestStub:
    """The narrow attribute surface ``from_path`` reads off a manifest.

    The real ``RunManifest`` carries code/config/environment/topology — none of the
    four attributes below. Any object ``load_manifest`` returns must expose this
    surface for the declared-vs-present gates to have a denominator at all.
    """

    declared_fqns: tuple[str, ...]
    num_experts: int | None
    num_moe_layers: int | None
    expected_expert_bytes: int | None


def _install_manifest(monkeypatch: pytest.MonkeyPatch, stub: _ManifestStub) -> None:
    """Point ``from_path``'s manifest call at ``stub``; tensor metadata stays real."""
    monkeypatch.setattr(fsckpt, "load_manifest", lambda _path: stub, raising=False)


# ---------------------------------------------------------------------------
# On-disk checkpoint writers
# ---------------------------------------------------------------------------


def _expert_fqns() -> list[str]:
    return [
        f"model.layers.{layer}.mlp.experts.linear_fc{proj}.weight"
        for layer in _EXPERT_LAYERS
        for proj in (1, 2)
    ]


def _write_healthy_moe_checkpoint(root: Path, torch_mod: Any) -> tuple[str, ...]:
    """Fused-layout MoE checkpoint: one (experts, in, out) tensor per weight/layer."""
    tensors: dict[str, tuple[tuple[int, ...], list[tuple[tuple[int, ...], Any]]]] = {}
    for i, fqn in enumerate(_expert_fqns()):
        blob = torch_mod.arange(120, dtype=torch_mod.float32).reshape(_EXPERT_SHAPE)
        tensors[fqn] = (_EXPERT_SHAPE, [((0, 0, 0), blob + float(i * 1000))])
    for i, layer in enumerate(_EXPERT_LAYERS):
        fqn = f"model.layers.{layer}.attention.qkv.weight"
        blob = torch_mod.arange(24, dtype=torch_mod.float32).reshape(4, 6)
        tensors[fqn] = ((4, 6), [((0, 0), blob + float(i * 10_000))])
    _write_dcp_checkpoint(root, tensors=tensors, nontensor=["_extra_state.optimizer"])
    return tuple(sorted(tensors))


def _write_aliased_dcp_checkpoint(
    root: Path,
    *,
    blob_for_fqns: list[tuple[list[str], Any]],
    dense: dict[str, Any],
) -> None:
    """Write a DCP checkpoint in which every FQN group shares ONE stored byte range.

    This is Incident #1 rendered as bytes: each alias group's FQNs get their own
    tensor metadata and their own ``MetadataIndex``, but every storage record in a
    group is the same ``(relative_path, offset, length)`` — 128 (here 32) names,
    2 distinct allocations. ``_write_dcp_checkpoint`` cannot express this, because
    it appends fresh bytes per chunk; aliasing requires writing the blob once.
    """
    torch = pytest.importorskip("torch")
    from torch.distributed.checkpoint import filesystem as dcfs
    from torch.distributed.checkpoint import metadata as dcmeta

    storage_info_cls = None
    for mod in (dcfs, dcmeta):
        storage_info_cls = getattr(mod, "_StorageInfo", None)
        if storage_info_cls is not None:
            break
    if storage_info_cls is None:  # pragma: no cover - torch API drift
        pytest.skip("torch's private _StorageInfo moved again; fixture needs updating")

    shard_name = "shard_0.distcp"
    state_dict_metadata: dict[str, Any] = {}
    storage_data: dict[Any, Any] = {}
    parts: list[bytes] = []
    cursor = 0

    for fqns, tensor in blob_for_fqns:
        data = _archive_bytes(tensor)
        parts.append(data)
        blob_offset = cursor
        cursor += len(data)
        for fqn in fqns:
            index = dcmeta.MetadataIndex(fqn=fqn, offset=torch.Size((0,) * tensor.dim()))
            # Distinct storage RECORDS, one identical byte-range TRIPLE apiece —
            # the way the real aliased artifact actually looked on disk.
            storage_data[index] = storage_info_cls(
                relative_path=shard_name, offset=blob_offset, length=len(data)
            )
            state_dict_metadata[fqn] = dcmeta.TensorStorageMetadata(
                size=torch.Size(tuple(tensor.shape)),
                properties=dcmeta.TensorProperties(dtype=tensor.dtype),
                chunks=[
                    dcmeta.ChunkStorageMetadata(
                        offsets=torch.Size((0,) * tensor.dim()),
                        sizes=torch.Size(tuple(tensor.shape)),
                    )
                ],
            )

    for fqn, tensor in dense.items():
        data = _archive_bytes(tensor)
        parts.append(data)
        index = dcmeta.MetadataIndex(fqn=fqn, offset=torch.Size((0,) * tensor.dim()))
        storage_data[index] = storage_info_cls(
            relative_path=shard_name, offset=cursor, length=len(data)
        )
        cursor += len(data)
        state_dict_metadata[fqn] = dcmeta.TensorStorageMetadata(
            size=torch.Size(tuple(tensor.shape)),
            properties=dcmeta.TensorProperties(dtype=tensor.dtype),
            chunks=[
                dcmeta.ChunkStorageMetadata(
                    offsets=torch.Size((0,) * tensor.dim()),
                    sizes=torch.Size(tuple(tensor.shape)),
                )
            ],
        )

    metadata = dcmeta.Metadata(state_dict_metadata=state_dict_metadata, storage_data=storage_data)
    (root / ".metadata").write_bytes(pickle.dumps(metadata))
    (root / shard_name).write_bytes(b"".join(parts))


def _write_two_tensor_safetensors(shard: Path) -> int:
    """Two-tensor safetensors file written with stdlib only. Returns payload offset."""
    payload = struct.pack("<4f", 1.0, 2.0, 3.0, 4.0)
    header = json.dumps(
        {
            "a": {"dtype": "F32", "shape": [2], "data_offsets": [0, 8]},
            "b": {"dtype": "F32", "shape": [2], "data_offsets": [8, 16]},
        }
    ).encode("utf-8")
    shard.write_bytes(struct.pack("<Q", len(header)) + header + payload)
    return 8 + len(header)


# ---------------------------------------------------------------------------
# 1. Healthy round trip
# ---------------------------------------------------------------------------


def test_healthy_checkpoint_round_trips_from_disk_to_passing_gates(
    tmp_path: Path, torch_mod: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A correct checkpoint must leave every checkpoint gate PASS — with a count.

    A green verdict over zero examined units is the ``all([])`` success this
    framework exists to kill, so the coverage counts are asserted as hard numbers,
    not merely ``> 0``: the exact expert tensors, declared tensors and sub-gates
    this checkpoint actually contains. If this fails, either the bytes-to-metadata
    bridge dropped data, or a gate blocks a correct save — either way the
    framework's green light means nothing everywhere else.
    """
    declared = _write_healthy_moe_checkpoint(tmp_path, torch_mod)
    _install_manifest(
        monkeypatch,
        _ManifestStub(
            declared_fqns=declared,
            num_experts=_NUM_EXPERTS,
            num_moe_layers=len(_EXPERT_LAYERS),
            expected_expert_bytes=EXPECTED_EXPERT_BYTES,
        ),
    )

    ctx = CheckpointGateContext.from_path(tmp_path)

    real = [t for t in ctx.tensors if t.kind == "tensor"]
    assert len(real) == 6  # 4 expert projections + 2 dense attention weights
    assert all(t.storage_id is not None for t in real)
    assert all(t.dtype == "float32" for t in real)

    results = {gid: REGISTRY.get(gid).run(ctx) for gid in _CHECKPOINT_GATE_IDS}

    for gid, result in results.items():
        assert result.verdict is Verdict.PASS, f"{gid}: {result.detail}"
        assert result.coverage.checked > 0, f"{gid} passed over zero units"

    assert results["checkpoint.expert_distinctness"].coverage.checked == 4
    assert results["checkpoint.expert_bytes"].coverage.checked == 4
    assert results["checkpoint.save_complete"].coverage.checked == 6
    assert results["checkpoint.first_save"].coverage.checked == 3


# ---------------------------------------------------------------------------
# 2. Incident #1, end to end
# ---------------------------------------------------------------------------


def test_aliased_expert_bytes_produce_equal_storage_ids_and_block_distinctness(
    tmp_path: Path, torch_mod: Any
) -> None:
    """32 expert FQNs backed by 2 stored blobs must equal-alias and must block.

    This is the only test in the package that walks the whole chain the incident
    walked: torch-written shard bytes -> DCP metadata -> ``read_metadata`` ->
    ``from_path`` -> ``TensorMeta.storage_id`` -> expert-distinctness gate.
    Every FQN is present, every shape and dtype is right, the count even matches
    the sharded layout — the only observable is that 16 names of each projection
    resolve to one storage_id. A failure anywhere in that chain means an 8-way
    aliased expert save can again reach disk while every gate reports green.
    """
    fc1 = [f"model.layers.0.mlp.experts.linear_fc1.weight{i}" for i in range(16)]
    fc2 = [f"model.layers.0.mlp.experts.linear_fc2.weight{i}" for i in range(16)]
    dense_names = {
        "model.layers.0.attention.qkv.weight": torch_mod.arange(15).reshape(3, 5),
        "model.layers.1.attention.qkv.weight": torch_mod.arange(15).reshape(3, 5) + 100.0,
    }
    _write_aliased_dcp_checkpoint(
        tmp_path,
        blob_for_fqns=[
            (fc1, torch_mod.full((3, 5), 7.0)),
            (fc2, torch_mod.full((3, 5), 9.0)),
        ],
        dense=dense_names,
    )

    # Independent forensic proof the aliasing is physical, not asserted into
    # existence: two different FQNs read through the raw DCP reader return
    # byte-identical tensors because they point at the same stored range.
    reader = DcpReader(tmp_path)
    assert torch_mod.equal(reader.read_full(fc1[0]).tensor, reader.read_full(fc1[7]).tensor)

    ctx = CheckpointGateContext.from_path(tmp_path)
    meta = {t.fqn: t for t in ctx.tensors}

    fc1_ids = {meta[f].storage_id for f in fc1}
    fc2_ids = {meta[f].storage_id for f in fc2}
    assert None not in fc1_ids, "storage_id None reads as 'unknown', never as aliased"
    assert len(fc1_ids) == 1, "16 FQNs over one byte range must collapse to one storage_id"
    assert len(fc2_ids) == 1
    assert fc1_ids != fc2_ids, "distinct bytes must never share a storage_id"
    dense_ids = {meta[f].storage_id for f in dense_names}
    assert len(dense_ids) == 2, "two distinct dense tensors must not alias"
    assert not (dense_ids & (fc1_ids | fc2_ids))

    result = REGISTRY.get("checkpoint.expert_distinctness").run(ctx)

    assert result.verdict is Verdict.FAIL, result.detail
    assert result.blocking
    assert "share" in result.detail
    assert result.coverage.checked == 32


# ---------------------------------------------------------------------------
# 3. No manifest beside the checkpoint
# ---------------------------------------------------------------------------


def test_checkpoint_without_manifest_builds_context_but_cannot_attest_completeness(
    tmp_path: Path, torch_mod: Any
) -> None:
    """A manifest absence is a normal case — and the completeness gate must not pass.

    Most checkpoints on disk have no provenance manifest at all. ``from_path``
    must survive that (``declared_fqns=None``), and the gate that compares
    present-against-declared must then visibly fail to do its job: anything but
    PASS. If the context build crashed, ordinary checkpoints would be
    unverifiable; if the gate passed, "no manifest" would read as "nothing
    wrong" — the ``all([])`` shape one level up.
    """
    _write_healthy_moe_checkpoint(tmp_path, torch_mod)

    ctx = CheckpointGateContext.from_path(tmp_path)  # must not raise

    assert ctx.declared_fqns is None
    assert ctx.num_experts is None
    assert ctx.num_moe_layers is None
    assert ctx.expected_expert_bytes is None
    assert ctx.tensors  # the checkpoint itself reached the context unharmed

    result = REGISTRY.get("checkpoint.save_complete").run(ctx)

    assert result.verdict is not Verdict.PASS
    # Was `is Verdict.SKIP`, which pinned the defect this docstring already argued
    # against: SKIP is non-blocking, so a manifest-less checkpoint sailed through. SKIP
    # belongs to a check that is genuinely inapplicable — a dense model has no experts,
    # so an expert gate has nothing to say. Completeness is applicable to every
    # checkpoint; here the gate merely has no denominator. That is missing evidence, and
    # missing evidence must block.
    assert result.verdict.blocking
    assert "manifest" in result.detail.lower()


# Was a strict xfail: SaveCompletenessGate returned SKIP (non-blocking) when
# declared_fqns was None, so "no manifest" read as "nothing wrong". Fixed — the gate now
# returns VACUOUS. The marker is gone rather than left in place because a strict xfail
# that has started passing is itself a failing test, which is the whole point of strict.
def test_completeness_gate_blocks_when_nothing_is_declared(tmp_path: Path, torch_mod: Any) -> None:
    """Strict form of the requirement: absence of a denominator must block."""
    _write_healthy_moe_checkpoint(tmp_path, torch_mod)
    ctx = CheckpointGateContext.from_path(tmp_path)

    result = REGISTRY.get("checkpoint.save_complete").run(ctx)

    assert result.verdict in (Verdict.VACUOUS, Verdict.UNDERCOVERED)


# Was a strict xfail: FirstSaveGate reported "distinctness, byte volume and completeness
# all hold at first save" while two of its three sub-gates had SKIPPED — one sentence
# asserting three facts, one of which was established. Fixed: the composite downgrades to
# UNDERCOVERED over any abstention, and its message is now built from the sub-gates that
# actually ran.
def test_first_save_composite_must_not_pass_over_skipped_subgates(
    tmp_path: Path, torch_mod: Any
) -> None:
    """A composite whose byte-volume and completeness gates abstained may not pass."""
    _write_healthy_moe_checkpoint(tmp_path, torch_mod)
    ctx = CheckpointGateContext.from_path(tmp_path)

    result = REGISTRY.get("checkpoint.first_save").run(ctx)

    assert result.verdict is not Verdict.PASS


# ---------------------------------------------------------------------------
# 4. A manifest that disagrees with the checkpoint
# ---------------------------------------------------------------------------


def test_manifest_declaring_128_experts_blocks_checkpoint_holding_16(
    tmp_path: Path, torch_mod: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Declared 128, on-disk 16: the gate must block and name both numbers.

    This is the configuration half of Incident #1: the local-name save wrote one
    tensor per *local* expert. The shards here carry distinct bytes on purpose —
    aliasing is deliberately absent — so the only thing that can fire is the
    declared-vs-present count. If this test stops failing, the denominator the
    run manifest carries is no longer reaching the gate, and the count check is
    decorative.
    """
    tensors: dict[str, tuple[tuple[int, ...], list[tuple[tuple[int, ...], Any]]]] = {}
    for proj in (1, 2):
        for i in range(16):
            fqn = f"model.layers.0.mlp.experts.linear_fc{proj}.weight{i}"
            tensors[fqn] = ((3, 5), [((0, 0), torch_mod.full((3, 5), float(100 * proj + i)))])
    _write_dcp_checkpoint(tmp_path, tensors=tensors)
    _install_manifest(
        monkeypatch,
        _ManifestStub(
            declared_fqns=tuple(
                f"model.layers.0.mlp.experts.linear_fc{proj}.weight" for proj in (1, 2)
            ),
            num_experts=128,
            num_moe_layers=1,
            expected_expert_bytes=None,
        ),
    )

    ctx = CheckpointGateContext.from_path(tmp_path)
    result = REGISTRY.get("checkpoint.expert_distinctness").run(ctx)

    assert result.verdict is Verdict.FAIL, result.detail
    assert result.blocking
    assert "16" in result.detail
    assert "128" in result.detail


# ---------------------------------------------------------------------------
# 5. read_metadata is metadata-only
# ---------------------------------------------------------------------------


def test_read_metadata_reports_tensors_whose_bytes_do_not_exist(
    tmp_path: Path, torch_mod: Any
) -> None:
    """Metadata about absent or huge tensors must come back without touching bytes.

    The honest form of "never materialises tensor data" available to this API:
    describe tensors whose payload bytes were truncated to nothing, plus one
    tensor declared at ~(10**9, 4096) — far too large to upcast, let alone load.
    If ``read_metadata`` read payloads, this checkpoint could not be described at
    all: the shard is empty and the huge tensor would exhaust memory. The paired
    raw-reader refusal is the control: it proves the payloads really are unreadable
    at the moment read_metadata succeeds. A failure here means the cheap
    pre-flight is actually a full load, and every gate that assumes metadata-only
    cost inherits an 11.5 GB surprise.
    """
    hshape = (10**9, 4096)
    _write_dcp_checkpoint(
        tmp_path,
        tensors={
            "plain_w": (
                (2, 2),
                [((0, 0), torch_mod.arange(4, dtype=torch_mod.float32).reshape(2, 2))],
            ),
            "huge_w": (hshape, [((0, 0), torch_mod.zeros(1, 1))]),
            "top.embed.weight": ((4, 6), [((0, 0), torch_mod.arange(24).reshape(4, 6))]),
        },
        nontensor=["_extra_state.optimizer"],
    )
    (tmp_path / "shard_0.distcp").write_bytes(b"")
    assert (tmp_path / "shard_0.distcp").stat().st_size == 0

    meta = fsckpt.read_metadata(os.fspath(tmp_path))

    assert meta.format == "dcp"
    assert meta.origin
    assert {"plain_w", "huge_w", "top.embed.weight"} <= set(meta.tensors)
    assert tuple(meta.tensors["huge_w"].shape) == hshape
    assert meta.tensors["plain_w"].dtype == "float32"  # plain, no "torch." prefix
    assert meta.tensors["plain_w"].is_extra_state is False
    assert meta.tensors["plain_w"].storage_id is not None
    if "_extra_state.optimizer" in meta.tensors:
        assert meta.tensors["_extra_state.optimizer"].is_extra_state is True

    # Control: the payloads really are gone, so if read_metadata had relied on
    # them it could not have answered the assertions above.
    with pytest.raises(ChunkReadError):
        DcpReader(tmp_path).read_full("plain_w")


def test_read_metadata_describes_safetensors_layout_without_tensor_payloads(
    tmp_path: Path, torch_mod: Any
) -> None:
    """The same metadata-only guarantee must hold for the safetensors format.

    Byte identity for safetensors is derivable from (file, data_offsets) in the
    header, so storage_id must be present and must differ between the two
    tensors — on the same 'same bytes, same id / different bytes, different id'
    rule the incident gate depends on. The payload is truncated after the header
    was written: any implementation that reads tensor data at metadata time
    cannot answer these assertions. assert torch_mod covers the convention; safetensors
    is skipped only if the optional dependency is absent.
    """
    pytest.importorskip("safetensors")
    assert torch_mod is not None
    shard = tmp_path / "model.safetensors"
    payload_at = _write_two_tensor_safetensors(shard)
    shard.write_bytes(shard.read_bytes()[:payload_at])
    assert shard.stat().st_size == payload_at

    meta = fsckpt.read_metadata(os.fspath(tmp_path))

    assert meta.format == "safetensors"
    assert set(meta.tensors) == {"a", "b"}
    assert tuple(meta.tensors["a"].shape) == (2,)
    assert meta.tensors["a"].dtype == "float32"
    assert meta.tensors["a"].storage_id is not None
    assert meta.tensors["b"].storage_id is not None
    assert meta.tensors["a"].storage_id != meta.tensors["b"].storage_id

    # Control: the payload bytes really are gone. The reader DOES wrap this — a
    # truncated shard surfaces as CheckpointFormatError — so the control can name the
    # package's own base error rather than catching anything at all. A blind
    # `Exception` would also pass if the call raised TypeError from a signature
    # change, which would prove nothing about the payload being unreadable.
    with pytest.raises(CheckpointError):
        SafetensorsReader(tmp_path).read_full("a")
