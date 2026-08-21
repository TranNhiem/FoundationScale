"""Acceptance tests for the checkpoint-denominator BLOCKERs (W-ckpt-1, F-ckpt-1).

W-ckpt-1: the byte-volume gate priced metadata-implied bytes per FQN, so 128
count-correct, shape-correct FQNs aliased to one physical tensor scored ratio
1.000 and every checkpoint gate went green on the count-correct variant of the
founding incident. The distinctness gate's FQN fallback simultaneously printed
"every shard group occupies distinct storage" on metadata that carried no
storage identity at all.

F-ckpt-1: every denominator the gates adjudicate was read off manifest
attributes no shipped code produced, so real checkpoints landed in SKIP/VACUOUS
forever — and a real ``RunManifest`` without flat attributes loaded
``declared_fqns=()``, which ``SaveCompletenessGate`` reported as "all 0
declared tensors present".

Every test in this module names, in its docstring, the single assertion that
fails against the code as given.
"""

from __future__ import annotations

import json
import sys
import types
from dataclasses import replace

import pytest

from foundationscale.gates import checkpoint_gates as ckpt_gates
from foundationscale.gates import fixtures as fx
from foundationscale.gates.checkpoint_gates import (
    CheckpointGateContext,
    TensorMeta,
)
from foundationscale.gates.core import REGISTRY, Verdict
from foundationscale.provenance import manifest as prov
from foundationscale.provenance.manifest import (
    CapturedEnvironment,
    CaptureStatus,
    CodeProvenance,
    ManifestError,
    RunManifest,
    Topology,
)

_SHARD_SHAPE = (2, 3)  # bfloat16 expert shard: 2*3*2 = 12 implied bytes
_NUM_ALIASED = 128
_DECLARED_ALIASED_BYTES = _NUM_ALIASED * 2 * 3 * 2  # 1536


def _aliased_shard_ctx(*, with_storage_ids: bool) -> CheckpointGateContext:
    """128 right-shaped expert shard FQNs, all on one storage (or none)."""
    tensors = tuple(
        TensorMeta(
            fqn=f"model.layers.0.mlp.experts.linear_fc1.weight{i}",
            shape=_SHARD_SHAPE,
            dtype="bfloat16",
            storage_id="S" if with_storage_ids else None,
        )
        for i in range(_NUM_ALIASED)
    )
    return CheckpointGateContext(
        tensors=tensors,
        declared_fqns=tuple(t.fqn for t in tensors),
        num_experts=_NUM_ALIASED,
        num_moe_layers=1,
        expected_expert_bytes=_DECLARED_ALIASED_BYTES,
        origin="synthetic:count-correct-128",
    )


def _minimal_manifest(declared: object = None) -> RunManifest:
    return RunManifest(
        run_id="run-1",
        attempt=1,
        code=CodeProvenance(
            status=CaptureStatus.CLEAN,
            root="/repo",
            commit="0" * 40,
            dirty_files=0,
            untracked_files=0,
            diff_sha256=None,
            diff_bytes=0,
            paths=(),
        ),
        config={},
        environment=CapturedEnvironment(allowlist=("SLURM_",), values={}, source_var_count=3),
        topology=Topology(
            nodes=1,
            gpus_per_node=8,
            tensor_parallel=1,
            pipeline_parallel=1,
            data_parallel=8,
        ),
        created_at="2024-01-01T00:00:00+00:00",
        declared=declared,  # type: ignore[arg-type]  # TypeError today: the field does not exist
    )


def _install_fake_checkpoint(
    monkeypatch: pytest.MonkeyPatch,
    *,
    tensors: dict[str, object],
    manifest: object,
) -> None:
    fake = types.ModuleType("foundationscale.checkpoint")
    fake.read_metadata = lambda path: types.SimpleNamespace(tensors=tensors)  # type: ignore[attr-defined]
    fake.load_manifest = lambda path: manifest  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "foundationscale.checkpoint", fake)
    # The sys.modules entry alone is not enough. ``from foundationscale import
    # checkpoint`` resolves the attribute on the parent package first and only
    # falls back to sys.modules when it is absent -- so once any earlier test
    # has imported the real submodule, the parent holds it and the fake is
    # never seen. Passes alone, fails in a full run; bind both.
    import foundationscale

    monkeypatch.setattr(foundationscale, "checkpoint", fake, raising=False)


# ---------------------------------------------------------------------------
# W-ckpt-1: physical storage pricing
# ---------------------------------------------------------------------------


def test_byte_gate_prices_distinct_storage() -> None:
    """Count-correct aliasing must FAIL the byte gate on physical bytes.

    Assertion that fails today: ``result.verdict is Verdict.FAIL`` — the gate
    sums 128 per-FQN implied sizes (128*12 = 1536) against a declared 1536,
    ratio 1.000, and returns PASS.
    """
    result = REGISTRY.get("checkpoint.expert_bytes").run(_aliased_shard_ctx(with_storage_ids=True))
    assert result.verdict is Verdict.FAIL, result.detail
    assert "distinct storage" in result.detail
    assert result.evidence["implied_expert_bytes"] == _DECLARED_ALIASED_BYTES
    assert result.evidence["physical_expert_bytes"] == 2 * 3 * 2
    assert result.evidence["expected_expert_bytes"] == _DECLARED_ALIASED_BYTES


def test_distinct_storage_bytes_counts_each_storage_once() -> None:
    """Helper contract: one implied size per distinct storage key; completeness flag.

    Assertion that fails today: ``ckpt_gates._distinct_storage_bytes`` does not
    exist (AttributeError at the first call).
    """
    experts = [
        TensorMeta("m.experts.linear_fc1.weight0", (2, 2), "float32", storage_id="a"),
        TensorMeta("m.experts.linear_fc1.weight1", (2, 2), "float32", storage_id="b"),
        TensorMeta("m.experts.linear_fc1.weight2", (2, 2), "float32", storage_id="a"),
        TensorMeta("m.experts.linear_fc1.weight3", (2, 2), "float32", storage_id=None),
    ]
    physical, complete = ckpt_gates._distinct_storage_bytes(experts)
    assert physical == 3 * 16  # a, b, plus the FQN-keyed id-less tensor
    assert complete is False

    keyed = [
        TensorMeta("m.experts.linear_fc1.weight0", (2, 2), "float32", storage_id="a"),
        TensorMeta("m.experts.linear_fc1.weight1", (2, 2), "float32", storage_id="b"),
        TensorMeta("m.experts.linear_fc1.weight2", (2, 2), "float32", storage_id="a"),
        TensorMeta("m.experts.linear_fc1.weight3", (2, 2), "float32", storage_id="b"),
    ]
    physical, complete = ckpt_gates._distinct_storage_bytes(keyed)
    assert physical == 2 * 16
    assert complete is True


def test_distinctness_claim_degrades_without_storage_ids() -> None:
    """Without storage identity the PASS may not claim distinct storage.

    Assertion that fails today: ``"distinct storage" not in result.detail`` —
    the sharded-pass branch prints "every shard group occupies distinct storage
    (sharded)" on FQN uniqueness alone, a claim broader than its evidence.
    """
    result = REGISTRY.get("checkpoint.expert_distinctness").run(
        _aliased_shard_ctx(with_storage_ids=False)
    )
    assert result.verdict is Verdict.PASS, result.detail
    assert "distinct storage" not in result.detail
    assert result.coverage.sampled is True
    assert result.coverage.sample_reason == "no storage identity"


def test_distinctness_still_fires_on_shared_storage() -> None:
    """Regression guard: capturable aliasing (storage ids present) still FAILs."""
    result = REGISTRY.get("checkpoint.expert_distinctness").run(
        _aliased_shard_ctx(with_storage_ids=True)
    )
    assert result.verdict is Verdict.FAIL, result.detail


def test_byte_gate_pass_is_labeled_when_metadata_only() -> None:
    """A metadata-implied PASS must be labelled as such, not stated flatly.

    Assertion that fails today: ``"metadata-implied only" in result.detail`` —
    the success message today is the unqualified "expert byte volume 192
    matches declared 192".
    """
    fqns = (
        "model.layers.0.mlp.experts.linear_fc1.weight",
        "model.layers.0.mlp.experts.linear_fc2.weight",
    )
    ctx = CheckpointGateContext(
        tensors=tuple(TensorMeta(fqn, (8, 2, 3), "bfloat16", storage_id=None) for fqn in fqns),
        declared_fqns=fqns,
        num_experts=8,
        num_moe_layers=1,
        expected_expert_bytes=2 * 8 * 2 * 3 * 2,
        origin="synthetic:metadata-only-fused",
    )
    result = REGISTRY.get("checkpoint.expert_bytes").run(ctx)
    assert result.verdict is Verdict.PASS, result.detail
    assert "metadata-implied only" in result.detail
    assert result.evidence["physical_expert_bytes"] is None
    assert result.evidence["storage_identity"] == "absent-or-partial"


def test_implied_nbytes_renamed() -> None:
    """The per-FQN byte figure is metadata-implied; the API must say so.

    Assertion that fails today: ``not hasattr(TensorMeta, "nbytes")`` — the
    property still exists under the name that launders implied bytes into
    "bytes".
    """
    assert hasattr(TensorMeta, "implied_nbytes")
    assert not hasattr(TensorMeta, "nbytes")
    assert TensorMeta("w.weight", (2, 3), "float32").implied_nbytes == 24


def test_zero_declared_byte_volume_blocks() -> None:
    """A non-positive declared volume is malformed, not "matched".

    Assertion that fails today: ``result.verdict is Verdict.VACUOUS`` — the gate
    returns PASS with detail "expert byte volume … matches declared 0".
    """
    ctx = replace(fx.healthy_fused_moe_ctx(), expected_expert_bytes=0)
    result = REGISTRY.get("checkpoint.expert_bytes").run(ctx)
    assert result.verdict is Verdict.VACUOUS, result.detail
    assert result.blocking
    assert "not positive" in result.detail


# ---------------------------------------------------------------------------
# F-ckpt-1: the declared block, its producers, and the manifest->gate join
# ---------------------------------------------------------------------------


def test_declared_block_round_trip_and_wiring(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    """DeclaredCheckpoint canonicalizes, serializes, and reaches the gates.

    Assertions that fail today: ``prov.DeclaredCheckpoint(...)`` raises
    AttributeError; and (for a reader that stubs the class) ``ctx.num_experts
    == 128`` — ``from_path`` reads no ``declared`` block today.
    """
    block = prov.DeclaredCheckpoint(
        num_experts=128,
        num_moe_layers=1,
        expected_expert_bytes=_DECLARED_ALIASED_BYTES,
        declared_fqns=("b.weight1", "a.weight1", "a.weight1"),
        dtype_widths={"bfloat16": 2},
    )
    assert block.declared_fqns == ("a.weight1", "b.weight1")  # canonical wire form

    doc = _minimal_manifest(declared=block).to_dict()
    loaded = RunManifest.from_dict(doc)
    assert loaded.declared == block
    assert loaded.declared is not None
    assert loaded.declared.num_experts == 128
    assert loaded.declared.dtype_widths == {"bfloat16": 2}

    # v1 documents that predate the block load with declared=None; the
    # schema-version refusal elsewhere is untouched (SUPPORTED stays (1,)).
    legacy_doc = _minimal_manifest(declared=block).to_dict()
    del legacy_doc["declared"]
    legacy = RunManifest.from_dict(legacy_doc)
    assert legacy.declared is None

    # The block names verification targets, not computation: fingerprints hold.
    assert (
        _minimal_manifest(declared=None).fingerprint()
        == _minimal_manifest(declared=block).fingerprint()
    )

    # The manifest->gate join: from_path adopts the block's four denominators.
    fqns = (
        "model.layers.0.mlp.experts.linear_fc1.weight",
        "model.layers.0.mlp.experts.linear_fc2.weight",
    )
    joined = prov.DeclaredCheckpoint(
        num_experts=128,
        num_moe_layers=1,
        expected_expert_bytes=2 * 128 * 2 * 3 * 2,
        declared_fqns=fqns,
        dtype_widths={"bfloat16": 2},
    )
    _install_fake_checkpoint(
        monkeypatch,
        tensors={
            fqn: types.SimpleNamespace(
                shape=(128, 2, 3), dtype="bfloat16", storage_id=f"storage:{fqn}"
            )
            for fqn in fqns
        },
        manifest=_minimal_manifest(declared=joined),
    )
    ctx = CheckpointGateContext.from_path(tmp_path)
    assert ctx.num_experts == 128
    assert ctx.num_moe_layers == 1
    assert ctx.expected_expert_bytes == 2 * 128 * 2 * 3 * 2
    assert ctx.declared_fqns == fqns
    assert ctx.expert_storage_bytes == 2 * 128 * 2 * 3 * 2

    byte_result = REGISTRY.get("checkpoint.expert_bytes").run(ctx)
    assert byte_result.verdict is Verdict.PASS, byte_result.detail  # adjudicating, not SKIP
    assert byte_result.coverage.checked == 2
    assert byte_result.coverage.expected == 2


def test_declared_block_refuses_unknown_keys() -> None:
    """A misspelled denominator must fail at the boundary, not load as None.

    Assertion that fails today: ``prov.DeclaredCheckpoint`` and the
    ``RunManifest.declared`` field do not exist, so there is nothing that could
    refuse "num_expert" (sic).
    """
    doc = _minimal_manifest(declared=prov.DeclaredCheckpoint(num_experts=128)).to_dict()
    doc["declared"]["num_expert"] = 64  # type: ignore[index]  # sic
    with pytest.raises(ManifestError, match="unknown declared"):
        RunManifest.from_dict(doc)


def test_declared_block_present_but_not_a_mapping_refuses() -> None:
    """A scalar ``declared`` block raises, never coerces."""
    doc = _minimal_manifest(declared=prov.DeclaredCheckpoint(num_experts=128)).to_dict()
    doc["declared"] = 42
    with pytest.raises(ManifestError, match="declared"):
        RunManifest.from_dict(doc)


def test_declared_from_hf_config(tmp_path) -> None:
    """HF producer: routed-expert count, layer depth, dtype widths.

    Assertion that fails today: ``prov.declared_from_hf_config`` raises
    AttributeError.
    """
    config = tmp_path / "config.json"
    config.write_text(
        json.dumps(
            {
                "num_local_experts": 128,
                "num_hidden_layers": 4,
                "torch_dtype": "bfloat16",
            }
        )
    )
    block = prov.declared_from_hf_config(config)
    assert block.num_experts == 128
    assert block.num_moe_layers == 4
    assert block.dtype_widths == {"bfloat16": 2}
    assert block.naming_convention == "hf-moe"


def test_declared_from_hf_config_refuses_unpriced_dtype(tmp_path) -> None:
    """An unknown dtype is an explicit error, not a guessed width."""
    config = tmp_path / "config.json"
    config.write_text(json.dumps({"num_local_experts": 128, "torch_dtype": "float8_e4m3fn"}))
    with pytest.raises(ValueError, match="float8_e4m3fn"):
        prov.declared_from_hf_config(config)


def test_declared_from_megatron_args() -> None:
    """Megatron producer: resolved args mirrored into the same block.

    Assertion that fails today: ``prov.declared_from_megatron_args`` raises
    AttributeError.
    """
    block = prov.declared_from_megatron_args(
        types.SimpleNamespace(num_experts=128, num_layers=4, bf16=True, fp16=False)
    )
    assert block.num_experts == 128
    assert block.num_moe_layers == 4
    assert block.dtype_widths == {"bfloat16": 2}
    assert block.tensors_per_expert_layer == 2
    assert block.naming_convention == "megatron-core"


def test_capture_state_dict_keys_excludes_extra_state() -> None:
    """The declared FQN list counts tensors, not metadata blobs.

    Assertion that fails today: ``prov.capture_state_dict_keys`` raises
    AttributeError.
    """
    state = {
        "layers.0.mlp.weight": object(),
        "layers.0._extra_state": b"bLOB",
        "layers.0.attn.weight": object(),
    }
    keys = prov.capture_state_dict_keys(state)
    assert keys == ("layers.0.attn.weight", "layers.0.mlp.weight")

    class _Model:
        def state_dict(self) -> dict[str, object]:
            return state

    assert prov.capture_state_dict_keys(_Model()) == keys


def test_manifest_without_denominators_yields_none_not_empty_tuple(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A manifest with no denominators must not fabricate an empty one.

    Assertions that fail today: ``ctx.declared_fqns is None`` — the adapter
    loads ``tuple(getattr(manifest, "declared_fqns", ()))``, i.e. ``()``; and
    ``save_result.verdict is not Verdict.PASS`` — the completeness gate then
    prints "all 0 declared tensors present" with expected=0.
    """
    _install_fake_checkpoint(
        monkeypatch,
        tensors={},
        manifest=types.SimpleNamespace(),  # real-manifest shape: no flat attrs, no block
    )
    ctx = CheckpointGateContext.from_path(tmp_path)
    assert ctx.declared_fqns is None
    assert ctx.num_experts is None
    assert ctx.expected_expert_bytes is None
    save_result = REGISTRY.get("checkpoint.save_complete").run(ctx)
    assert save_result.verdict is not Verdict.PASS, save_result.detail
    assert save_result.blocking


def test_empty_declared_fqns_blocks_completeness() -> None:
    """An explicit empty declaration blocks rather than auto-satisfying.

    Assertion that fails today: ``result.verdict is not Verdict.PASS`` — the
    gate returns "all 0 declared tensors present (excluding 0 _extra_state
    metadata blobs)" as a PASS with coverage 0/0.
    """
    ctx = replace(fx.healthy_fused_moe_ctx(), declared_fqns=())
    result = REGISTRY.get("checkpoint.save_complete").run(ctx)
    assert result.verdict is not Verdict.PASS, result.detail
    assert result.blocking
    assert "empty" in result.detail


def test_legacy_flat_adapter_still_wires_denominators(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Regression guard: the pre-block flat-attribute adapter keeps working."""
    stub = types.SimpleNamespace(
        declared_fqns=("model.layers.0.mlp.experts.linear_fc1.weight",),
        num_experts=8,
        num_moe_layers=1,
        expected_expert_bytes=8 * 2 * 3 * 2,
    )
    _install_fake_checkpoint(
        monkeypatch,
        tensors={
            "model.layers.0.mlp.experts.linear_fc1.weight": types.SimpleNamespace(
                shape=(8, 2, 3), dtype="bfloat16", storage_id="s1"
            )
        },
        manifest=stub,
    )
    ctx = CheckpointGateContext.from_path(tmp_path)
    assert ctx.num_experts == 8
    assert ctx.declared_fqns == ("model.layers.0.mlp.experts.linear_fc1.weight",)
    assert ctx.expert_storage_bytes == 8 * 2 * 3 * 2
