"""Regression tests for the dense-declaration repair batch.

Defects covered (audit batch: dense-declaration family):
  1. The probe minted ``num_experts=0`` from ABSENCE of a count key, over a
     drifted two-name key list; ``0`` earns NOT_APPLICABLE and shrinks
     FirstSaveGate's denominator, so a misclassified MoE artifact went 1/1 PASS.
  2. The emitter knew dense (``enable_moe_block=false``) and wrote ``None``,
     blocking every healthy dense first save at 1/3 forever.
  3. The independence guard compared resolved path STRINGS; mount aliases and
     case variants bypassed it into a tautological completeness check.
  4. ``--cp`` was required, consumed, and silently discarded; CP=1 and CP=8
     fingerprinted equal.

Every test fails on the pre-patch tree and passes after. The pre-patch failure
MODE is stated at each test — what the detector-shaped house rules demand.
"""

from __future__ import annotations

import importlib.util
import json
import math
import shutil
import sys
from pathlib import Path
from types import ModuleType

import pytest

from foundationscale.gates.checkpoint_gates import (
    CheckpointGateContext,
    ExpertByteVolumeGate,
    ExpertDistinctnessGate,
    FirstSaveGate,
    TensorMeta,
)
from foundationscale.gates.core import Verdict
from foundationscale.provenance.manifest import (
    CapturedEnvironment,
    CaptureStatus,
    CodeProvenance,
    DeclaredCheckpoint,
    ManifestStore,
    RunManifest,
    Topology,
)

_REPO_ROOT = Path(__file__).resolve().parents[1]


def _load_tool(name: str) -> ModuleType:
    """Load a tools/ script as a module (tools/ is not an importable package)."""
    spec = importlib.util.spec_from_file_location(name, _REPO_ROOT / "tools" / f"{name}.py")
    if spec is None or spec.loader is None:  # pragma: no cover - defensive
        raise RuntimeError(f"cannot load tools/{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


_ST_ENCODINGS: dict[str, tuple[str, int]] = {"bfloat16": ("BF16", 2), "float32": ("F32", 4)}


def _write_safetensors(path: Path, tensors: dict[str, tuple[tuple[int, ...], str]]) -> None:
    """A minimal safetensors file (8-byte LE header length + JSON header + zeros)."""
    header: dict[str, object] = {}
    offset = 0
    for fqn, (shape, dtype) in sorted(tensors.items()):
        code, width = _ST_ENCODINGS[dtype]
        nbytes = math.prod(shape) * width
        header[fqn] = {
            "dtype": code, "shape": list(shape), "data_offsets": [offset, offset + nbytes]
        }
        offset += nbytes
    encoded = json.dumps(header).encode("utf-8")
    path.write_bytes(len(encoded).to_bytes(8, "little") + encoded + b"\0" * offset)


# ---------------------------------------------------------------------------
# Defect 1: the probe must never mint dense from absence (MUST_FIRE family)
# ---------------------------------------------------------------------------


def test_probe_shares_the_library_key_list() -> None:
    """n_routed_experts is MoE to the library, so it is MoE to the probe.

    Fails before: pre-patch derive_declared had no census kwargs (TypeError);
    called its old way it minted 0 here because its private list lacked
    ``n_routed_experts``.
    """
    probe = _load_tool("real_checkpoint_probe")
    declared = probe.derive_declared(
        {"model_type": "deepseek_v3", "text_config": {"n_routed_experts": 8}},
        expert_family_census=16,
        expert_family_sample=("model.layers.3.mlp.experts.0.gate_proj.weight",),
    )
    assert declared["num_experts"] == 8
    assert "dense" not in declared["basis"]["num_experts"].lower()


def test_probe_absence_stays_unknown() -> None:
    """A config that says nothing must not be told it said dense.

    Fails before: pre-patch code path minted num_experts=0 from exactly this
    input shape (and the census kwargs did not exist).
    """
    probe = _load_tool("real_checkpoint_probe")
    declared = probe.derive_declared(
        {"model_type": "mystery_arch"},
        expert_family_census=12,
        expert_family_sample=("model.layers.0.block_sparse_moe.experts.0.w1.weight",),
    )
    assert declared["num_experts"] is None
    basis = declared["basis"]["num_experts"].lower()
    assert "stays none" in basis
    assert "not declaring dense" in basis


def test_probe_dense_flag_against_expert_census_is_a_contradiction() -> None:
    """Config says dense, artifact holds experts: stated, never adjudicated.

    Fails before: pre-patch minted 0 (the flag was never even read).
    """
    probe = _load_tool("real_checkpoint_probe")
    declared = probe.derive_declared(
        {"text_config": {"enable_moe_block": False}},
        expert_family_census=5,
        expert_family_sample=("model.layers.0.mlp.experts.0.gate_proj.weight",),
    )
    assert declared["num_experts"] is None
    assert "contradiction" in declared["basis"]["num_experts"].lower()


def test_probe_moe_flag_without_an_understood_count_stays_unknown() -> None:
    """enable_moe_block=true affirms MoE; a missing count is not a dense license.

    Fails before: flag unread, absence minted to 0.
    """
    probe = _load_tool("real_checkpoint_probe")
    declared = probe.derive_declared(
        {"text_config": {"enable_moe_block": True}},
        expert_family_census=4,
        expert_family_sample=("model.layers.0.mlp.experts.14.w2.weight",),
    )
    assert declared["num_experts"] is None
    assert "affirm" in declared["basis"]["num_experts"].lower()


def test_probe_corroborated_dense_mints_zero_with_its_evidence() -> None:
    """The mint rule's positive side: affirmative flag + zero-census -> 0.

    Fails before: census kwargs did not exist (TypeError); also the basis
    named no census.
    """
    probe = _load_tool("real_checkpoint_probe")
    declared = probe.derive_declared(
        {"text_config": {"enable_moe_block": False, "num_experts": None}},
        expert_family_census=0,
    )
    assert declared["num_experts"] == 0
    basis = declared["basis"]["num_experts"].lower()
    assert "corroborated" in basis
    assert "enable_moe_block" in basis
    assert "0 expert-family tensors" in basis


def test_probe_derived_moe_context_reaches_a_real_distinctness_verdict() -> None:
    """End to end at gate level: a correctly derived MoE count is EXAMINED, not skipped.

    Fails before: pre-patch derive minted 0 for this config, and the gate then
    answered the dense-model SKIP instead of PASSing the examined sharded set.
    """
    probe = _load_tool("real_checkpoint_probe")
    declared = probe.derive_declared(
        {"text_config": {"n_routed_experts": 2}},
        expert_family_census=2,
        expert_family_sample=("model.layers.0.mlp.experts.linear_fc1.weight0",),
    )
    tensors = tuple(
        TensorMeta(
            fqn=f"model.layers.0.mlp.experts.linear_fc1.weight{i}",
            shape=(4, 4),
            dtype="bfloat16",
            storage_id=f"storage-{i}",
        )
        for i in range(2)
    )
    ctx = CheckpointGateContext(
        tensors=tensors,
        declared_fqns=None,
        num_experts=declared["num_experts"],
        num_moe_layers=None,
        expected_expert_bytes=None,
        origin="synthetic:probe-derived-moe",
    )
    result = ExpertDistinctnessGate().run(ctx)
    assert result.verdict is Verdict.PASS


# ---------------------------------------------------------------------------
# Defect 1d: bool/float/complex look-alikes of 0 must not buy the denominator
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("gate_cls", [ExpertDistinctnessGate, ExpertByteVolumeGate])
@pytest.mark.parametrize("bad", [False, 0.0, 0j], ids=["bool", "float", "complex"])
def test_gates_block_malformed_num_experts(gate_cls: type, bad: object) -> None:
    """`== 0` is True for all three of these; none is a dense declaration.

    Fails before: every parametrization took the dense-model SKIP
    (non-blocking). Passes after as a blocking VACUOUS that names the type.
    """
    ctx = CheckpointGateContext(
        tensors=(),
        declared_fqns=None,
        num_experts=bad,  # the defect: an untyped integrator glue value
        num_moe_layers=None,
        expected_expert_bytes=None,
        origin="synthetic:malformed-num-experts",
    )
    result = gate_cls().run(ctx)
    assert result.blocking, f"{gate_cls.id} must not SKIP on num_experts={bad!r}"
    assert result.verdict is Verdict.VACUOUS
    assert "not a genuine non-negative integer" in result.detail


# ---------------------------------------------------------------------------
# Manifest layer: 0 is admitted only as a recorded positive declaration
# ---------------------------------------------------------------------------


def test_declared_zero_requires_recorded_basis() -> None:
    """A bare num_experts=0 is refused WITH THE BASIS NAMED in the message.

    Fails before: the old message was "must be a positive int or None" and
    never mentioned the basis requirement.
    """
    with pytest.raises(ValueError, match="basis"):
        DeclaredCheckpoint(num_experts=0)


def test_declared_zero_with_basis_round_trips_and_first_save_goes_one_of_one() -> None:
    """The Defect-2 positive path, manifest through composite.

    Fails before: DeclaredCheckpoint(num_experts=0, ...) raised (0 was
    invalid), so the healthy dense 1/1 shape was unreachable by construction.
    """
    declared = DeclaredCheckpoint(
        num_experts=0,
        declared_fqns=("model.layers.0.self_attn.q_proj.weight",),
        moe_layer_basis=(
            "dense: enable_moe_block=false AND base census 0 expert-family tensors"
        ),
    )
    assert DeclaredCheckpoint.from_dict(declared.to_dict()).num_experts == 0
    ctx = CheckpointGateContext(
        tensors=(
            TensorMeta(
                fqn="model.layers.0.self_attn.q_proj.weight",
                shape=(4, 4),
                dtype="bfloat16",
            ),
        ),
        declared_fqns=declared.declared_fqns,
        num_experts=declared.num_experts,
        num_moe_layers=declared.num_moe_layers,
        expected_expert_bytes=None,
        origin="synthetic:declared-dense",
    )
    result = FirstSaveGate().run(ctx)
    assert result.verdict is Verdict.PASS
    assert "1/1" in result.detail
    assert "checkpoint.expert_distinctness" in result.detail


# ---------------------------------------------------------------------------
# Defect 4: --cp is recorded and splits the fingerprint
# ---------------------------------------------------------------------------


def test_context_parallel_is_recorded_and_fingerprinted() -> None:
    """CP=1 and CP=8 must not be the same computation.

    Fails before: Topology(context_parallel=...) raised TypeError — the knob
    existed only on the launcher argv.
    """
    code = CodeProvenance(
        status=CaptureStatus.NOT_A_REPOSITORY,
        root=None,
        commit=None,
        dirty_files=0,
        untracked_files=0,
        diff_sha256=None,
        diff_bytes=0,
        paths=(),
    )
    env = CapturedEnvironment(allowlist=(), values={}, source_var_count=0)

    def _manifest(cp: int) -> RunManifest:
        return RunManifest(
            run_id="cp-split",
            attempt=1,
            code=code,
            config={},
            environment=env,
            topology=Topology(
                nodes=1,
                gpus_per_node=8,
                tensor_parallel=1,
                pipeline_parallel=1,
                data_parallel=8,
                context_parallel=cp,
            ),
        )

    assert _manifest(1).fingerprint() != _manifest(8).fingerprint()
    record = _manifest(8)
    assert record.to_dict()["topology"]["context_parallel"] == 8  # type: ignore[index]
    reloaded = Topology.from_dict(record.topology.to_dict())
    assert reloaded.context_parallel == 8
    legacy = Topology.from_dict(
        {
            "nodes": 1,
            "gpus_per_node": 8,
            "tensor_parallel": 1,
            "pipeline_parallel": 1,
            "data_parallel": 8,
            "expert_parallel": None,
        }
    )
    assert legacy.context_parallel is None  # "never recorded", never backfilled


# ---------------------------------------------------------------------------
# Defect 2: emitter mints corroborated dense, refuses contradiction (chain tests)
# ---------------------------------------------------------------------------


def _emit_args(run_id: str, out_dir: Path, ckpt_dir: Path, base: Path, config: Path) -> list[str]:
    return [
        "--run-id",
        run_id,
        "--out-dir",
        str(out_dir),
        "--checkpoint-dir",
        str(ckpt_dir),
        "--nodes",
        "1",
        "--gpus-per-node",
        "8",
        "--tp",
        "1",
        "--pp",
        "1",
        "--cp",
        "2",
        "--dp",
        "8",
        "--full-ft",
        "--base-checkpoint",
        str(base),
        "--hf-config",
        str(config),
    ]


def test_emit_dense_chain_reaches_first_save_one_of_one(tmp_path: Path) -> None:
    """The audit's measured 1/3 shape inverted: healthy dense first save PASSES.

    Fails before: the emitter wrote num_experts=None, so from_path handed the
    gates UNKNOWN and FirstSave blocked 1/3 (VACUOUS expert sub-gates).
    """
    emit = _load_tool("emit_run_manifest")
    base_dir = tmp_path / "converted-base"
    base_dir.mkdir()
    base_tensors: dict[str, tuple[tuple[int, ...], str]] = {
        "model.embed_tokens.weight": ((8, 4), "bfloat16"),
        "model.layers.0.self_attn.q_proj.weight": ((4, 4), "bfloat16"),
    }
    _write_safetensors(base_dir / "model.safetensors", base_tensors)
    config_path = tmp_path / "config.json"
    config_path.write_text(
        json.dumps(
            {
                "model_type": "gemma4-test-dense",
                "text_config": {"enable_moe_block": False, "num_experts": None},
            }
        ),
        encoding="utf-8",
    )
    out_dir = tmp_path / "run-root"
    out_dir.mkdir()
    ckpt_dir = out_dir / "ckpts"
    rc = emit.main(_emit_args("dense-chain", out_dir, ckpt_dir, base_dir, config_path))
    assert rc == 0

    manifest = ManifestStore(out_dir / "provenance").latest("dense-chain")
    assert manifest.declared is not None
    assert manifest.declared.num_experts == 0
    assert "census" in (manifest.declared.moe_layer_basis or "")

    # Simulate the first save: the judged artifact mirrors the emitted census.
    iter_dir = ckpt_dir / "iter_0000001"
    shutil.copytree(base_dir, iter_dir)
    ctx = CheckpointGateContext.from_path(str(iter_dir))
    result = FirstSaveGate().run(ctx)
    assert result.verdict is Verdict.PASS
    assert "1/1" in result.detail
    assert "checkpoint.expert_distinctness" in result.detail


def test_emit_refuses_dense_flag_against_expert_base(tmp_path: Path) -> None:
    """Affirmative dense + expert-bearing base census = refusal, never a winner.

    Fails before: pre-patch emitted dense-as-None with rc=0 over exactly this
    input pair (the census half did not exist).
    """
    emit = _load_tool("emit_run_manifest")
    base_dir = tmp_path / "converted-moe-base"
    base_dir.mkdir()
    _write_safetensors(
        base_dir / "model.safetensors",
        {
            "model.layers.0.mlp.experts.0.gate_proj.weight": ((4, 4), "bfloat16"),
            "model.layers.0.self_attn.q_proj.weight": ((4, 4), "bfloat16"),
        },
    )
    config_path = tmp_path / "config.json"
    config_path.write_text(
        json.dumps({"text_config": {"enable_moe_block": False}}), encoding="utf-8"
    )
    out_dir = tmp_path / "run-root"
    out_dir.mkdir()
    rc = emit.main(
        _emit_args("contradiction", out_dir, out_dir / "ckpts", base_dir, config_path)
    )
    assert rc == emit.EXIT_REFUSED


def test_emit_real_moe_keeps_its_denominator_and_reports_the_census(tmp_path: Path) -> None:
    """Affirmed MoE with an expert-bearing base: count kept, denominator stays /3.

    Fails before: the returned info carried no census key (KeyError), and no
    part of the emission corroborated the artifact.
    """
    emit = _load_tool("emit_run_manifest")
    base_dir = tmp_path / "converted-moe"
    base_dir.mkdir()
    _write_safetensors(
        base_dir / "model.safetensors",
        {
            "model.layers.0.mlp.experts.0.w1.weight": ((4, 4), "bfloat16"),
            "model.layers.0.mlp.experts.1.w1.weight": ((4, 4), "bfloat16"),
            "model.layers.0.mlp.experts.0.w2.weight": ((4, 4), "bfloat16"),
            "model.layers.0.mlp.experts.1.w2.weight": ((4, 4), "bfloat16"),
        },
    )
    config_path = tmp_path / "config.json"
    config_path.write_text(
        json.dumps(
            {
                "text_config": {
                    "enable_moe_block": True,
                    "num_experts": 128,
                    "num_hidden_layers": 2,
                }
            }
        ),
        encoding="utf-8",
    )
    judged = tmp_path / "snapshots"
    judged.mkdir()
    declared, info = emit.derive_declared_full_ft(base_dir, judged, config_path)
    assert declared.num_experts == 128
    assert info["expert_family_census"] == 4

    ctx = CheckpointGateContext(
        tensors=(),  # nothing written yet: the composite must block, /3 intact
        declared_fqns=declared.declared_fqns,
        num_experts=declared.num_experts,
        num_moe_layers=declared.num_moe_layers,
        expected_expert_bytes=declared.expected_expert_bytes,
        origin="synthetic:real-moe-pre-save",
    )
    result = FirstSaveGate().run(ctx)
    assert result.blocking
    assert result.coverage.expected == 3  # the shrink must NOT fire for real MoE


# ---------------------------------------------------------------------------
# Defect 3: independence cannot be assumed
# ---------------------------------------------------------------------------


def test_independence_guard_refuses_when_it_cannot_establish(tmp_path: Path) -> None:
    """An unstat-able judged tree is REFUSED, not allowed by assumption.

    Fails before: the string-only guard allowed this silently (samefile
    raised, resolved strings differed, and the audit's tautology walked in).
    """
    emit = _load_tool("emit_run_manifest")
    base = tmp_path / "base"
    base.mkdir()
    judged = tmp_path / "run" / "ckpts"  # never created: cannot be established
    with pytest.raises(emit.EmitRefused, match="cannot establish independence"):
        emit.ensure_declaration_is_independent(base, judged)
