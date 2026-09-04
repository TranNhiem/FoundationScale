"""Fail-before/pass-after pins for the declared-depth launderings (manifest.py)
and the reserved-name || true collapse (dcp_meta.py).

Denominators for this suite
---------------------------
* 9 MUST-FIRE tests: FAIL on the current tree (each asserts a refusal or an
  abstention the current code does not produce) and PASS after the patch.
* 7 MUST-PASS controls: PASS before AND after. Their job is doctrine 3's
  other half: prove the patched producer did not become a raise-on-everything
  bucket, and prove load_manifest did not trade a false pass for a false
  failure on files it does not own.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from foundationscale.checkpoint.dcp import CheckpointFormatError
from foundationscale.checkpoint.dcp_meta import load_manifest
from foundationscale.provenance.manifest import (
    CapturedEnvironment,
    CaptureStatus,
    CodeProvenance,
    RunManifest,
    Topology,
    declared_from_hf_config,
    declared_from_megatron_args,
)

# ---------------------------------------------------------------------------
# Finding 4 — MUST-FIRE: a stated, invalid num_moe_layers is refused, never
# laundered into num_layers' depth (one direction) or into a fake abstention
# (the other). All three FAIL before (no ValueError raised) and PASS after.
# ---------------------------------------------------------------------------


class TestFinding4StatedInvalidDepth:
    def test_stated_zero_num_moe_layers_raises_instead_of_laundering(self):
        # Before: returns num_moe_layers=60 with basis "num_layers" — the
        # contradictory config's 0 erased from both value and provenance.
        args = SimpleNamespace(num_experts=8, num_moe_layers=0, num_layers=60, bf16=True)
        with pytest.raises(ValueError, match="num_moe_layers must be a positive int"):
            declared_from_megatron_args(args)

    def test_stated_zero_without_num_layers_raises_instead_of_fake_abstention(self):
        # Before: `0 or None` -> None, indistinguishable from "never asked".
        args = SimpleNamespace(num_experts=8, num_moe_layers=0, bf16=True)
        with pytest.raises(ValueError, match="num_moe_layers must be a positive int"):
            declared_from_megatron_args(args)

    def test_stated_false_bool_num_moe_layers_raises(self):
        # The `or` cannot tell False from absent either: before, `False or 24`
        # launders to 24. The HF producer refuses bools on the same field.
        args = SimpleNamespace(num_experts=8, num_moe_layers=False, num_layers=24, bf16=True)
        with pytest.raises(ValueError, match="num_moe_layers must be a positive int"):
            declared_from_megatron_args(args)


# ---------------------------------------------------------------------------
# Finding 5 — MUST-FIRE: an unmodelled interleave forces a STATED abstention,
# never a routed-layer count invented from total depth. FAIL before (declares
# 24, basis "num_layers"); PASS after.
# ---------------------------------------------------------------------------


class TestFinding5InterleaveAbstention:
    def test_moe_layer_freq_forces_loud_abstention(self):
        args = SimpleNamespace(num_experts=8, num_layers=24, moe_layer_freq=2, bf16=True)
        declared = declared_from_megatron_args(args)
        assert declared.num_experts == 8
        assert declared.num_moe_layers is None, (
            "an interleave this producer does not model must yield no denominator, "
            f"got the fabricated depth {declared.num_moe_layers}"
        )
        assert declared.moe_layer_basis is not None
        assert declared.moe_layer_basis.startswith("abstained:")
        assert "moe_layer_freq" in declared.moe_layer_basis

    def test_megatron_abstention_matches_hf_verdict_for_identical_facts(self):
        # One set of gates consumes both producers' output, so identical facts
        # must yield the identical verdict class. Before: HF says None while
        # Megatron says 24 — this assertion is the parity pin and FAILS there.
        hf = declared_from_hf_config(
            {
                "num_experts": 8,
                "num_hidden_layers": 24,
                "moe_layer_freq": 2,
                "torch_dtype": "bfloat16",
            }
        )
        meg = declared_from_megatron_args(
            SimpleNamespace(num_experts=8, num_layers=24, moe_layer_freq=2, bf16=True)
        )
        assert hf.num_moe_layers is None
        assert meg.num_moe_layers == hf.num_moe_layers
        assert hf.moe_layer_basis is not None and meg.moe_layer_basis is not None
        # Shared wording after "declares " pins that the discipline, not merely
        # the outcome, was mirrored rather than reinvented.
        assert (
            hf.moe_layer_basis.split("declares ", 1)[1]
            == meg.moe_layer_basis.split("declares ", 1)[1]
        )


# ---------------------------------------------------------------------------
# MUST-PASS controls, producer side: healthy resolutions must keep resolving.
# All three PASS before AND after; a failure after the patch would mean the
# refusal/abstention machinery now fires on inputs it must not touch.
# ---------------------------------------------------------------------------


class TestMegatronDepthControls:
    def test_explicit_num_moe_layers_overrides_interleave_abstention(self):
        # An operator who knows the depth says so explicitly — mirrored from
        # HF, where an explicit num_moe_layers bypasses the sparsity check.
        args = SimpleNamespace(
            num_experts=8, num_moe_layers=4, num_layers=24, moe_layer_freq=2, bf16=True
        )
        declared = declared_from_megatron_args(args)
        assert declared.num_moe_layers == 4
        assert declared.moe_layer_basis == "num_moe_layers"

    def test_all_routed_depth_still_derives_from_num_layers(self):
        # No interleave keys anywhere: num_layers remains the all-routed depth.
        args = SimpleNamespace(num_experts=8, num_layers=24, bf16=True)
        declared = declared_from_megatron_args(args)
        assert declared.num_moe_layers == 24
        assert declared.moe_layer_basis == "num_layers"

    def test_dense_args_carry_no_depth_and_no_abstention(self):
        # A dense model with a stray interleave default must not abstain:
        # depth logic runs only when experts exist (mirrors the HF producer).
        args = SimpleNamespace(num_layers=24, moe_layer_freq=2, bf16=True)
        declared = declared_from_megatron_args(args)
        assert declared.num_experts is None
        assert declared.num_moe_layers is None
        assert declared.moe_layer_basis is None


# ---------------------------------------------------------------------------
# Finding 2 — MUST-FIRE: run_manifest.json is this framework's reserved name,
# so its corruption is a finding, not an absence. All four FAIL before
# (load_manifest returns None) and PASS after.
# ---------------------------------------------------------------------------


class TestFinding2ReservedNameStrictness:
    def test_truncated_reserved_manifest_raises(self, tmp_path: Path):
        (tmp_path / "run_manifest.json").write_bytes(b'{"run_id": "x", "co')
        with pytest.raises(CheckpointFormatError, match="reserved run-manifest name"):
            load_manifest(tmp_path)

    def test_undecodable_reserved_manifest_raises(self, tmp_path: Path):
        (tmp_path / "run_manifest.json").write_bytes(b"\xff\xfe{\x00")
        with pytest.raises(CheckpointFormatError, match="reserved run-manifest name"):
            load_manifest(tmp_path)

    def test_partial_reserved_manifest_raises(self, tmp_path: Path):
        # Parses cleanly but lost topology/code/environment to a torn write:
        # partial provenance, not absent provenance.
        (tmp_path / "run_manifest.json").write_text('{"run_id": "x"}', encoding="utf-8")
        with pytest.raises(CheckpointFormatError, match="lacks the required manifest keys"):
            load_manifest(tmp_path)

    def test_reserved_name_non_manifest_json_raises(self, tmp_path: Path):
        (tmp_path / "run_manifest.json").write_text("[1, 2, 3]", encoding="utf-8")
        with pytest.raises(CheckpointFormatError, match="lacks the required manifest keys"):
            load_manifest(tmp_path)


# ---------------------------------------------------------------------------
# MUST-PASS controls, loader side: the states that must NOT change polarity.
# All four PASS before AND after — they pin that the patch's strictness is
# ownership-scoped and did not mint false failures on files it does not own.
# ---------------------------------------------------------------------------


class TestLoadManifestControls:
    def test_absent_manifest_remains_the_normal_case(self, tmp_path: Path):
        assert load_manifest(tmp_path) is None

    def test_unparseable_shared_name_stays_lenient(self, tmp_path: Path):
        # manifest.json is a shared name; other tools legitimately write it.
        # Unparseable under a shared name means "not ours", still -> None.
        (tmp_path / "manifest.json").write_bytes(b"{not json")
        assert load_manifest(tmp_path) is None

    def test_unparseable_attempt_glob_stays_lenient(self, tmp_path: Path):
        # In a checkpoint dir, attempt-*.json is not provably a store record.
        (tmp_path / "attempt-0001.json").write_bytes(b"not json at all")
        assert load_manifest(tmp_path) is None

    def test_valid_reserved_manifest_round_trips(self, tmp_path: Path):
        manifest = RunManifest(
            run_id="audit-run",
            attempt=1,
            code=CodeProvenance(
                status=CaptureStatus.NOT_A_REPOSITORY,
                root=None,
                commit=None,
                dirty_files=0,
                untracked_files=0,
                diff_sha256=None,
                diff_bytes=0,
                paths=(),
            ),
            config={},
            environment=CapturedEnvironment(allowlist=(), values={}, source_var_count=0),
            topology=Topology(
                nodes=1,
                gpus_per_node=8,
                tensor_parallel=1,
                pipeline_parallel=1,
                data_parallel=8,
            ),
        )
        (tmp_path / "run_manifest.json").write_text(manifest.to_json(), encoding="utf-8")
        loaded = load_manifest(tmp_path)
        assert loaded is not None
        assert loaded.run_id == "audit-run"
        assert loaded.attempt == 1
        assert loaded.fingerprint() == manifest.fingerprint()
