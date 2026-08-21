"""Tests for the verified review findings against ``provenance.manifest``.

Each test names the defect it pins in a WHY comment. Per the review protocol
every behaviour change is covered by a test that fails against the unfixed
tree; where a test instead guards the *fix's* own correctness, that is stated
in the comment.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from foundationscale.provenance.manifest import (
    CapturedEnvironment,
    CaptureStatus,
    CodeProvenance,
    DeclaredCheckpoint,
    ManifestError,
    RunManifest,
    Topology,
    _hermetic_git_env,
    declared_from_hf_config,
    load,
)


def test_first_k_dense_replace_depth_is_derived_not_fabricated() -> None:
    # WHY: DeepSeek-V3-shaped config — 61 hidden layers, the first 3 dense. The
    # unfixed producer declared num_moe_layers=61 ("the HF-MoE default") against
    # a truth of 58: a manufactured denominator handed to gates that adjudicate
    # against it.
    declared = declared_from_hf_config(
        {
            "n_routed_experts": 256,
            "num_hidden_layers": 61,
            "first_k_dense_replace": 3,
            "torch_dtype": "bfloat16",
        }
    )
    assert declared.num_experts == 256
    assert declared.num_moe_layers == 61 - 3
    # The basis must be recorded so a reader can audit where 58 came from.
    assert declared.moe_layer_basis is not None
    assert "first_k_dense_replace" in declared.moe_layer_basis


def test_unmodeled_sparsity_keys_force_abstention() -> None:
    # WHY: decoder_sparse_step announces a sparse/dense interleave this producer
    # does not model. An absent denominator makes the gate abstain loudly; the
    # unfixed producer instead invented 28 and let the gate lie quietly.
    declared = declared_from_hf_config(
        {"num_local_experts": 64, "num_hidden_layers": 28, "decoder_sparse_step": 1}
    )
    assert declared.num_experts == 64
    assert declared.num_moe_layers is None


def test_text_config_nested_moe_resolves_expert_count_and_depth() -> None:
    # WHY: measured on the real Gemma-4 26B-A4B config.json (48 GiB on disk): no
    # expert key at the top level, text_config.num_experts == 128. The
    # nesting-blind producer returned a *dense* declaration for that MoE and
    # silently disarmed every expert gate.
    declared = declared_from_hf_config(
        {
            "architectures": ["Gemma4ForConditionalGeneration"],
            "model_type": "gemma4",
            "dtype": "bfloat16",
            "text_config": {
                "model_type": "gemma4_text",
                "num_experts": 128,
                "num_hidden_layers": 30,
            },
        }
    )
    assert declared.num_experts == 128
    assert declared.num_moe_layers == 30
    assert declared.dtype_widths == {"bfloat16": 2}


def test_text_config_explicit_null_stays_dense() -> None:
    # WHY: an explicit null is "no routed experts", not an expert count. This
    # guards the nesting fix itself — reading text_config must not trip over
    # the null or hallucinate scope. (Against the unfixed tree this fails on
    # the absent moe_layer_basis attribute; the dense outcome it asserts is the
    # behaviour a naive fix most easily breaks.)
    declared = declared_from_hf_config(
        {
            "text_config": {
                "model_type": "dense_lm",
                "num_experts": None,
                "num_hidden_layers": 30,
            }
        }
    )
    assert declared.num_experts is None
    assert declared.num_moe_layers is None
    assert declared.moe_layer_basis is None


def test_hermetic_git_env_neutralizes_file_config_channels(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # WHY: stripping GIT_* closed only the environment channel; HOME-carried
    # system/global gitconfig (diff drivers, core.autocrlf) still shaped the
    # hashed diff bytes while the docstring claimed full hermeticity.
    monkeypatch.setenv("GIT_DIR", "/tmp/decoy-repo")
    monkeypatch.setenv("GIT_CONFIG_COUNT", "1")
    monkeypatch.setenv("GIT_CONFIG_KEY_0", "core.autocrlf")
    monkeypatch.setenv("GIT_CONFIG_VALUE_0", "true")
    env = _hermetic_git_env()
    assert "GIT_DIR" not in env
    assert "GIT_CONFIG_COUNT" not in env
    assert not [key for key in env if key.startswith("GIT_CONFIG_KEY_")]
    assert env["GIT_OPTIONAL_LOCKS"] == "0"
    assert env["GIT_CONFIG_NOSYSTEM"] == "1"
    assert env["GIT_CONFIG_SYSTEM"] == os.devnull
    assert env["GIT_CONFIG_GLOBAL"] == os.devnull


def _minimal_manifest_with_declared(tmp_path: Path) -> Path:
    manifest = RunManifest(
        run_id="declared-error-contract",
        attempt=1,
        code=CodeProvenance(
            status=CaptureStatus.CLEAN,
            root="/recorded-root",
            commit="0" * 40,
            dirty_files=0,
            untracked_files=0,
            diff_sha256="0" * 64,
            diff_bytes=0,
            paths=(),
        ),
        config={},
        environment=CapturedEnvironment(
            allowlist=("FOUNDATIONSCALE_",), values={}, source_var_count=0
        ),
        topology=Topology(
            nodes=1,
            gpus_per_node=8,
            tensor_parallel=1,
            pipeline_parallel=1,
            data_parallel=1,
        ),
        declared=DeclaredCheckpoint(num_experts=128, num_moe_layers=58),
    )
    path = tmp_path / "manifest.json"
    path.write_text(manifest.to_json(), encoding="utf-8")
    return path


def test_corrupt_declared_raises_manifest_error_through_load(tmp_path: Path) -> None:
    # WHY: load() documents ManifestError for corrupt content, and callers route
    # failures by that type. The unfixed tree let DeclaredCheckpoint's bare
    # ValueError escape from_dict -> load(), uncatchable under the contract.
    path = _minimal_manifest_with_declared(tmp_path)
    data = json.loads(path.read_text(encoding="utf-8"))
    declared_block = data["declared"]
    assert isinstance(declared_block, dict)
    declared_block["num_experts"] = 0  # fails DeclaredCheckpoint validation
    path.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(ManifestError, match="declared"):
        load(path)
