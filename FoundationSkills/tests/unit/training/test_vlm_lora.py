"""Unit tests for the VLM competition LoRA shared contract (A4.1 Phase 1).

Exercises *vlm_gates* rules TR-IN-005..008 and TR-PL-006..008 (positive plus
near-miss negatives for every rule), the planner goal→stage field-copying
layout, the emit argv/env contract shapes, and boundary cases for 128K,
gemma-4-31B-it 32K, and --fused-loss liger.
"""

from __future__ import annotations

import json

from pathlib import Path

import pytest

from types import SimpleNamespace

from foundationskills.skills.training.planner import vlm_gates


# ---------------------------------------------------------------------------
# Minimal dict / namespace builders (shared-contract shapes as filled by the
# planner; no FoundationSkills API dependencies).
# ---------------------------------------------------------------------------


def _caps(
    *,
    registered_families: frozenset[str] | None = None,
    train_flags: frozenset[str] | None = None,
    module_inventory: frozenset[str] | None = None,
) -> SimpleNamespace:
    """Return a minimal caps namespace for *vlm_gates*."""
    return SimpleNamespace(
        registered_families=(
            frozenset() if registered_families is None else registered_families
        ),
        train_flags=frozenset() if train_flags is None else train_flags,
        module_inventory=module_inventory,
    )



def _gates(stage: dict, goal: dict, caps: object) -> list[tuple[str, str]]:
    """vlm_gates with the real identity channel: the model comes from goal.base_model (the knowledge
    base resolves its FS family); a caps-level module inventory is carried on the stage, where the gate reads it."""
    goal = {**goal}
    goal.setdefault("base_model", {"name_or_path": stage.get("family")})
    inventory = getattr(caps, "module_inventory", None)
    if inventory is not None:
        stage = {**stage, "module_inventory": sorted(inventory)}
    return vlm_gates(stage, goal, caps)

def _stage(
    *,
    modalities: tuple[str, ...] = ("image", "text", "video"),
    fused_loss: str | None = None,
    lora_targets: list[str] | None = None,
    max_sequence_length: int = 16384,
    video_column: str | None = "video",
    video_frames: int | None = 8,
    overlong: str | None = "drop",
    family: str = "gemma-4-12B-it",
    tp: int = 1,
    cp: int = 1,
) -> dict:
    """Return a training stage in the shared-contract layout."""
    return {
        "family": family,
        "modalities": list(modalities),
        "hparams": {
            "fused_loss": fused_loss,
            "lora_targets": lora_targets,
            "max_sequence_length": max_sequence_length,
            "tp": tp,
            "cp": cp,
        },
        "data": {
            "fs_columns": {
                "video_column": video_column,
                "video_frames": video_frames,
                "overlong": overlong,
            },
        },
    }


def _goal(
    *,
    modalities: list[str] | None = None,
    video_frames: int = 8,
    max_sequence_length: int = 16384,
    lora_targets: list[str] | None = None,
    fused_loss: str | None = None,
    tp: int = 1,
    cp: int = 1,
) -> dict:
    """Return a goal dict in the shared-contract declaration layout."""
    return {
        "modalities": (
            modalities if modalities is not None else ["image", "text", "video"]
        ),
        "video_frames": video_frames,
        "max_sequence_length": max_sequence_length,
        "lora_targets": lora_targets,
        "fused_loss": fused_loss,
        "tp": tp,
        "cp": cp,
    }


def _ids(findings: list[tuple[str, str]]) -> set[str]:
    return {fid for fid, _ in findings}


# ---------------------------------------------------------------------------
# TR-IN-005 — VLM family not FS-registered and no explicit lora_targets.
# ---------------------------------------------------------------------------


def test_tr_in_005_unregistered_family_without_targets() -> None:
    """TR-IN-005 fires: unregistered VLM family and no explicit lora_targets."""
    caps = _caps(registered_families=frozenset())
    stage = _stage(family="mse-vlm-family", lora_targets=None)
    goal = _goal(lora_targets=None)
    findings = _gates(stage, goal, caps)
    assert "TR-IN-005" in _ids(findings)
    msg = next(m for fid, m in findings if fid == "TR-IN-005")
    assert "register the family in FS" in msg
    assert "--adapter-target" in msg or "lora_targets" in msg


def test_tr_in_005_near_miss_registered_family_ok() -> None:
    """TR-IN-005 does not fire when the VLM family is FS-registered."""
    caps = _caps(registered_families=frozenset({"gemma-4-12B-it"}))
    stage = _stage(family="gemma-4-12B-it", lora_targets=None)
    goal = _goal(lora_targets=None)
    assert "TR-IN-005" not in _ids(_gates(stage, goal, caps))


def test_tr_in_005_near_miss_explicit_targets_ok() -> None:
    """TR-IN-005 does not fire when explicit lora_targets are provided.

    Even with an unregistered family the explicit targets resolve the
    unknown-family problem (TR-IN-005 is skipped).
    """
    caps = _caps(registered_families=frozenset())
    stage = _stage(family="mse-vlm-family", lora_targets=["q_proj", "k_proj"])
    goal = _goal(lora_targets=["q_proj", "k_proj"])
    assert "TR-IN-005" not in _ids(_gates(stage, goal, caps))


# ---------------------------------------------------------------------------
# TR-IN-006 — video rows without video_column or video_frames.
# ---------------------------------------------------------------------------


def test_tr_in_006_absent_video_column_gets_the_documented_default() -> None:
    """The contract defaults video_column to "video" when video rows are declared, so an absent column
    is filled, not refused; the env name it feeds is FOUNDATIONSCALE_TRAIN_VIDEO_COLUMN."""
    caps = _caps(registered_families=frozenset({"gemma-4-12B-it"}))
    stage = _stage(modalities=("text", "video"), video_column=None, video_frames=8)
    assert "TR-IN-006" not in _ids(_gates(stage, _goal(), caps))


def test_tr_in_006_video_without_video_frames() -> None:
    """TR-IN-006 fires: video rows declared and no video_frames anywhere (no default exists)."""
    caps = _caps(registered_families=frozenset({"gemma-4-12B-it"}))
    stage = _stage(modalities=("text", "video"), video_column="video", video_frames=None)
    findings = _gates(stage, _goal(video_frames=None), caps)
    assert "TR-IN-006" in _ids(findings)
    msg = next(m for fid, m in findings if fid == "TR-IN-006")
    assert "FOUNDATIONSCALE_TRAIN_VIDEO_FRAMES" in msg


def test_tr_in_006_near_miss_video_complete_ok() -> None:
    """TR-IN-006 does not fire when video_rows have both video_column and video_frames."""
    caps = _caps(registered_families=frozenset({"gemma-4-12B-it"}))
    stage = _stage(modalities=("text", "video"), video_column="video", video_frames=8)
    goal = _goal()
    assert "TR-IN-006" not in _ids(_gates(stage, goal, caps))


def test_tr_in_006_near_miss_no_video_modality_ok() -> None:
    """TR-IN-006 does not fire when modalities is text-only (no video rows)."""
    caps = _caps(registered_families=frozenset({"gemma-4-12B-it"}))
    stage = _stage(modalities=("text",), video_column=None, video_frames=None)
    goal = _goal(modalities=["text"])
    assert "TR-IN-006" not in _ids(_gates(stage, goal, caps))


# ---------------------------------------------------------------------------
# TR-IN-007 — explicit lora_targets empty after normalisation or no module match.
# ---------------------------------------------------------------------------


def test_tr_in_007_empty_explicit_targets() -> None:
    """TR-IN-007 fires: explicit lora_targets is an empty list (after normalisation)."""
    caps = _caps(registered_families=frozenset({"gemma-4-12B-it"}))
    stage = _stage(lora_targets=[])
    goal = _goal(lora_targets=[])
    assert "TR-IN-007" in _ids(_gates(stage, goal, caps))


def test_tr_in_007_targets_not_in_module_inventory() -> None:
    """TR-IN-007 fires: explicit lora_target matches no module in the known inventory."""
    caps = _caps(
        registered_families=frozenset({"gemma-4-12B-it"}),
        module_inventory=frozenset(
            {
                "model.layers.0.self_attn.q_proj",
                "model.layers.0.self_attn.k_proj",
            }
        ),
    )
    stage = _stage(lora_targets=["model.layers.0.mlp.bogus_layer"])
    goal = _goal(lora_targets=["model.layers.0.mlp.bogus_layer"])
    assert "TR-IN-007" in _ids(_gates(stage, goal, caps))


def test_tr_in_007_near_miss_all_targets_match_ok() -> None:
    """TR-IN-007 does not fire when every explicit target matches some module."""
    caps = _caps(
        registered_families=frozenset({"gemma-4-12B-it"}),
        module_inventory=frozenset(
            {
                "model.layers.0.self_attn.q_proj",
                "model.layers.0.self_attn.k_proj",
                "model.layers.1.self_attn.q_proj",
            }
        ),
    )
    stage = _stage(lora_targets=["model.layers.0.self_attn.q_proj", "k_proj"])
    goal = _goal(lora_targets=["model.layers.0.self_attn.q_proj", "k_proj"])
    assert "TR-IN-007" not in _ids(_gates(stage, goal, caps))


def test_tr_in_007_near_miss_no_inventory_ok() -> None:
    """TR-IN-007 does not fire when no module inventory is carried (FS refuses post-wrap)."""
    caps = _caps(registered_families=frozenset({"gemma-4-12B-it"}), module_inventory=None)
    stage = _stage(lora_targets=["model.layers.0.mlp.bogus_layer"])
    goal = _goal(lora_targets=["model.layers.0.mlp.bogus_layer"])
    assert "TR-IN-007" not in _ids(_gates(stage, goal, caps))


# ---------------------------------------------------------------------------
# TR-IN-008 — tp>1, or cp>1 with image/video rows.
# ---------------------------------------------------------------------------


def test_tr_in_008_tp_gt_one_refused() -> None:
    """TR-IN-008 fires: tp > 1 is refused (FS cannot span tensor-parallel drivers)."""
    caps = _caps(registered_families=frozenset({"gemma-4-12B-it"}))
    stage = _stage(tp=2, cp=1)
    goal = _goal(tp=2, cp=1)
    assert "TR-IN-008" in _ids(_gates(stage, goal, caps))


def test_tr_in_008_cp_gt_one_with_vision_refused() -> None:
    """TR-IN-008 fires: cp > 1 with image/video rows (unmeasured end to end)."""
    caps = _caps(registered_families=frozenset({"gemma-4-12B-it"}))
    stage = _stage(modalities=("image", "text", "video"), tp=1, cp=2)
    goal = _goal(tp=1, cp=2)
    assert "TR-IN-008" in _ids(_gates(stage, goal, caps))


def test_tr_in_008_near_miss_cp_text_only_ok() -> None:
    """TR-IN-008 does not fire: cp > 1 with text-only rows (measured end to end)."""
    caps = _caps(registered_families=frozenset({"gemma-4-12B-it"}))
    stage = _stage(modalities=("text",), tp=1, cp=2)
    goal = _goal(modalities=["text"], tp=1, cp=2)
    assert "TR-IN-008" not in _ids(_gates(stage, goal, caps))


def test_tr_in_008_near_miss_all_ones_ok() -> None:
    """TR-IN-008 does not fire when tp=1 and cp=1."""
    caps = _caps(registered_families=frozenset({"gemma-4-12B-it"}))
    stage = _stage(tp=1, cp=1)
    goal = _goal(tp=1, cp=1)
    assert "TR-IN-008" not in _ids(_gates(stage, goal, caps))


# ---------------------------------------------------------------------------
# TR-PL-006 — max_sequence_length >= 131072 (128K) RED.
# ---------------------------------------------------------------------------


def test_tr_pl_006_128k_boundary_red() -> None:
    """TR-PL-006 fires at the 128K boundary (131072)."""
    caps = _caps(registered_families=frozenset({"gemma-4-12B-it"}))
    stage = _stage(max_sequence_length=131072, fused_loss="liger")
    goal = _goal(max_sequence_length=131072, fused_loss="liger")
    assert "TR-PL-006" in _ids(_gates(stage, goal, caps))


def test_tr_pl_006_near_miss_64k_ok() -> None:
    """TR-PL-006 does not fire below 128K (65536 = 64K stays green)."""
    caps = _caps(registered_families=frozenset({"gemma-4-12B-it"}))
    stage = _stage(max_sequence_length=65536, fused_loss="liger")
    goal = _goal(max_sequence_length=65536, fused_loss="liger")
    assert "TR-PL-006" not in _ids(_gates(stage, goal, caps))


# ---------------------------------------------------------------------------
# TR-PL-007 — gemma-4-31B-it planned above 32768 RED.
# ---------------------------------------------------------------------------


def test_tr_pl_007_31b_above_32k_red() -> None:
    """TR-PL-007 fires: gemma-4-31B-it planned at 65536 (above 32768)."""
    caps = _caps(registered_families=frozenset({"gemma-4-31B-it"}))
    stage = _stage(
        family="gemma-4-31B-it", max_sequence_length=65536, fused_loss="liger"
    )
    goal = _goal(max_sequence_length=65536, fused_loss="liger")
    assert "TR-PL-007" in _ids(_gates(stage, goal, caps))


def test_tr_pl_007_near_miss_31b_at_32k_ok() -> None:
    """TR-PL-007 does not fire: gemma-4-31B-it at exactly 32768 (the measured cap)."""
    caps = _caps(registered_families=frozenset({"gemma-4-31B-it"}))
    stage = _stage(
        family="gemma-4-31B-it", max_sequence_length=32768, fused_loss="liger"
    )
    goal = _goal(max_sequence_length=32768, fused_loss="liger")
    assert "TR-PL-007" not in _ids(_gates(stage, goal, caps))


def test_tr_pl_007_near_miss_other_family_ok() -> None:
    """TR-PL-007 does not fire: gemma-4-12B-it is not subject to the 31B limit."""
    caps = _caps(registered_families=frozenset({"gemma-4-12B-it"}))
    stage = _stage(
        family="gemma-4-12B-it", max_sequence_length=65536, fused_loss="liger"
    )
    goal = _goal(max_sequence_length=65536, fused_loss="liger")
    assert "TR-PL-007" not in _ids(_gates(stage, goal, caps))


# ---------------------------------------------------------------------------
# TR-PL-008 — max_sequence_length >= 32768 without fused_loss == "liger".
# ---------------------------------------------------------------------------


def test_tr_pl_008_32k_without_liger_refused() -> None:
    """TR-PL-008 fires at the 32K boundary without fused_loss='liger'."""
    caps = _caps(registered_families=frozenset({"gemma-4-12B-it"}))
    stage = _stage(max_sequence_length=32768, fused_loss=None)
    goal = _goal(max_sequence_length=32768, fused_loss=None)
    findings = _gates(stage, goal, caps)
    assert "TR-PL-008" in _ids(findings)
    msg = next(m for fid, m in findings if fid == "TR-PL-008")
    assert "--fused-loss" in msg


def test_tr_pl_008_above_32k_without_liger_refused() -> None:
    """TR-PL-008 fires above 32K without fused_loss='liger'."""
    caps = _caps(registered_families=frozenset({"gemma-4-12B-it"}))
    stage = _stage(max_sequence_length=65536, fused_loss=None)
    goal = _goal(max_sequence_length=65536, fused_loss=None)
    assert "TR-PL-008" in _ids(_gates(stage, goal, caps))


def test_tr_pl_008_near_miss_32k_with_liger_ok() -> None:
    """TR-PL-008 does not fire at 32K when fused_loss='liger' is set."""
    caps = _caps(registered_families=frozenset({"gemma-4-12B-it"}))
    stage = _stage(max_sequence_length=32768, fused_loss="liger")
    goal = _goal(max_sequence_length=32768, fused_loss="liger")
    assert "TR-PL-008" not in _ids(_gates(stage, goal, caps))


def test_tr_pl_008_near_miss_below_32k_ok() -> None:
    """TR-PL-008 does not fire below 32K (16384 has no fused-loss requirement)."""
    caps = _caps(registered_families=frozenset({"gemma-4-12B-it"}))
    stage = _stage(max_sequence_length=16384, fused_loss=None)
    goal = _goal(max_sequence_length=16384, fused_loss=None)
    assert "TR-PL-008" not in _ids(_gates(stage, goal, caps))


def test_tr_pl_008_non_liger_value_refused() -> None:
    """TR-PL-008 fires when fused_loss is set to a non-'liger' value above 32K."""
    caps = _caps(registered_families=frozenset({"gemma-4-12B-it"}))
    stage = _stage(max_sequence_length=32768, fused_loss="megatron")
    goal = _goal(max_sequence_length=32768, fused_loss="megatron")
    assert "TR-PL-008" in _ids(_gates(stage, goal, caps))


# ---------------------------------------------------------------------------
# 128K / 31B / fused-loss boundary scenarios (cross-rule).
# ---------------------------------------------------------------------------


def test_128k_with_liger_still_red() -> None:
    """128K carries the RED regardless of the fused-loss setting (unmeasured capability)."""
    caps = _caps(registered_families=frozenset({"gemma-4-12B-it"}))
    stage = _stage(max_sequence_length=131072, fused_loss="liger")
    goal = _goal(max_sequence_length=131072, fused_loss="liger")
    ids = _ids(_gates(stage, goal, caps))
    assert "TR-PL-006" in ids and "TR-PL-008" not in ids


def test_31b_32k_with_liger_clean() -> None:
    """gem-4-31B-it at 32K with --fused-loss liger passes the whole sequence rule family."""
    caps = _caps(registered_families=frozenset({"gemma-4-31B-it"}))
    stage = _stage(
        family="gemma-4-31B-it", max_sequence_length=32768, fused_loss="liger"
    )
    goal = _goal(max_sequence_length=32768, fused_loss="liger")
    ids = _ids(_gates(stage, goal, caps))
    assert "TR-PL-006" not in ids
    assert "TR-PL-007" not in ids
    assert "TR-PL-008" not in ids


def test_31b_64k_with_liger_red_only_007() -> None:
    """gemma-4-31B-it at 64K with liger: TR-PL-007 RED only (TR-PL-008 satisfied)."""
    caps = _caps(registered_families=frozenset({"gemma-4-31B-it"}))
    stage = _stage(
        family="gemma-4-31B-it", max_sequence_length=65536, fused_loss="liger"
    )
    goal = _goal(max_sequence_length=65536, fused_loss="liger")
    ids = _ids(_gates(stage, goal, caps))
    assert "TR-PL-007" in ids
    assert "TR-PL-008" not in ids
    assert "TR-PL-006" not in ids


# ---------------------------------------------------------------------------
# planner copying: goal fields → stage sub-dictionary layout (contract shape).
# ---------------------------------------------------------------------------


def test_planner_copies_modalities_sorted() -> None:
    """The planner copies goal['modalities'] into stage['modalities'] as a sorted list."""
    goal = _goal(modalities=["video", "image", "text"])
    stage = _stage(modalities=tuple(sorted(goal["modalities"])))
    assert stage["modalities"] == ["image", "text", "video"]


def test_planner_copies_hparams_and_fs_columns() -> None:
    """Goal fields land in the described stage sub-dict keys (contract mapping)."""
    goal = _goal(
        video_frames=8,
        max_sequence_length=16384,
        lora_targets=["model.layers.0.self_attn.q_proj"],
        fused_loss="liger",
    )
    stage = _stage(
        modalities=tuple(sorted(goal["modalities"])),
        fused_loss=goal["fused_loss"],
        lora_targets=goal["lora_targets"],
        max_sequence_length=goal["max_sequence_length"],
        video_column="video",
        video_frames=goal["video_frames"],
        overlong="drop",
    )

    assert stage["hparams"]["fused_loss"] == "liger"
    assert stage["hparams"]["lora_targets"] == ["model.layers.0.self_attn.q_proj"]
    assert stage["hparams"]["max_sequence_length"] == 16384
    assert stage["data"]["fs_columns"]["video_column"] == "video"
    assert stage["data"]["fs_columns"]["video_frames"] == 8
    assert stage["data"]["fs_columns"]["overlong"] == "drop"

    caps = _caps(registered_families=frozenset({"gemma-4-12B-it"}))
    assert _ids(_gates(stage, goal, caps)) == set()


def test_planner_default_video_column_and_overlong() -> None:
    """Defaults apply for video_column ('video') and overlong ('drop')."""
    goal = _goal(video_frames=8)
    stage = _stage(
        modalities=("text", "video"),
        video_column="video",
        video_frames=goal["video_frames"],
        overlong="drop",
    )
    assert stage["data"]["fs_columns"]["video_column"] == "video"
    assert stage["data"]["fs_columns"]["overlong"] == "drop"


# ---------------------------------------------------------------------------
# emit: --fused-loss argv pair and video / overlong env vars (contract output).
# ---------------------------------------------------------------------------


from foundationskills.interfaces.fs.capabilities import FSCapabilities  # noqa: E402
from foundationskills.interfaces.fs.emit_train import emit_train  # noqa: E402

_EMIT_FLAGS = frozenset({
    "--model", "--dataset", "--output-dir", "--max-steps", "--per-device-batch-size", "--learning-rate",
    "--save-interval", "--max-sequence-length", "--objective", "--seed", "--dp", "--nodes", "--gpus-per-node",
    "--profile-name", "--precision", "--adapter", "--adapter-rank", "--adapter-alpha", "--adapter-target",
    "--adapter-dropout", "--gradient-accumulation-steps", "--sharding-strategy", "--gradient-checkpointing",
    "--logging-steps", "--fused-loss", "--dry-run",
})


def _emit(modalities: tuple[str, ...], *, fused_loss: str | None = None, tmp_path: Path,
          kernels: dict | None = None) -> dict:
    """The real emit_train over a VLM LoRA stage (no hand-rebuilt env: the emitter decides)."""
    caps = FSCapabilities(available=True, fs_version="test", train_flags=_EMIT_FLAGS, train_objectives=("sft",),
                          sharding_strategies=("ddp", "fsdp"), executed_axes=("cp",), refused_axes=("pp", "ep"),
                          axes_measured=True, families={"gemma4": ("gemma4",)}, backends=("ddp", "fsdp"),
                          fused_loss_kernels={"liger": True} if kernels is None else kernels)
    hparams = {"max_sequence_length": 16384, "lora_rank": 16}
    if fused_loss:
        hparams["fused_loss"] = fused_loss
    stage = {"name": "vlm", "stage": "sft", "algorithm": None, "method": "lora", "family": "gemma4",
             "modalities": list(modalities), "hparams": hparams,
             "data": {"format": "mm_sft", "fs_columns": {"video_column": "video", "video_frames": 16, "overlong": "drop"}}}
    dataset = {"format": "mm_sft", "shards": [{"path": f"{tmp_path}/ds/shard-00000.jsonl", "sha256": "0" * 64, "records": 3}],
               "fs_columns": {"text_column": None, "image_column": "image", "gold_key": None}}
    return emit_train(stage, dataset=dataset, model="models/gemma-4-12B-it", output_dir=str(tmp_path / "out"),
                      hardware={"id": "local", "scheduler": None}, nodes=1, gpus_per_node=1, caps=caps, run_name="vlm")


def test_emit_passes_fused_loss_only_when_declared(tmp_path: Path) -> None:
    argv = _emit(("image", "text"), fused_loss="liger", tmp_path=tmp_path)["argv"]
    assert argv[argv.index("--fused-loss") + 1] == "liger"
    assert "--fused-loss" not in _emit(("image", "text"), tmp_path=tmp_path)["argv"]


def test_emit_env_for_image_video_text_matches_the_measured_launcher(tmp_path: Path) -> None:
    """examples/vlm-2xH200/scripts/train_sft.sh:13-17 declares these five variables."""
    env = _emit(("image", "text", "video"), tmp_path=tmp_path)["env"]
    assert env["FOUNDATIONSCALE_TRAIN_CONVERSATIONS_COLUMN"] == "conversations"
    assert env["FOUNDATIONSCALE_TRAIN_IMAGE_COLUMN"] == "image"
    assert env["FOUNDATIONSCALE_TRAIN_VIDEO_COLUMN"] == "video"
    assert env["FOUNDATIONSCALE_TRAIN_VIDEO_FRAMES"] == "16"
    assert env["FOUNDATIONSCALE_TRAIN_OVERLONG"] == "drop"


def test_emit_env_omits_video_without_video_and_multimodal_vars_for_text(tmp_path: Path) -> None:
    env = _emit(("image", "text"), tmp_path=tmp_path)["env"]
    assert "FOUNDATIONSCALE_TRAIN_VIDEO_COLUMN" not in env and "FOUNDATIONSCALE_TRAIN_VIDEO_FRAMES" not in env
    text_env = _emit(("text",), tmp_path=tmp_path)["env"]
    assert "FOUNDATIONSCALE_TRAIN_OVERLONG" not in text_env
    assert "FOUNDATIONSCALE_TRAIN_CONVERSATIONS_COLUMN" not in text_env


def test_fs_in_004_accepts_conversation_rows_for_the_text_modality(tmp_path: Path) -> None:
    """Real-data finding (A4.1 GPU run): a conversation dataset feeds 'text' via conversations_column, so
    FS-IN-004 must not demand a flat text_column; a dataset with neither still refuses."""
    from foundationskills.core.contract import SkillContext
    from foundationskills.skills.training.emit_skill import TrainingEmitSkill

    skill = TrainingEmitSkill()
    fixture = skill.must_fire_fixtures()["FS-IN-004"]
    request = json.loads(json.dumps(fixture["request"]))
    request["plan"]["stages"][0]["modalities"] = ["text"]
    request["dataset"]["fs_columns"] = {"text_column": None, "conversations_column": "conversations",
                                        "image_column": None, "gold_key": None}
    findings = skill.check_inputs(request, SkillContext(tmp_path))
    assert "FS-IN-004" not in [f.rule_id for f in findings]
    request["dataset"]["fs_columns"]["conversations_column"] = None
    assert "FS-IN-004" in [f.rule_id for f in skill.check_inputs(request, SkillContext(tmp_path))]


def test_vlm_long_context_stage_plans_one_row_per_device(tmp_path: Path) -> None:
    """Real-GPU finding (A4.1): a text recipe's micro-batch 4 x 16K VLM rows priced 330 GB/GPU; the measured
    VLM recipe is batch 1 with grad-accum (examples/vlm-2xH200/scripts/train_sft.sh:24)."""
    from foundationskills.skills.training.planner import plan

    goal = {"objective": "vlm lora", "target_capabilities": ["general_chat"], "domain": "vlm",
            "base_model": {"name_or_path": "gemma-4-12B-it", "model_type": "vlm"}, "method": "lora",
            "modalities": ["image", "video", "text"], "video_frames": 16, "max_sequence_length": 16384,
            "data": {"sources": [{"uri": str(tmp_path / "mix.jsonl"), "kind": "jsonl", "approx_examples": 300,
                                  "use_for": ["sft"]}]},
            "hardware": {"gpu": "GB200", "gpus_per_node": 1, "nodes": 1}}
    caps = FSCapabilities(available=True, fs_version="test", train_flags=_EMIT_FLAGS, train_objectives=("sft",),
                          sharding_strategies=("ddp", "fsdp"), executed_axes=("cp",), refused_axes=("pp", "ep"),
                          axes_measured=True, families={"gemma4": ("gemma4",)}, backends=("ddp", "fsdp"))
    stages = [s for s in plan(goal, caps=caps, hardware_id="gb200")["stages"] if s["stage"] == "sft"]
    assert stages and stages[0]["hparams"]["per_device_batch_size"] == 1


@pytest.mark.parametrize("kernels, executable, needle", [
    ({"liger": True}, True, None),
    ({"liger": False}, False, "foundationscale[train]"),
    ({}, False, "unmeasured"),
])
def test_fused_loss_needs_a_measured_importable_kernel(tmp_path: Path, kernels: dict, executable: bool,
                                                       needle: str | None) -> None:
    """Real-GPU finding (A4.1): FS lists 'liger' as a --fused-loss choice without liger_kernel installed and
    refuses at start (exit 96); the spec is executable only when the kernel is measured importable."""
    out = _emit(("image", "text"), fused_loss="liger", tmp_path=tmp_path, kernels=kernels)
    assert out["executable"] is executable, out.get("missing")
    if needle:
        assert needle in (out.get("missing") or "")
