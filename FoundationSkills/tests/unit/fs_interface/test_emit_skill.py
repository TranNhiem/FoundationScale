
"""Tests for TrainingEmitSkill: PASS paths and every MUST_FIRE fixture."""
from __future__ import annotations

from foundationskills.core import SkillContext, Status
from foundationskills.interfaces.fs.capabilities import FSCapabilities
from foundationskills.skills.training.emit_skill import TrainingEmitSkill

_ALL_FLAGS = frozenset(
    {
        "--model", "--dataset", "--output-dir", "--max-steps",
        "--per-device-batch-size", "--learning-rate", "--save-interval",
        "--max-sequence-length", "--objective", "--seed", "--dp",
        "--tp", "--pp", "--ep", "--cp", "--nodes", "--gpus-per-node",
        "--profile-name", "--profile-path", "--dry-run", "--precision",
        "--adapter", "--adapter-rank", "--adapter-alpha", "--adapter-target",
        "--adapter-dropout", "--optimizer", "--gradient-accumulation-steps",
        "--lr-scheduler-type", "--warmup-steps", "--sharding-strategy",
        "--gradient-checkpointing",
    }
)


def make_caps(**over):
    base = dict(
        available=True,
        fs_version="0.1.0-test",
        train_flags=_ALL_FLAGS,
        train_objectives=("sft",),
        sharding_strategies=("ddp", "fsdp"),
        executed_axes=("tp", "cp"),
        refused_axes=("pp", "ep"),
        axes_measured=True,
        rl_algorithms=("dr_grpo", "gspo", "dapo", "dpo"),
        rl_saves_checkpoint=True,  # hypothetical: FS 77bfa65 measures False (see test_fix_*)
        rl_reward_kinds=("mcq_letter",),
        rl_runnable={"dr_grpo": None, "gspo": None, "dapo": None, "dpo": "not wired"},
        families={"gemma4": ("gemma4",)},
        backends=("ddp", "fsdp"),
    )
    base.update(over)
    return FSCapabilities(**base)


def good_stage(**hparams):
    return {"name": "sft1", "stage": "sft", "algorithm": None, "method": "full",
            "hparams": hparams, "family": "gemma4", "data": {"format": "sft"}}


def dataset(fmt="sft", tmp_path=None):
    root = tmp_path if tmp_path is not None else "."
    return {
        "format": fmt,
        "shards": [{"path": f"{root}/ds/shard-00000.jsonl", "sha256": "0" * 64, "records": 3}],
        "fs_columns": {"text_column": "text", "image_column": None, "gold_key": "answer"},
    }


def ctx_for(tmp_path, caps):
    return SkillContext(workdir=tmp_path, capabilities=caps)


def base_request(tmp_path, stages, with_dataset=True):
    req = {"plan": {"stages": stages}, "model": "models/g4", "output_root": str(tmp_path / "runs"),
           "hardware_id": "local", "nodes": 1, "gpus_per_node": 1}
    if with_dataset:
        req["dataset"] = dataset(tmp_path=tmp_path)
    return req


def test_fixtures_cover_every_rule():
    skill = TrainingEmitSkill()
    assert set(skill.must_fire_fixtures()) == {r.rule_id for r in skill.rules}


def test_pass_writes_spec_artifact(tmp_path):
    skill = TrainingEmitSkill()
    result = skill.execute(base_request(tmp_path, [good_stage()]), ctx_for(tmp_path, make_caps()))
    assert result.status is Status.PASS
    assert len(result.artifacts) == 1
    ref = result.artifacts[0]
    assert ref.type == "fs_launch_spec"
    spec = result.payload["specs"][0]
    assert spec["executable"] is True
    assert spec["argv"][0] == "python"  # world == 1
    assert spec["sbatch"] is None  # local hardware -> no sbatch
    assert spec["output_dir"] == f"{tmp_path}/runs/fskills-sft1"


def test_cluster_target_writes_sbatch_file(tmp_path):
    skill = TrainingEmitSkill()
    req = base_request(tmp_path, [good_stage()])
    req["hardware_id"] = "gb200-tray"
    req["nodes"] = 1
    req["gpus_per_node"] = 4
    result = skill.execute(req, ctx_for(tmp_path, make_caps()))
    assert result.status is Status.PASS
    spec = result.payload["specs"][0]
    assert spec["sbatch"] is not None
    assert "(exec 3<>/dev/tcp/master/8081)" in spec["sbatch"]  # IMEX preamble
    sbatch_file = tmp_path / "launch" / "fskills-sft1.sbatch"
    assert sbatch_file.exists()


def test_must_fire_fs_in_001_empty_plan(tmp_path):
    skill = TrainingEmitSkill()
    result = skill.execute(base_request(tmp_path, []), ctx_for(tmp_path, make_caps()))
    assert result.status is Status.REFUSED
    assert "FS-IN-001" in result.refusal


def test_must_fire_fs_in_002_missing_dataset(tmp_path):
    skill = TrainingEmitSkill()
    result = skill.execute(
        base_request(tmp_path, [good_stage()], with_dataset=False),
        ctx_for(tmp_path, make_caps()),
    )
    assert result.status is Status.REFUSED
    assert "FS-IN-002" in result.refusal
    assert "sft1" in result.refusal


def test_must_fire_fs_in_003_fs_not_importable(tmp_path):
    skill = TrainingEmitSkill()
    caps = make_caps(available=False, errors=("import foundationscale failed: no module",))
    result = skill.execute(base_request(tmp_path, [good_stage()]), ctx_for(tmp_path, caps))
    assert result.status is Status.REFUSED
    assert "FS-IN-003" in result.refusal
    assert "missing: importable foundationscale" in result.refusal


def test_must_fire_fs_ho_001_warn_on_one_non_executable(tmp_path):
    skill = TrainingEmitSkill()
    bad = good_stage(pp=2)
    bad["name"] = "bad"
    result = skill.execute(base_request(tmp_path, [good_stage(), bad]), ctx_for(tmp_path, make_caps()))
    assert result.status is Status.PASS
    warn = [f for f in result.findings if f.rule_id == "FS-HO-001"]
    assert len(warn) == 1
    assert warn[0].severity.value == "WARN"
    assert "pp>1" in warn[0].message
    assert all(f.rule_id != "FS-HO-002" for f in result.findings)


def test_must_fire_fs_ho_002_all_non_executable_is_red(tmp_path):
    skill = TrainingEmitSkill()
    result = skill.execute(base_request(tmp_path, [good_stage(pp=2)]), ctx_for(tmp_path, make_caps()))
    assert result.status is Status.RED
    assert any(f.rule_id == "FS-HO-002" for f in result.findings)


def test_must_fire_fs_ho_003_rl_single_device_info(tmp_path):
    skill = TrainingEmitSkill()
    rl = {"name": "rl1", "stage": "rl", "algorithm": "dr_grpo", "method": "full",
          "hparams": {}, "data": {"format": "rl"}}
    req = base_request(tmp_path, [rl])
    req["dataset"] = dataset(fmt="rl", tmp_path=tmp_path)
    result = skill.execute(req, ctx_for(tmp_path, make_caps()))
    assert result.status is Status.PASS
    spec = result.payload["specs"][0]
    assert spec["entry"] == "fskills-rl"
    assert spec["rl_config"]["gold_key"] == "answer"
    assert spec["executable"] is True
    info = [f for f in result.findings if f.rule_id == "FS-HO-003"]
    assert len(info) == 1
    assert info[0].severity.value == "INFO"
