"""``resolve_shard_ref`` and its wiring into the SFT/RL emitters.

The bug under test: both emitters flattened a dataset's shard list into ONE
directory for a consumer that then globbed every ``*.json*``/``*.jsonl`` in that
directory -- so an eval/test split beside a train shard became training data,
and shards in other directories silently dropped out of the run.
"""
from __future__ import annotations

from pathlib import Path

from foundationskills.interfaces.fs.capabilities import FSCapabilities
from foundationskills.interfaces.fs.emit_rl import emit_rl
from foundationskills.interfaces.fs.emit_train import emit_train
from foundationskills.interfaces.fs.shard_paths import resolve_shard_ref


def _file(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("{}\n", encoding="utf-8")
    return path


def test_declared_shards_exactly_fill_the_directory(tmp_path):
    ds = tmp_path / "ds"
    s0 = _file(ds / "shard-00000.jsonl")
    s1 = _file(ds / "shard-00001.jsonl")
    assert resolve_shard_ref([{"path": str(s0)}, {"path": str(s1)}], "*.json*") == (str(ds), None, [])


def test_eval_split_beside_the_train_shard_passes_the_file_and_names_the_stray(tmp_path):
    ds = tmp_path / "ds"
    shard = _file(ds / "shard-00000.jsonl")
    _file(ds / "test.jsonl")
    path, refusal, notes = resolve_shard_ref([{"path": str(shard)}], "*.json*")
    assert refusal is None
    assert path == str(shard)
    assert notes == [
        f"passing the shard file, not {ds}: the directory also holds 1 other file(s) "
        f"the trainer would load: test.jsonl"
    ]


def test_two_shards_and_a_stray_file_is_refused_naming_the_stray(tmp_path):
    ds = tmp_path / "ds"
    s0 = _file(ds / "shard-00000.jsonl")
    s1 = _file(ds / "shard-00001.jsonl")
    _file(ds / "test.jsonl")
    path, refusal, notes = resolve_shard_ref([{"path": str(s0)}, {"path": str(s1)}], "*.json*")
    assert (path, notes) == (None, [])
    assert refusal == (
        f"shard directory {ds} also holds 1 non-shard file(s) the trainer would "
        f"load: test.jsonl; move the shards into their own directory"
    )


def test_a_declared_shard_absent_from_the_directory_is_refused(tmp_path):
    ds = tmp_path / "ds"
    _file(ds / "shard-00000.jsonl")
    present = str(ds / "shard-00000.jsonl")
    gone = str(ds / "shard-00001.jsonl")
    path, refusal, notes = resolve_shard_ref([{"path": present}, {"path": gone}], "*.json*")
    assert (path, notes) == (None, [])
    assert refusal == f"dataset shard(s) missing from {ds}: shard-00001.jsonl"


def test_shards_in_two_directories_are_refused_naming_both(tmp_path):
    a_dir = tmp_path / "a"
    b_dir = tmp_path / "b"
    a = _file(a_dir / "shard-00000.jsonl")
    b = _file(b_dir / "shard-00001.jsonl")
    path, refusal, notes = resolve_shard_ref([{"path": str(b)}, {"path": str(a)}], "*.json*")
    assert (path, notes) == (None, [])
    assert refusal == (
        f"dataset shards span 2 directories ({a_dir}, {b_dir}); the trainer takes one file or one "
        f"directory - co-locate the shards in one directory"
    )


def test_a_shard_without_a_path_is_refused_by_index(tmp_path):
    shard = _file(tmp_path / "ds" / "shard-00000.jsonl")
    path, refusal, notes = resolve_shard_ref([{"path": str(shard)}, {"records": 2}], "*.json*")
    assert (path, notes) == (None, [])
    assert refusal == "dataset shard 1 has no path"


def test_no_shards_at_all_is_refused_by_name_and_reason():
    empty = (None, "dataset payload has no shards to pass to --dataset", [])
    assert resolve_shard_ref([], "*.json*") == empty
    assert resolve_shard_ref([{"records": 3}], "*.jsonl") == empty


def test_absent_single_shard_directory_is_passed_as_before(tmp_path):
    ds = tmp_path / "absent"
    # Rule 3b, one shard: an unverifiable single-shard directory is passed as before
    # (the reporting difference is the note only the multi-shard branch emits).
    assert resolve_shard_ref([{"path": str(ds / "shard-00000.jsonl")}], "*.json*") == (str(ds), None, [])


def test_absent_multi_shard_directory_is_passed_but_reported_unverified(tmp_path):
    ds = tmp_path / "absent"
    shards = [{"path": str(ds / "shard-00000.jsonl")}, {"path": str(ds / "shard-00001.jsonl")}]
    path, refusal, notes = resolve_shard_ref(shards, "*.json*")
    assert (path, refusal) == (str(ds), None)
    assert notes == [f"could not verify {ds} holds only the declared shards: directory absent on this host"]


def test_rl_pattern_ignores_a_json_file_the_fs_pattern_would_load(tmp_path):
    ds = tmp_path / "ds"
    shard = _file(ds / "shard-00000.jsonl")
    _file(ds / "notes.json")
    assert resolve_shard_ref([{"path": str(shard)}], "*.jsonl") == (str(ds), None, [])
    path, refusal, notes = resolve_shard_ref([{"path": str(shard)}], "*.json*")
    assert refusal is None and path == str(shard)
    assert "notes.json" in notes[0]


# --------------------------------------------------------------------------
# End to end through the emitters (built like tests/unit/fs_interface
# test_emit_train.py / test_emit_rl.py, with REAL files on disk).
# --------------------------------------------------------------------------

_TRAIN_FLAGS = frozenset(
    {
        "--model", "--dataset", "--output-dir", "--max-steps",
        "--per-device-batch-size", "--learning-rate", "--save-interval",
        "--max-sequence-length", "--objective", "--seed", "--dp", "--tp",
        "--pp", "--ep", "--cp", "--nodes", "--gpus-per-node", "--profile-name",
        "--profile-path", "--dry-run", "--precision",
    }
)


def _caps():
    return FSCapabilities(
        available=True,
        fs_version="0.1.0-test",
        train_flags=_TRAIN_FLAGS,
        train_objectives=("sft",),
        sharding_strategies=("ddp", "fsdp"),
        executed_axes=("tp", "cp"),
        refused_axes=("pp", "ep"),
        axes_measured=True,
        rl_algorithms=("dr_grpo", "gspo", "dapo", "dpo"),
        rl_saves_checkpoint=True,
        rl_reward_kinds=("mcq_letter",),
        rl_runnable={"dr_grpo": None, "gspo": None, "dapo": None, "dpo": None},
        families={"gemma4": ("gemma4",)},
        backends=("ddp", "fsdp"),
    )


def test_emit_train_lists_the_refusal_and_stays_non_executable(tmp_path):
    ds = tmp_path / "ds"
    s0 = _file(ds / "shard-00000.jsonl")
    s1 = _file(ds / "shard-00001.jsonl")
    _file(ds / "test.jsonl")
    dataset = {
        "format": "sft",
        "shards": [
            {"path": str(s0), "sha256": "0" * 64, "records": 5},
            {"path": str(s1), "sha256": "1" * 64, "records": 5},
        ],
        "fs_columns": {"text_column": "text", "image_column": None, "gold_key": None},
    }
    stage = {
        "name": "sft1",
        "stage": "sft",
        "algorithm": None,
        "method": "full",
        "hparams": {"max_sequence_length": 2048},
        "family": "gemma4",
        "data": {"format": "sft"},
    }
    spec = emit_train(
        stage,
        dataset=dataset,
        model="models/g4",
        output_dir=str(tmp_path / "out"),
        hardware={"id": "local", "scheduler": None},
        nodes=1,
        gpus_per_node=1,
        caps=_caps(),
        run_name="fskills-sft1",
    )
    assert spec["executable"] is False
    assert "test.jsonl" in spec["missing"]
    assert "move the shards into their own directory" in spec["missing"]
    argv = spec["argv"]
    assert argv[argv.index("--dataset") + 1] == "<no shards>"


def _rl_stage():
    return {
        "name": "rl1",
        "stage": "rl",
        "algorithm": "dr_grpo",
        "method": "full",
        "hparams": {"group_size": 8, "max_steps": 12},
        "data": {"format": "rl"},
    }


def _rl_dataset(shard_paths):
    return {
        "format": "rl",
        "shards": [{"path": p, "sha256": "0" * 64, "records": 5} for p in shard_paths],
        "fs_columns": {"text_column": None, "image_column": None, "gold_key": "answer"},
    }


def test_emit_rl_lists_the_refusal_and_stays_non_executable(tmp_path):
    ds = tmp_path / "rl"
    s0 = _file(ds / "shard-00000.jsonl")
    s1 = _file(ds / "shard-00001.jsonl")
    _file(ds / "test.jsonl")
    spec = emit_rl(
        _rl_stage(),
        dataset=_rl_dataset([str(s0), str(s1)]),
        model="models/g4",
        output_dir=str(tmp_path / "out"),
        caps=_caps(),
        run_name="fskills-rl1",
    )
    assert spec["executable"] is False
    assert "non-shard" in spec["missing"] and "test.jsonl" in spec["missing"]
    assert spec["rl_config"]["dataset"] == "<no shards>"


def test_emit_rl_passes_the_single_shard_file_with_a_note_naming_the_stray(tmp_path):
    ds = tmp_path / "rl"
    s0 = _file(ds / "shard-00000.jsonl")
    _file(ds / "test.jsonl")
    spec = emit_rl(
        _rl_stage(),
        dataset=_rl_dataset([str(s0)]),
        model="models/g4",
        output_dir=str(tmp_path / "out"),
        caps=_caps(),
        run_name="fskills-rl1",
    )
    assert spec["executable"] is True
    assert spec["rl_config"]["dataset"] == str(s0)
    assert any("test.jsonl" in note for note in spec["notes"])


def test_emit_rl_treats_a_stray_json_file_as_corpus_data(tmp_path):
    # FS rl/corpus.py globs *.json as well as *.jsonl, so notes.json would be loaded too.
    ds = tmp_path / "rl"
    s0 = _file(ds / "shard-00000.jsonl")
    _file(ds / "notes.json")
    spec = emit_rl(
        _rl_stage(),
        dataset=_rl_dataset([str(s0)]),
        model="models/g4",
        output_dir=str(tmp_path / "out"),
        caps=_caps(),
        run_name="fskills-rl1",
    )
    assert spec["rl_config"]["dataset"] == str(s0)
    assert any("notes.json" in note for note in spec["notes"])
