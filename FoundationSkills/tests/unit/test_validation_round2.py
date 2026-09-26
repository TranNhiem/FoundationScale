"""Regressions for defects found by real runs (GB200, 2026-09-26) and the
adversarial review. Each test names the follow-up (F*) or review item (rv*)
from artifacts/foundationskills/FOLLOWUPS.md that it pins."""
from __future__ import annotations

import json
import types

import pytest

from foundationskills.interfaces.fs.capabilities import FSCapabilities
from foundationskills.skills.data_engine.ops.base import OpStats


def caps(**over):
    base = dict(
        available=True, fs_version="t", train_objectives=("sft",), sharding_strategies=("ddp", "fsdp"),
        executed_axes=("tp", "cp"), refused_axes=("pp", "ep"), axes_measured=True,
        rl_algorithms=("dr_grpo", "dpo"), rl_runnable={"dr_grpo": None, "dpo": "not wired"},
        backends=("ddp", "fsdp"), families={"gemma4": ("gemma4",)}, rl_reward_kinds=("mcq_letter",),
        rl_saves_checkpoint=False,
        train_flags=frozenset({"--model", "--dataset", "--output-dir", "--nodes", "--gpus-per-node", "--profile-name",
                               "--dp", "--objective", "--max-steps", "--learning-rate", "--max-sequence-length",
                               "--gradient-checkpointing", "--per-device-batch-size", "--dry-run",
                               "--sharding-strategy", "--adapter", "--adapter-rank", "--adapter-alpha",
                               "--adapter-dropout", "--warmup-steps", "--lr-scheduler-type"}),
        train_flag_choices={"--gradient-checkpointing": ("true", "false")},
    )
    base.update(over)
    return FSCapabilities(**base)


# ---------------------------------------------------------------- clean (F3, rv21-25)
@pytest.mark.parametrize("text", ["01000001 01000111 01000001 01000111", "0.0000000000000004648 g",
                                  "version 1.2.3.4.5", "pi 3.14159265358979", "1234 5678 9012"])
def test_f3_pii_false_positives_left_alone(text):
    from foundationskills.skills.data_engine.ops.clean import _redact_pii_builtin
    assert _redact_pii_builtin(text, {"credit_card", "ipv4", "phone"}) == (text, {})


@pytest.mark.parametrize("text,kind", [("4111 1111 1111 1111", "credit_card"), ("4111-1111-1111-1111", "credit_card"),
                                       ("(415)555-0100", "phone"), ("+886 2 2345 6789", "phone"),
                                       ("10.0.0.12", "ipv4"), ("A123456789", "national_id")])
def test_f3_pii_true_positives_redacted(text, kind):
    from foundationskills.skills.data_engine.ops.clean import _redact_pii_builtin
    assert _redact_pii_builtin(text, {kind})[1] == {kind: 1}


def test_rv22_national_id_needs_checksum_and_rv25_unknown_type_refuses():
    from foundationskills.skills.data_engine.ops.clean import _redact_pii_builtin
    assert _redact_pii_builtin("A123456788", {"national_id"})[1] == {}
    with pytest.raises(ValueError, match="unknown PII type"):
        _redact_pii_builtin("x", {"emial"})


def test_rv24_presidio_requested_but_missing_is_a_refusal(monkeypatch):
    import builtins
    from foundationskills.skills.data_engine.ops.clean import CleanError, _clean_op
    real_import = builtins.__import__

    def fake_import(name, *a, **k):
        if name.startswith("presidio"):
            raise ImportError(name)
        return real_import(name, *a, **k)
    monkeypatch.setattr(builtins, "__import__", fake_import)
    with pytest.raises(CleanError, match="presidio-analyzer"):
        list(_clean_op([{"text": "hi"}], {"pii": {"backend": "presidio"}}, OpStats("clean")))


def test_f3_html_keeps_think_tags_and_measures_pii_remaining():
    from foundationskills.skills.data_engine.ops.clean import _clean_op
    st = OpStats("clean")
    recs = [{"id": "1", "messages": [{"role": "user", "content": "a<5mm?"},
                                     {"role": "assistant", "content": "<think>r</think> ok",
                                      "reasoning_content": "mail a@b.com"}]},
            {"id": "2", "text": "<p>Hi</p><script>x()</script><think>t</think>"}]
    out = list(_clean_op(recs, {}, st))
    assert out[0]["messages"][1]["content"] == "<think>r</think> ok"
    assert out[0]["messages"][0]["content"] == "a<5mm?"
    assert out[0]["messages"][1]["reasoning_content"] == "mail <PII_EMAIL>"
    assert "x()" not in out[1]["text"] and "<think>t</think>" in out[1]["text"]
    assert st.modified["html_stripped"] == 1 and st.extra["pii_remaining"] == 0


# ---------------------------------------------------------------- ingest (F3e, F7, rv20)
def test_f7_mcq_ingest_relabels_numeric_choices():
    from foundationskills.skills.data_engine.ops.ingest import _map_record
    rec = _map_record({"question": "Q", "choices": {"text": ["x", "y"], "label": ["1", "2"]}, "answerKey": "2"},
                      source_uri="u", index=0, options={})
    assert [c["label"] for c in rec["choices"]] == ["A", "B"] and rec["answer"] == "B"


def test_rv20_non_dict_turns_counted():
    from foundationskills.skills.data_engine.ops.ingest import _normalize_messages
    dropped: list[int] = []
    assert len(_normalize_messages([{"role": "user", "content": "a"}, "junk"], dropped)) == 1 and dropped == [1]


# ---------------------------------------------------------------- format (F3f, F7, BOS)
class _FakeTok:
    """Mimics a template that DROPS reasoning_content (Gemma-4 behaviour)."""
    chat_template = "fake"
    bos_token = "<bos>"

    def apply_chat_template(self, messages, tokenize=False, **kw):
        head = "<|turn>system\n<|think|>\n<turn|>\n" if kw.get("enable_thinking") else ""
        body = "".join(f"<|turn>{'model' if m['role'] == 'assistant' else 'user'}\n{m['content']}<turn|>\n"
                       for m in messages)
        return "<bos>" + head + body


def test_f3f_gemma4_trace_is_injected_and_verified():
    from foundationskills.skills.data_engine.ops.format import render_with_reasoning
    msgs = [{"role": "user", "content": "Q"}, {"role": "assistant", "content": "A", "reasoning_content": "T"}]
    text, mode = render_with_reasoning(msgs, family="gemma4", tokenizer=_FakeTok())
    assert mode == "injected" and "<|turn>model\n<|channel>thought\nT\n<channel|>A" in text
    assert render_with_reasoning(msgs, family=None, tokenizer=_FakeTok())[1] == "lost"


def test_f3f_lost_trace_record_is_dropped_and_bos_stripped():
    from foundationskills.skills.data_engine.ops.format import _format_records
    st = OpStats("format")
    msgs = [{"role": "user", "content": "Q"}, {"role": "assistant", "content": "A", "reasoning_content": "T"}]
    out = list(_format_records([{"id": "1", "messages": msgs}],
                               {"target_format": "sft", "tokenizer": _FakeTok()}, st))
    assert out == [] and st.dropped["reasoning_lost_in_template"] == 1
    st = OpStats("format")
    out = list(_format_records([{"id": "2", "messages": msgs[:1] + [{"role": "assistant", "content": "A"}]}],
                               {"target_format": "sft", "tokenizer": _FakeTok()}, st))
    assert not out[0]["text"].startswith("<bos>")  # FS's tokenizer call adds BOS itself


# ---------------------------------------------------------------- capabilities (F7, F16, rv0, rv1)
def test_f16_rl_checkpoint_and_f7_answer_kind_gate_rl():
    c = caps()
    assert "no checkpoint" in c.check("rl", algorithm="dr_grpo")
    assert c.check("rl", algorithm="dr_grpo", require_checkpoint=False) is None
    assert "free_form" in c.check("rl", algorithm="dr_grpo", require_checkpoint=False, answer_kind="free_form")
    assert "unknown stage" in c.check("merge")
    assert "tp unmeasured" in caps(axes_measured=False, refused_axes=()).check("sft", tp=2)


# ---------------------------------------------------------------- emit (F1, F2, F11-13, rv2, rv17, rv18)
def _emit(stage, **kw):
    from foundationskills.interfaces.fs.emit_train import emit_train
    ds = {"format": "sft", "shards": [{"path": "/d/shards/shard-00000.jsonl"}], "num_tokens": 10_000_000}
    hw = {"id": "gb200-189gb", "scheduler": "slurm", "env": {"NCCL_MNNVL_ENABLE": "0"},
          "bf16_dense_tflops": 1503.4, "peak_provenance": "measured"}
    return emit_train(stage, dataset=ds, model="/m", output_dir="/o", hardware=hw, nodes=1,
                      gpus_per_node=kw.get("gpus", 4), caps=kw.get("caps", caps()), run_name=kw.get("name", "run"))


def test_f1_learning_rate_and_seq_len_aliases_reach_fs():
    spec = _emit({"stage": "sft", "method": "full", "hparams": {"learning_rate": 1e-4, "max_sequence_length": 4096,
                                                                "grad_ckpt": True, "epochs": 1}})
    argv = spec["argv"]
    assert argv[argv.index("--learning-rate") + 1] == "0.0001"
    assert argv[argv.index("--max-sequence-length") + 1] == "4096"
    assert argv[argv.index("--gradient-checkpointing") + 1] == "true"
    assert "--max-steps" in argv  # epochs x dataset tokens -> steps


def test_f12_f13_env_and_standalone_torchrun_and_rv17_meta_block():
    spec = _emit({"stage": "sft", "method": "full", "hparams": {"max_steps": 5},
                  "fs": {"args": {"dry_run": True, "model": "/evil"}}})
    assert spec["env"]["NCCL_MNNVL_ENABLE"] == "0" and spec["env"]["FS_DEVICE_PEAK_TFLOPS"] == "1503.4"
    assert "--standalone" in spec["argv"] and "$MASTER_ADDR:29500" not in spec["argv"]
    assert spec["argv"].count("--dry-run") == 0 and "/evil" not in spec["argv"]


def test_f11_invalid_flag_value_is_named():
    c = caps(train_flag_choices={"--gradient-checkpointing": ("true", "false"), "--sharding-strategy": ("ddp",)})
    spec = _emit({"stage": "sft", "method": "full", "hparams": {"max_steps": 5, "sharding": "fsdp"}}, caps=c)
    assert spec["executable"] is False and "--sharding-strategy value 'fsdp'" in spec["missing"]


def test_rv18_unsafe_run_name_refused():
    with pytest.raises(ValueError):
        _emit({"stage": "sft", "hparams": {"max_steps": 1}}, name="a;rm -rf")


def test_rv2_sbatch_expands_master_addr_and_cds_to_repo_root():
    from foundationskills.interfaces.fs.sbatch import render_sbatch
    text = render_sbatch(["torchrun", "--rdzv-endpoint", "${MASTER_ADDR}:29500", "-m", "x"], env={},
                         hardware_id="gb200-189gb", nodes=2, gpus_per_node=4, job_name="j", log_dir="/l",
                         launcher="torchrun", cwd="/repo root", cpus_per_task=30)
    assert '"${MASTER_ADDR}:29500"' in text and "'${MASTER_ADDR}" not in text
    assert "cd '/repo root'" in text and "--cpus-per-task=30" in text
    with pytest.raises(ValueError):
        render_sbatch(["python"], env={}, hardware_id="x", nodes=1, gpus_per_node=1, job_name="bad name",
                      log_dir="/l", launcher="python")


def test_f4_stage_chaining_names_the_lora_merge_and_rl_gaps(tmp_path):
    from foundationskills.core import SkillContext
    from foundationskills.skills.training.emit_skill import TrainingEmitSkill
    ds = {"format": "sft", "shards": [{"path": f"{tmp_path}/shards/shard-00000.jsonl", "sha256": "0" * 64,
                                       "records": 1}], "num_records": 1, "schema": {"columns": ["text"]},
          "num_tokens": 1000, "tokenizer": None, "chat_template_family": None,
          "fs_columns": {"text_column": "text", "image_column": None, "gold_key": None}}
    plan = {"stages": [
        {"name": "sft", "stage": "sft", "algorithm": "sft", "method": "lora", "hparams": {"max_steps": 1, "max_sequence_length": 512}},
        {"name": "sft2", "stage": "sft", "algorithm": "sft", "method": "full", "hparams": {"max_steps": 1, "max_sequence_length": 512}}]}
    res = TrainingEmitSkill().execute({"plan": plan, "dataset": ds, "model": "/m", "output_root": str(tmp_path),
                                       "nodes": 1, "gpus_per_node": 1, "hardware_id": "local"},
                                      SkillContext(workdir=tmp_path, capabilities=caps()))
    second = res.payload["specs"][1]
    assert second["model_source"] == "stage:sft" and "merge" in second["missing"]
    assert second["argv"][second["argv"].index("--model") + 1].endswith("fskills-sft/final")


def test_f16_emit_rl_is_measurement_only():
    from foundationskills.interfaces.fs.emit_rl import emit_rl
    spec = emit_rl({"name": "rl", "stage": "rl", "algorithm": "dr_grpo", "hparams": {"answer_pattern": "x"}},
                   dataset={"format": "rl", "shards": [{"path": "/d/shards/s.jsonl"}],
                            "fs_columns": {"gold_key": "answer"}},
                   model="/m", output_dir="/o", caps=caps(), run_name="rl")
    assert spec["executable"] is False and "no checkpoint" in spec["missing"]
    assert spec["measurement_only_ok"] is True and "answer_pattern" not in spec["rl_config"]


# ---------------------------------------------------------------- driver / launch (F6, F2, rv16)
def test_f6_fs_refusal_classes_map_to_96():
    from foundationskills.interfaces.fs import rl_driver
    BatchRefusal = type("BatchRefusal", (Exception,), {"__module__": "foundationscale.rl.interfaces"})
    assert rl_driver._is_fs_refusal(BatchRefusal("x"))
    assert not rl_driver._is_fs_refusal(ValueError("x"))
    assert rl_driver._exit_from_systemexit(SystemExit(3)) == 5
    assert rl_driver._exit_from_systemexit(SystemExit(96)) == 96


def test_f2_launch_runs_in_spec_cwd_with_spec_env_and_measurement_gate():
    from foundationskills.core import plan_hash
    from foundationskills.interfaces.fs.launch import LaunchRefused, launch
    calls = []

    def runner(argv, **kw):
        calls.append(kw)
        return types.SimpleNamespace(returncode=0, stdout="", stderr="")
    spec = {"stage_name": "rl", "entry": "fskills-rl", "argv": ["fskills-rl"], "env": {"A": "1"}, "sbatch": None,
            "dry_run_argv": None, "expected_outputs": [], "executable": False, "missing": "no checkpoint",
            "measurement_only_ok": True, "cwd": "/repo"}
    with pytest.raises(LaunchRefused):
        launch(spec, confirm=plan_hash(spec), runner=runner)
    launch(spec, confirm=plan_hash(spec), runner=runner, measurement_only=True)
    assert calls[-1]["cwd"] == "/repo" and calls[-1]["env"]["A"] == "1"


# ---------------------------------------------------------------- estimator (F14, F15, rv12, EXTRA-2)
def _variant():
    return types.SimpleNamespace(total_params=8e9, active_params=4e9, arch="dense", hidden=2560, layers=42,
                                 heads=8, vocab=262144, id="e4b")


def test_f14_logits_term_matches_gb200_measurement():
    from foundationskills.skills.training.estimate import estimate_memory
    est = estimate_memory(_variant(), method="full", seq_len=4096, micro_batch=1, grad_ckpt=True,
                          sharding="fsdp", world=4)
    assert est.total_per_gpu_gb == pytest.approx(48.3, rel=0.15)  # measured peak allocated on 4x GB200


def test_rv12_fsdp_divisor_is_world_not_world_times_tp():
    from foundationskills.skills.training.estimate import estimate_memory
    a = estimate_memory(_variant(), method="full", seq_len=512, micro_batch=1, sharding="fsdp", world=8, tp=2)
    b = estimate_memory(_variant(), method="full", seq_len=512, micro_batch=1, sharding="fsdp", world=8, tp=1)
    assert a.optimizer_gb == pytest.approx(b.optimizer_gb)


def test_f15_mfu_point_by_config_and_rl_factor():
    from foundationskills.skills.training.knowledge import load_hardware
    from foundationskills.skills.training.estimate import estimate_time
    hw = load_hardware()["gb200-189gb"]
    fsdp = estimate_time(_variant(), tokens=10**6, hardware=hw, gpus=4, sharding="fsdp", micro_batch=1,
                         grad_ckpt=True)
    ddp = estimate_time(_variant(), tokens=10**6, hardware=hw, gpus=4, sharding="ddp", micro_batch=2,
                        grad_ckpt=False)
    assert (fsdp.mfu, fsdp.mfu_provenance) == (0.046, "measured") and ddp.mfu == 0.318
    rl = estimate_time(_variant(), tokens=10**6, hardware=hw, gpus=1, method="lora", stage="rl")
    lora = estimate_time(_variant(), tokens=10**6, hardware=hw, gpus=1, method="lora")
    assert rl.total_flops == pytest.approx(3.0 * lora.total_flops)


# ---------------------------------------------------------------- planner (F7, F8, rv13)
def test_rv13_non_numeric_goal_facts_refuse_by_name():
    from foundationskills.skills.training.planner import PlanningRefusal, plan
    goal = {"objective": "x", "target_capabilities": ["reasoning"], "base_model": {"name_or_path": "m", "size_b": 7},
            "data": {"sources": [{"uri": "u", "kind": "jsonl", "approx_examples": "lots"}]},
            "hardware": {"gpu": "NVIDIA GB200", "gpus_per_node": 4, "nodes": 1}}
    with pytest.raises(PlanningRefusal, match="approx_examples"):
        plan(goal, caps=caps())


# ---------------------------------------------------------------- round-2b (rv5-11, rv14, rv19, rv35, rv40-42, F10)
def test_rv5_rv6_decontam_short_items_and_empty_benchmarks(tmp_path):
    from foundationskills.skills.data_engine.ops.decontam import _decontam_op
    bench = tmp_path / "b.jsonl"
    bench.write_text(json.dumps({"q": "What is two plus two?"}) + "\n")
    empty = tmp_path / "e.jsonl"
    empty.write_text(json.dumps({"q": "   "}) + "\n")
    st = OpStats("decontam")
    out = list(_decontam_op([{"text": "What is two plus two?"}, {"text": "unrelated text here"}],
                            {"benchmarks": ["b", "e"], "sources": {"b": str(bench), "e": str(empty)}}, st))
    assert len(out) == 1 and st.extra["decontam_hits"] == 1       # short leaked item caught whole
    assert "e" in st.extra["unmeasured_benchmarks"] and "e" not in st.extra["benchmarks_checked"]


def test_rv7_no_evaluated_rule_is_unmeasured_not_perfect():
    from foundationskills.skills.data_engine.ops.quality import _quality_op
    st = OpStats("quality")
    out = list(_quality_op([{"text": "x"}], {"heuristics": "none", "min_quality_score": 0.9}, st))
    assert out and out[0]["meta"]["quality"]["score"] is None and st.modified["quality_unmeasured"] == 1


def test_rv8_rv9_rv27_dedup_semantics():
    from foundationskills.skills.data_engine.ops.dedup import _dedup_op, _record_key
    a = {"messages": [{"role": "user", "content": "a\nb"}]}
    b = {"messages": [{"role": "user", "content": "a"}, {"role": "user", "content": "b"}]}
    assert _record_key(a, "auto") != _record_key(b, "auto")
    st = OpStats("dedup")
    out = list(_dedup_op([{"text": "x = 1"}, {"text": "x 1"}], {"near": {"enabled": False}}, st))
    assert len(out) == 2  # light normalization keeps punctuation-distinct records
    with pytest.raises(ValueError, match="divide"):
        list(_dedup_op([{"text": "x"}], {"near": {"num_perm": 100, "bands": 16}}, OpStats("dedup")))


def test_rv14_orchestrator_keeps_prior_artifacts_with_explicit_inputs():
    import inspect
    from foundationskills.core import orchestrator
    assert 'request["inputs"] = inputs' in inspect.getsource(orchestrator.Orchestrator.run)


def test_rv19_artifact_ids_cannot_escape_directory(tmp_path):
    from foundationskills.core import Artifact, make_provenance, write_artifact
    art = Artifact(type="goal_spec", id="../evil", payload={}, provenance=make_provenance("t", "0"))
    with pytest.raises(ValueError):
        write_artifact(art, tmp_path)


def test_rv35_mix_refuses_duplicate_component_names():
    from foundationskills.skills.data_engine.ops.mix import _mix as _mix_op
    with pytest.raises(ValueError, match="unique"):
        list(_mix_op([], {"components": [{"name": "a", "path": "/x.jsonl", "ratio": 0.5},
                                         {"name": "a", "path": "/y.jsonl", "ratio": 0.5}], "seed": 0}, OpStats("mix")))


def test_rv41_cpt_unknown_size_refuses():
    from foundationskills.skills.training.cpt import cpt_policy
    with pytest.raises(ValueError, match="size_b"):
        cpt_policy(types.SimpleNamespace(size_b=None), 10**9, "domain_expert", True)


def test_f10_drop_overlong_counts_drops():
    from foundationskills.skills.data_engine.ops.tokenize import _tokenize_records
    st = OpStats("tokenize")
    # tokenize counts the rendered `text` the format op writes upstream
    recs = [{"messages": [], "text": "w " * 2000}, {"messages": [], "text": "hi"}]
    out = list(_tokenize_records(recs, {"seq_len": 256, "drop_overlong": True}, st))
    assert len(out) == 1 and st.dropped["overlong"] == 1 and st.extra["truncation_rate"] == 0.0


def test_every_recommended_op_config_matches_the_op_schema():
    """Found on real data: the recommender sent {"ruleset": ...}/{"pii_mode": ...}
    which the ops silently ignored, so quality measured nothing on 23k records."""
    from foundationskills.core.schema import validate
    from foundationskills.skills.data_engine.ops import OPS
    from foundationskills.skills.data_engine.recommend import recommend_pipeline
    for fmt in ("pretrain", "cpt", "sft", "mm_sft", "preference", "rl"):
        for benchmarks in ([], ["gsm8k"]):
            spec = recommend_pipeline(target_format=fmt, sources=[{"uri": "/x", "kind": "jsonl"}], goal="reasoning",
                                      algorithm=None, tokenizer="/t", chat_template_family="gemma4", domain="d",
                                      benchmarks=benchmarks)
            for step in spec["ops"]:
                assert validate(step["config"], OPS[step["op"]].config_schema) == [], (fmt, step)


def test_pipeline_refuses_foreign_op_config_keys(tmp_path):
    """MUST_FIRE: an unknown key must fail loudly, not be ignored."""
    from foundationskills.skills.data_engine.pipeline import PipelineError, run_pipeline
    src = tmp_path / "c.jsonl"
    src.write_text(json.dumps({"text": "hello"}) + "\n")
    spec = {"target_format": "cpt", "seed": 0, "tokenizer": None, "rationale": [],
            "ops": [{"op": "ingest", "config": {"sources": [{"uri": str(src), "kind": "jsonl"}]}},
                    {"op": "quality", "config": {"ruleset": "gopher"}}]}
    with pytest.raises(PipelineError, match="ruleset"):
        run_pipeline(spec, tmp_path / "out")


def test_emitted_config_is_the_estimated_config_and_conflicts_refused():
    """E2E on GB200: the planner estimated with grad ckpt on but did not write it
    into the stage, so FS ran its default (off): 63 GB measured vs 32 GB planned."""
    spec = _emit({"stage": "sft", "method": "lora", "hparams": {"max_steps": 1}})
    assert "max_sequence_length not planned" in spec["missing"]  # FS default would be 128 tokens
    spec = _emit({"stage": "sft", "method": "lora",
                  "hparams": {"max_steps": 1, "seq_len": 4096, "max_sequence_length": 2048}})
    assert "conflicting hparams for --max-sequence-length" in spec["missing"]


def test_f19_launch_keeps_output_and_recovers_fs_verdict_through_torchrun(tmp_path):
    from foundationskills.core import plan_hash
    from foundationskills.interfaces.fs.launch import fs_verdict, launch

    def runner(argv, **kw):  # torchrun: child refused (96) but torchrun exits 1
        return types.SimpleNamespace(returncode=1, stdout="[fs:train:refuse]  cluster profile refused\n", stderr="")
    spec = {"stage_name": "s", "entry": "foundationscale-train", "argv": ["torchrun"], "env": {}, "sbatch": None,
            "dry_run_argv": None, "expected_outputs": [], "executable": True, "missing": None,
            "output_dir": str(tmp_path / "run")}
    res = launch(spec, confirm=plan_hash(spec), submit=False, runner=runner)
    assert res["returncode"] == 96 and res["returncode_raw"] == 1 and res["fs_verdict"] == 96
    assert (tmp_path / "run" / "fskills_launch.log").read_text().startswith("[fs:train:refuse]")
    assert fs_verdict("[fs:train:done]          PASS") == 0 and fs_verdict("no fs lines") is None


def test_every_recipe_fs_value_is_legal_for_the_installed_stack():
    """GB200 E2E: recipe optimizer "adamw" passed FS's parser (no choices) and was
    refused by transformers at TrainingArguments construction (exit 96, after
    GPUs were allocated). Values are now checked against measured vocabularies."""
    from foundationskills.interfaces.fs.capabilities import probe
    from foundationskills.skills.training.knowledge import load_recipes
    choices = probe().train_flag_choices
    for recipe in load_recipes():
        for stage in recipe.raw.get("stages", []):
            for key, value in ((stage.get("fs") or {}).get("args") or {}).items():
                flag = "--" + str(key).lstrip("-").replace("_", "-")
                allowed = choices.get(flag)
                if allowed:
                    rendered = ("true" if value else "false") if isinstance(value, bool) else str(value)
                    assert rendered in allowed, f"{recipe.raw['id']}: {flag}={rendered!r} not in {allowed[:6]}..."


def test_lora_targets_string_emits_one_flag_per_module():
    """Reference scenario: a recipe's "q_proj,k_proj,..." string was iterated per
    CHARACTER, emitting "--adapter-target j" dozens of times."""
    c = caps(train_flags=caps().train_flags | {"--adapter-target"}, families={})
    spec = _emit({"stage": "sft", "method": "lora", "family": "llama3",
                  "hparams": {"max_steps": 1, "max_sequence_length": 512, "lora_targets": "q_proj,k_proj, v_proj"}},
                 caps=c)
    argv = spec["argv"]
    targets = [argv[i + 1] for i, a in enumerate(argv) if a == "--adapter-target"]
    assert targets == ["q_proj", "k_proj", "v_proj"]


@pytest.mark.parametrize("method,micro,measured", [("full", 1, 48.3), ("lora", 2, 37.7), ("lora", 4, 75.9)])
def test_f14_memory_estimate_tracks_three_gb200_measurements(method, micro, measured):
    """Gemma-4 E4B on 4x GB200, FSDP, seq 4096, grad ckpt: peak allocated per GPU
    measured by FS (2026-09-26). The estimator must stay within 15%."""
    from foundationskills.skills.training.estimate import estimate_memory
    est = estimate_memory(_variant(), method=method, seq_len=4096, micro_batch=micro, grad_ckpt=True,
                          sharding="fsdp", world=4)
    assert est.total_per_gpu_gb == pytest.approx(measured, rel=0.15)
