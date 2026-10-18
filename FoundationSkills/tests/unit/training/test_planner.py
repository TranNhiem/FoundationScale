
"""Planner unit tests with the D1-owned modules monkeypatched to simple fakes.

These isolate planner logic (rule evaluation, decision logging, executability,
schema validity). The end-to-end path with the real knowledge base lives in
tests/scenario/test_reference_scenario.py.
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest

from foundationskills.core import load_schema, validate
from foundationskills.interfaces.fs.capabilities import FSCapabilities
from foundationskills.skills.training import planner
from foundationskills.skills.training.planner import PLANNING_STEPS, PlanningRefusal


def _fake_caps(**over) -> FSCapabilities:
    base = dict(
        available=True,
        fs_version="0.0.0+77bfa65",
        train_flags=frozenset({"--model", "--dataset", "--output-dir", "--dry-run", "--adapter-target"}),
        train_objectives=("sft",),
        sharding_strategies=("ddp", "fsdp"),
        executed_axes=("tp", "cp"),
        refused_axes=("pp", "ep"),
        axes_measured=True,
        rl_algorithms=("dr_grpo", "gspo", "dapo", "grpo"),
        rl_runnable={"dr_grpo": None, "gspo": None, "dapo": None, "grpo": "refused: needs a reference model"},
        families={"gemma4": ("gemma4",), "qwen3.5": ("qwen3_5",)},
        backends=("ddp", "fsdp"),
    )
    base.update(over)
    return FSCapabilities(**base)


def _goal(**over) -> dict:
    goal = {
        "objective": "manufacturing-domain reasoning model",
        "target_capabilities": ["reasoning"],
        "goal": "domain_expert",
        "domain": "manufacturing",
        "preserve_general": True,
        "base_model": {"name_or_path": "Llama-3.1-8B", "family": "llama3", "size_b": 8.0, "arch": "dense"},
        "data": {
            "sources": [
                {"uri": "file:///data/corpus", "kind": "local_dir", "approx_tokens": 2_000_000_000},
                {"uri": "file:///data/qa.jsonl", "kind": "jsonl", "approx_examples": 20_000},
            ]
        },
        "hardware": {"gpu": "NVIDIA H100", "gpus_per_node": 8, "nodes": 2},
        "data_facts": {"has_instructions": True, "has_pairs": True, "has_verifiable_answers": True},
    }
    goal.update(over)
    return goal


def _rules() -> list[dict]:
    return [
        {"id": "SR-CPT", "when": {"domain_tokens_gte": 1}, "action": "add_stage", "stage": "cpt", "algorithm_hint": "causal_lm", "because": "domain corpus present"},
        {"id": "SR-SFT", "when": {"has_instructions": True}, "action": "add_stage", "stage": "sft", "algorithm_hint": "sft", "because": "instruction data present"},
        {"id": "SR-RL", "when": {"has_verifiable_answers": True}, "action": "add_stage", "stage": "rl", "algorithm_hint": "grpo", "because": "verifiable answers"},
        {"id": "SR-SKIP-PREF", "when": {"has_verifiable_answers": True}, "action": "skip_stage", "stage": "preference", "algorithm_hint": None, "because": "RL supersedes pairs"},
    ]


@pytest.fixture()
def patched(monkeypatch):
    family = SimpleNamespace(name="llama3", fs_family=None, raw={"name": "llama3"})
    variant = SimpleNamespace(
        id="Llama-3.1-8B", hf_id="meta-llama/Llama-3.1-8B", local_path=None,
        size_b=8.0, active_b=8.0, arch="dense", instruct=False,
        hidden=4096, layers=32, vocab=128256,
    )
    hw = SimpleNamespace(id="h100", gpu_name="NVIDIA H100", mem_gb=80.0, bf16_dense_tflops=989.0, gpus_per_node=8)
    recipe = SimpleNamespace(
        id="rcp-llama3-cpt",
        raw={
            "index": {"method": "full"},
            "stages": [{"name": "cpt", "stage": "cpt", "hparams": {"max_sequence_length": 4096}}],
            "provenance": {"status": "literature", "evidence": ["paper"]},
        },
    )
    match = SimpleNamespace(recipe=recipe, score=0.9, reasons=["stage+arch exact"], derived=True, adjustments=["size extrapolated"])

    monkeypatch.setattr(planner, "find_variant", lambda model: (family, variant))
    monkeypatch.setattr(planner, "load_hardware", lambda: {"h100": hw, "gb200": hw})
    monkeypatch.setattr(planner, "load_stage_rules", _rules)
    monkeypatch.setattr(planner, "select_recipe", lambda q: match)
    monkeypatch.setattr(planner, "rank_recipes", lambda q, k=3: [match])
    monkeypatch.setattr(
        planner,
        "select_method",
        lambda **kw: SimpleNamespace(
            method=kw.get("prefer") or ("lora" if kw["stage"] in ("sft", "preference") else "full"),
            because=["fixture method rule"],
            alternatives=[],
        ),
    )
    monkeypatch.setattr(
        planner,
        "estimate_memory",
        lambda variant, **kw: SimpleNamespace(
            weights_gb=16.0, grads_gb=16.0, optimizer_gb=64.0, activations_gb=8.0,
            overhead_gb=4.0, total_per_gpu_gb=60.0, assumptions=["fixture"],
        ),
    )
    monkeypatch.setattr(
        planner,
        "estimate_time",
        lambda variant, **kw: SimpleNamespace(
            total_flops=6e19, tokens_per_s_per_gpu=50_000.0, hours=10.0,
            gpu_hours=float(kw["gpus"]) * 10.0, mfu=0.318, mfu_provenance="measured",
            assumptions=["fixture 6ND"],
        ),
    )
    monkeypatch.setattr(
        planner,
        "check_feasibility",
        lambda variant, hw_, **kw: SimpleNamespace(verdict="ok", findings=[], alternatives=[]),
    )
    return SimpleNamespace(family=family, variant=variant, hw=hw, recipe=recipe)


def test_plan_end_to_end_with_fakes(patched) -> None:
    payload = planner.plan(_goal(), caps=_fake_caps())

    stages = payload["stages"]
    assert [s["stage"] for s in stages] == ["cpt", "sft", "rl"]

    cpt = stages[0]
    assert cpt["method"] == "full"
    assert 0.05 <= cpt["hparams"]["replay_ratio"] <= 0.30
    assert cpt["hparams"]["learning_rate"] == 2e-5
    assert cpt["executable"] is True and cpt["missing"] is None

    rl = stages[2]
    assert rl["algorithm"] == "dr_grpo"
    assert "reference-free" in rl["because"]
    assert rl["estimate"]["gpus"] == 1  # FS RLTrainer is single-device

    assert all(s["because"] for s in stages)
    assert all(s["estimate"]["gpu_hours"] > 0 for s in stages)
    assert payload["feasibility"]["verdict"] == "ok"

    steps = {d["step"] for d in payload["decisions"]}
    assert set(PLANNING_STEPS) <= steps

    # llama3 is not FS-registered: the adapter-target decision must be present
    assert any("adapter-target" in d["because"] for d in payload["decisions"] if d["step"] == "model_analysis")

    errors = validate(payload, load_schema("artifacts/training_plan"))
    assert errors == []


def test_data_handoff_present_when_no_readiness(patched) -> None:
    payload = planner.plan(_goal(), caps=_fake_caps())
    for stage in payload["stages"]:
        assert stage["data"]["handoff"] == "data_engine"
        assert stage["data"]["target_format"] == stage["data"]["format"]


def test_readiness_pass_marks_matching_data_ready(patched) -> None:
    readiness = {"verdict": "PASS", "format": "cpt", "stats": {"num_tokens": 2_000_000_000}}
    payload = planner.plan(_goal(), caps=_fake_caps(), readiness=readiness)
    cpt = next(s for s in payload["stages"] if s["stage"] == "cpt")
    assert cpt["data"]["ready"] is True
    assert "handoff" not in cpt["data"]


def test_skip_rules_override_adds(patched, monkeypatch) -> None:
    monkeypatch.setattr(
        planner,
        "load_stage_rules",
        lambda: [
            {"id": "A", "when": {}, "action": "add_stage", "stage": "sft", "algorithm_hint": None, "because": "always"},
            {"id": "B", "when": {}, "action": "add_stage", "stage": "rl", "algorithm_hint": None, "because": "always"},
            {"id": "C", "when": {}, "action": "skip_stage", "stage": "rl", "algorithm_hint": None, "because": "not today"},
        ],
    )
    payload = planner.plan(_goal(), caps=_fake_caps())
    assert [s["stage"] for s in payload["stages"]] == ["sft"]


def test_no_stage_selected_refuses_with_details(patched, monkeypatch) -> None:
    monkeypatch.setattr(
        planner,
        "load_stage_rules",
        lambda: [{"id": "X", "when": {"has_pairs": True}, "action": "add_stage", "stage": "preference", "algorithm_hint": None, "because": "pairs"}],
    )
    goal = _goal(data_facts={"has_instructions": False, "has_pairs": False, "has_verifiable_answers": False})
    with pytest.raises(PlanningRefusal, match="no stage selected"):
        planner.plan(goal, caps=_fake_caps())


def test_unknown_model_without_size_refuses(patched, monkeypatch) -> None:
    monkeypatch.setattr(planner, "find_variant", lambda model: None)
    goal = _goal()
    goal["base_model"] = {"name_or_path": "mystery/model"}
    with pytest.raises(PlanningRefusal, match="base_model.size_b"):
        planner.plan(goal, caps=_fake_caps())


ARCH = {"hidden": 4096, "layers": 32, "heads": 32, "kv_heads": 8, "vocab": 128256,
        "context_length": 131072}


def test_unknown_model_with_size_synthesizes_variant(patched, monkeypatch) -> None:
    monkeypatch.setattr(planner, "find_variant", lambda model: None)
    monkeypatch.setattr(planner, "load_hardware", lambda: {"h100": patched.hw})
    goal = _goal()
    # stated: there is no local checkpoint to read
    goal["base_model"] = dict(goal["base_model"], architecture=ARCH)
    payload = planner.plan(goal, caps=_fake_caps())
    assert payload["model"]["in_knowledge_base"] is False
    assert any("not in the knowledge base" in n for n in payload["provenance_notes"])


def test_unknown_hardware_refuses_naming_known_ids(patched, monkeypatch) -> None:
    monkeypatch.setattr(planner, "load_hardware", lambda: {"h100": patched.hw})
    goal = _goal(hardware={"gpu": "TPU v5p", "gpus_per_node": 8, "nodes": 1})
    with pytest.raises(PlanningRefusal, match="h100"):
        planner.plan(goal, caps=_fake_caps())


def test_missing_base_model_refuses(patched) -> None:
    with pytest.raises(PlanningRefusal, match="base_model.name_or_path"):
        planner.plan(_goal(base_model={}), caps=_fake_caps())


def test_fs_refusal_marks_stage_non_executable(patched) -> None:
    caps = _fake_caps(available=False, train_objectives=(), backends=())
    payload = planner.plan(_goal(), caps=caps)
    assert all(s["executable"] is False for s in payload["stages"])
    assert all(s["missing"] for s in payload["stages"])


def _override_rules() -> list[dict]:
    return [
        {"id": "ADD-SFT", "when": {}, "action": "add_stage", "stage": "sft", "algorithm_hint": None, "because": "always"},
        {"id": "SKIP-CPT", "when": {}, "action": "skip_stage", "stage": "cpt", "algorithm_hint": None, "because": "too little data"},
    ]


def test_goal_stages_overrides_rules_and_logs_why(patched, monkeypatch) -> None:
    monkeypatch.setattr(planner, "load_stage_rules", _override_rules)
    payload = planner.plan(_goal(stages=["cpt"]), caps=_fake_caps())
    assert [s["stage"] for s in payload["stages"]] == ["cpt"]
    overrides = [d for d in payload["decisions"] if d["step"] == "operator_override"]
    assert {d["choice"] for d in overrides} == {"+cpt", "-sft"}
    added = next(d for d in overrides if d["choice"] == "+cpt")
    assert "SKIP-CPT" in added["because"] and "too little data" in added["because"]


def test_goal_stages_in_lifecycle_order_without_overrides(patched, monkeypatch) -> None:
    monkeypatch.setattr(planner, "load_stage_rules", _override_rules)
    payload = planner.plan(_goal(stages=["sft", "cpt", "sft"]), caps=_fake_caps())
    assert [s["stage"] for s in payload["stages"]] == ["cpt", "sft"]
    assert [d["choice"] for d in payload["decisions"] if d["step"] == "operator_override"] == ["+cpt"]


@pytest.mark.parametrize("bad", [[], "cpt", ["cpt", "warmup"], [1]])
def test_goal_stages_invalid_refuses(patched, bad) -> None:
    with pytest.raises(PlanningRefusal, match="goal.stages"):
        planner.plan(_goal(stages=bad), caps=_fake_caps())


def test_goal_method_overrides_rules_and_logs_why(patched, monkeypatch) -> None:
    monkeypatch.setattr(planner, "load_stage_rules", _override_rules)
    ruled = planner.plan(_goal(stages=["sft"]), caps=_fake_caps())
    rule_method = ruled["stages"][0]["method"]
    forced = "full" if rule_method != "full" else "lora"
    payload = planner.plan(_goal(stages=["sft"], method=forced), caps=_fake_caps())
    assert payload["stages"][0]["method"] == forced
    override = [d for d in payload["decisions"] if d["step"] == "operator_override" and "method" in d["choice"]]
    assert len(override) == 1 and f"chose {rule_method}" in override[0]["because"]


@pytest.mark.parametrize("bad", ["adapter", 1, ""])
def test_goal_method_invalid_refuses(patched, bad) -> None:
    with pytest.raises(PlanningRefusal, match="goal.method"):
        planner.plan(_goal(method=bad), caps=_fake_caps())


def test_lora_rank_reaches_the_memory_estimate(patched, monkeypatch) -> None:
    seen: list[int] = []
    real = planner.estimate_memory
    monkeypatch.setattr(planner, "estimate_memory", lambda *a, **kw: (seen.append(kw.get("lora_rank")), real(*a, **kw))[1])
    monkeypatch.setattr(planner, "_recipe_stage_hparams", lambda raw, stage: {"lora_rank": 64})
    planner.plan(_goal(stages=["sft"]), caps=_fake_caps())
    assert seen and set(seen) == {64}


def test_lora_cpt_scales_the_full_ft_lr_and_says_why(patched) -> None:
    payload = planner.plan(_goal(stages=["cpt"], method="lora"), caps=_fake_caps())
    cpt = next(s for s in payload["stages"] if s["stage"] == "cpt")
    assert cpt["method"] == "lora"
    assert cpt["hparams"]["learning_rate"] == pytest.approx(2e-5 * planner.LORA_LR_MULTIPLIER)
    why = [d for d in payload["decisions"] if d["step"] == "hparams" and "learning_rate" in d["choice"]]
    assert len(why) == 1 and "full-FT" in why[0]["because"]


def test_stage_gap_blocks_every_alternative_and_is_named() -> None:
    # An alternative changes a memory knob; it cannot close a capability gap such as
    # "RLTrainer persists no checkpoint", so it must not read as a way forward.
    alts = [
        {"change": "micro_batch 2 -> 1", "executable": True, "missing": None, "verdict": "infeasible"},
        {"change": "tp=2", "executable": False, "missing": "missing: tp", "verdict": "ok"},
    ]
    gated = planner._gate_alternatives(alts, "missing: FS RLTrainer does not persist the trained policy")
    assert all(a["executable"] is False for a in gated)
    assert gated[0]["missing"] == "missing: FS RLTrainer does not persist the trained policy"
    assert gated[1]["missing"] == "missing: tp; missing: FS RLTrainer does not persist the trained policy"
    assert alts[0]["executable"] is True  # input is not mutated


def test_no_stage_gap_leaves_alternatives_alone() -> None:
    alts = [{"change": "micro_batch 2 -> 1", "executable": True, "missing": None}]
    assert planner._gate_alternatives(alts, None) == alts


def test_operator_stages_follow_an_explicit_stage_order() -> None:
    decisions: list = []
    selected = [{"stage": "sft", "rule_id": "R"}, {"stage": "cpt", "rule_id": "R"}]
    out = planner._apply_operator_stages(["cpt", "sft"], selected, [], [], lambda *a: decisions.append(a),
                                         stage_order=["sft", "cpt"])
    assert [s["stage"] for s in out] == ["sft", "cpt"]


def test_declared_rl_floor_reaches_only_the_rl_stage(patched) -> None:
    payload = planner.plan(_goal(rl_min_measured_fraction=0.4), caps=_fake_caps())
    by_stage = {s["stage"]: s for s in payload["stages"]}
    assert by_stage["rl"]["hparams"]["min_measured_fraction"] == 0.4
    assert "min_measured_fraction" not in by_stage["sft"]["hparams"]


def test_undeclared_rl_floor_is_not_guessed(patched) -> None:
    payload = planner.plan(_goal(), caps=_fake_caps())
    assert all("min_measured_fraction" not in s["hparams"] for s in payload["stages"])


def test_operator_rl_floor_is_not_clobbered_by_the_goal(patched, monkeypatch) -> None:
    original = planner._recipe_stage_hparams
    monkeypatch.setattr(planner, "_recipe_stage_hparams",
                        lambda raw, stage: {**original(raw, stage), **({"min_measured_fraction": 0.2} if stage == "rl" else {})})
    payload = planner.plan(_goal(rl_min_measured_fraction=0.8), caps=_fake_caps())
    by_stage = {s["stage"]: s for s in payload["stages"]}
    assert by_stage["rl"]["hparams"]["min_measured_fraction"] == 0.2


@pytest.mark.usefixtures("patched")
def test_unknown_model_without_architecture_names_remedies(monkeypatch) -> None:
    monkeypatch.setattr(planner, "find_variant", lambda _model: None)
    with pytest.raises(PlanningRefusal, match="base_model architecture") as err:
        planner.plan(_goal(), caps=_fake_caps())
    assert "config.json" in str(err.value) and "base_model.architecture" in str(err.value)


def test_unknown_local_checkpoint_reads_its_config_json(tmp_path) -> None:
    # measured 2026-10-17: Qwen3.5-4B (a VLM; facts under text_config) is no KB variant, and the old
    # SimpleNamespace stand-in crashed the planner (no total_params; None dims compared with ints)
    import json

    ckpt = tmp_path / "Qwen3.5-4B"
    ckpt.mkdir()
    (ckpt / "config.json").write_text(json.dumps({
        "model_type": "qwen3_5", "vision_config": {}, "tie_word_embeddings": True,
        "text_config": {"hidden_size": 2560, "num_hidden_layers": 32, "num_attention_heads": 16,
                        "num_key_value_heads": 4, "head_dim": 256, "vocab_size": 248320,
                        "max_position_embeddings": 262144}}))
    family, variant, note = planner._synthesize_variant(
        {"name_or_path": str(ckpt), "family": "qwen3.5", "size_b": 4.66})
    assert (variant.hidden, variant.layers, variant.heads, variant.kv_heads) == (2560, 32, 16, 4)
    assert variant.context_length == 262144 and variant.tied_embeddings is True
    assert variant.total_params == 4_660_000_000  # the property the stand-in lacked
    assert "from config.json" in note and variant in family.variants


def test_unknown_local_checkpoint_with_a_broken_config_refuses(tmp_path) -> None:
    (tmp_path / "config.json").write_text("{not json")
    with pytest.raises(PlanningRefusal, match="base_model architecture"):
        planner._synthesize_variant({"name_or_path": str(tmp_path), "size_b": 4.0})


def _caps_with_kernels(present: bool, status: str = "measured") -> SimpleNamespace:
    value = {"fla": present, "causal_conv1d": present}
    key = {"status": status, "value": value, "reason": "" if present else "missing"}
    return SimpleNamespace(probe_keys={"linear_attention_kernels": key})


def test_linear_attention_fallback_is_disclosed_only_when_measured_missing(tmp_path) -> None:
    import json

    (tmp_path / "config.json").write_text(json.dumps(
        {"text_config": {"layer_types": ["linear_attention", "full_attention"]}}))
    variant = SimpleNamespace(local_path=str(tmp_path))
    note = planner._linear_attention_note(str(tmp_path), variant, _caps_with_kernels(False))
    assert note and "torch fallback" in note and "time estimates assume the fast path" in note
    assert planner._linear_attention_note(str(tmp_path), variant, _caps_with_kernels(True)) is None
    unmeasured = _caps_with_kernels(False, "unmeasured")
    assert planner._linear_attention_note(str(tmp_path), variant, unmeasured) is None
    (tmp_path / "config.json").write_text(json.dumps({"layer_types": ["full_attention"]}))
    assert planner._linear_attention_note(str(tmp_path), variant, _caps_with_kernels(False)) is None


def test_rl_generation_budget_is_always_emitted(patched) -> None:
    # measured 2026-10-18: nothing emitted max_new_tokens, FS ran 64 and Qwen3.5-4B abstained on 32/32 steps
    payload = planner.plan(_goal(), caps=_fake_caps())
    rl = next(s for s in payload["stages"] if s["stage"] == "rl")
    card = planner._card_default_hparams(rl["algorithm"]).get("max_new_tokens") or 1024
    assert rl["hparams"]["max_new_tokens"] == card
    assert any(f"max_new_tokens {card}" in str(d) for d in payload["decisions"])


def test_rl_generation_budget_from_the_operator_wins_and_is_priced(patched) -> None:
    facts = _goal()["data_facts"]
    small = planner.plan(_goal(data_facts={**facts, "rl_max_new_tokens": 128}), caps=_fake_caps())
    large = planner.plan(_goal(), caps=_fake_caps())
    rl_small, rl_large = (next(s for s in p["stages"] if s["stage"] == "rl") for p in (small, large))
    assert rl_small["hparams"]["max_new_tokens"] == 128
    assert rl_small["estimate"]["tokens"] < rl_large["estimate"]["tokens"]  # estimate == command


def test_rl_generation_budget_from_the_recipe_is_priced(patched, monkeypatch) -> None:
    original = planner._recipe_stage_hparams
    monkeypatch.setattr(planner, "_recipe_stage_hparams",
                        lambda *a, **k: {**original(*a, **k), "max_new_tokens": 512})
    payload = planner.plan(_goal(), caps=_fake_caps())
    rl = next(s for s in payload["stages"] if s["stage"] == "rl")
    assert rl["hparams"]["max_new_tokens"] == 512
    assert any("512 generated" in a for a in str(payload["decisions"]).split(";"))


def test_rl_memory_is_priced_on_fs_real_step_shape(patched) -> None:
    # measured 2026-10-18: priced as seq 4096 x batch 2 (55.6 GB), FS held 32 rows x 1280 tokens and OOMed at 184 GiB
    payload = planner.plan(_goal(), caps=_fake_caps())
    rl = next(s for s in payload["stages"] if s["stage"] == "rl")
    hp = rl["hparams"]
    rows = int(hp.get("prompts_per_step") or 2) * int(hp.get("group_size") or 4)
    seq = 256 + int(hp["max_new_tokens"])
    assert any(f"rl: {rows} rows x {seq} tokens per step" in str(d) for d in payload["decisions"])
    assert isinstance(hp["gradient_checkpointing"], bool)  # always emitted, never FS's silent default
    assert int(hp.get("logprob_micro_batch") or rows) <= rows


def test_emit_rl_passes_the_memory_knobs() -> None:
    from foundationskills.interfaces.fs import emit_rl

    assert emit_rl._HPARAM_TO_RL["logprob_micro_batch"] == "logprob_micro_batch"
    assert emit_rl._HPARAM_TO_RL["gradient_checkpointing"] == "gradient_checkpointing"


def test_rl_chat_template_kwargs_are_declared_never_inferred(patched) -> None:
    plain = planner.plan(_goal(), caps=_fake_caps())
    assert all("chat_template_kwargs" not in s["hparams"] for s in plain["stages"])
    declared = planner.plan(_goal(rl_chat_template_kwargs={"enable_thinking": False}), caps=_fake_caps())
    by_stage = {s["stage"]: s for s in declared["stages"]}
    assert by_stage["rl"]["hparams"]["chat_template_kwargs"] == {"enable_thinking": False}
    assert "chat_template_kwargs" not in by_stage["sft"]["hparams"]


def test_emit_rl_passes_chat_template_kwargs() -> None:
    from foundationskills.interfaces.fs import emit_rl

    assert emit_rl._HPARAM_TO_RL["chat_template_kwargs"] == "chat_template_kwargs"
