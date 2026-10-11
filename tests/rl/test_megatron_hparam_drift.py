"""CPU legs for the Megatron RL lane's objective hyperparameter drift on save.

The lane records its hyperparameters at step 0 and hands the live values to every
save; ``objective.hparam_drift`` adjudicates the two. These legs pin how the live
values are read (the learning rate comes from the optimizer's param group, never
``args.lr``) and the three verdicts the gate can return on a save: SKIP with no
objective context, PASS on a matching record, and a refusal once the live rate has
moved. Everything runs on CPU over a real tiny ``LlamaForCausalLM``.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from types import MappingProxyType, SimpleNamespace
from typing import Any

from foundationscale.gates.objective_gates import fingerprint_hparams
from foundationscale.rl.megatron.driver import lane_hparams, lane_objective_context
from foundationscale.rl.megatron.save import (
    run_policy_save_gates,
    save_and_adjudicate,
    save_policy_checkpoint,
)

_TOPOLOGY = {
    "nodes": 1,
    "gpus_per_node": 1,
    "tensor_parallel": 1,
    "pipeline_parallel": 1,
    "data_parallel": 1,
    "expert_parallel": 1,
    "context_parallel": 1,
}


def _tiny_llama() -> Any:
    from transformers import LlamaConfig, LlamaForCausalLM

    config = LlamaConfig(
        vocab_size=32,
        hidden_size=8,
        intermediate_size=16,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=2,
        tie_word_embeddings=False,
    )
    return LlamaForCausalLM(config)


def _save(tmp_path: Path, model: Any, names: Any) -> Path:
    return save_policy_checkpoint(
        model,
        None,
        tmp_path / "final",
        declared_names=names,
        run_id="unit",
        topology=_TOPOLOGY,
        config={"steps": 1},
    )


def _hparams() -> dict[str, Any]:
    """The lane's live hyperparameters: ``args.lr`` is 9.0, the optimizer's rate is 3e-7."""
    args = argparse.Namespace(
        algorithm="dr_grpo",
        lr=9.0,
        gbs=8,
        mbs=1,
        seq_len=512,
        group_size=4,
        prompts_per_step=2,
        max_new_tokens=64,
        temperature=0.7,
    )
    optimizer = SimpleNamespace(param_groups=[{"lr": 3e-7}])
    return lane_hparams(args, optimizer)


def _verdicts(report: Any) -> dict[str, str]:
    return {result.gate_id: result.verdict.value for result in report.results}


def test_lane_hparams_reads_the_learning_rate_from_the_optimizer_param_group() -> None:
    hp = _hparams()
    assert hp["learning_rate"] == 3e-7
    assert set(hp) == {
        "algorithm",
        "learning_rate",
        "gbs",
        "mbs",
        "seq_len",
        "group_size",
        "prompts_per_step",
        "max_new_tokens",
        "temperature",
    }


def test_control_no_objective_context_leaves_the_drift_gate_at_skip(tmp_path: Path) -> None:
    model = _tiny_llama()
    out = _save(tmp_path, model, model.state_dict())
    report = run_policy_save_gates(out, first=False)
    assert _verdicts(report)["objective.hparam_drift"] == "SKIP"


def test_a_step0_record_matching_the_live_values_passes_the_drift_gate(
    tmp_path: Path,
) -> None:
    hp = _hparams()
    step0 = MappingProxyType(hp)
    fp = fingerprint_hparams(step0)
    objective = lane_objective_context(step0, fp, dict(hp))
    model = _tiny_llama()
    out = _save(tmp_path, model, model.state_dict())
    report = run_policy_save_gates(out, first=False, objective=objective)
    assert _verdicts(report)["objective.hparam_drift"] == "PASS"
    assert report.ok


def test_a_live_learning_rate_that_moved_refuses_the_save(tmp_path: Path) -> None:
    hp = _hparams()
    step0 = MappingProxyType(hp)
    fp = fingerprint_hparams(step0)
    live = dict(hp)
    live["learning_rate"] = 6e-7
    objective = lane_objective_context(step0, fp, live)
    model = _tiny_llama()
    out = _save(tmp_path, model, model.state_dict())
    report = run_policy_save_gates(out, first=False, objective=objective)
    verdict = _verdicts(report)["objective.hparam_drift"]
    assert verdict not in {"PASS", "SKIP"}
    assert not report.ok


def test_save_and_adjudicate_threads_the_objective_context(tmp_path: Path) -> None:
    hp = _hparams()
    step0 = MappingProxyType(hp)
    objective = lane_objective_context(step0, fingerprint_hparams(step0), dict(hp))
    model = _tiny_llama()
    record, refusal = save_and_adjudicate(
        model,
        None,
        tmp_path / "ck",
        tag="final",
        step=4,
        first=False,
        unwritten=0,
        declared_names=model.state_dict(),
        run_id="t",
        topology=_TOPOLOGY,
        config={},
        objective=objective,
    )
    assert record["gates"]["objective.hparam_drift"] == "PASS"
    assert refusal == ""
