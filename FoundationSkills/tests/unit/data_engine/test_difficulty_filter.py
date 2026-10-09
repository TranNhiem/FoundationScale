"""Tests for the ``difficulty_filter`` op: injected rollouts (no GPU, no models, no network) scored by the REAL
FS parser and reward -- what this filter measures must be exactly the sample RLTrainer trains on.

``OpUnavailable`` arrives with this op's dependency contract; the tiny shim below keeps this file runnable
against a base.py from the same lane and is a no-op once the class is declared there.
"""
from __future__ import annotations

import importlib.util

import pytest

import foundationskills.skills.data_engine.ops.base as _base

if not hasattr(_base, "OpUnavailable"):  # pragma: no cover - no-op once ops/base.py declares it
    class OpUnavailable(RuntimeError):
        pass

    _base.OpUnavailable = OpUnavailable

from foundationscale.rl.corpus import Sample, _parse_record
from foundationskills.skills.data_engine.ops import OPS
from foundationskills.skills.data_engine.ops import difficulty_filter as df_mod
from foundationskills.skills.data_engine.ops.base import OpStats, OpUnavailable
from foundationskills.skills.data_engine.ops.semantic_dedup import _resolve_revision

op = OPS["difficulty_filter"]
MODEL = "unit/fake-model"
DEFAULT_CFG = {"model": MODEL}
# several distinct A--Z letters: MCQLetterReward abstains on it, it is never read as a letter
ABSTAINING = "no clear pick from these options"


def check_invariant(stats: OpStats) -> None:
    assert stats.records_in == stats.records_out + sum(stats.dropped.values())


def rec(rid: str, *, answer: str | None = "C", meta: dict | None = None) -> dict:
    """One data_engine rl-format record (answer declared under ``answer`` unless dropped)."""
    record = {
        "id": rid,
        "conversations": [
            {"from": "human", "value": "Which option fits?\nAnswer with a single letter."},
            {"from": "gpt", "value": "C"},
        ],
    }
    if answer is not None:
        record["answer"] = answer
    if meta is not None:
        record["meta"] = meta
    return record


def fake_rollouts(monkeypatch, completions: dict[str, list[str]]) -> dict:
    """Patch the rollout backend: canned completions keyed by record id; every call recorded."""

    calls: dict = {"factory": [], "samples": []}

    def factory(cfg: dict):
        calls["factory"].append(dict(cfg))

        def rollout(samples):
            calls["samples"].append(list(samples))
            return [list(completions[s.sample_id]) for s in samples]

        return rollout

    monkeypatch.setattr(df_mod, "_ROLLOUT_FACTORY", factory)
    return calls


def run_op(records, cfg: dict | None = None) -> tuple[list[dict], OpStats]:
    stats = OpStats(name="difficulty_filter")
    out = list(op(iter(records), dict(DEFAULT_CFG if cfg is None else cfg), stats))
    check_invariant(stats)
    return out, stats


def specs_available(monkeypatch) -> None:
    """importlib probes answer """ + '"installed"' + """ for the three runtime packages, whatever this env has."""

    real = importlib.util.find_spec

    def fake(name, package=None):
        return real("builtins") if name in ("foundationscale", "torch", "transformers") else real(name, package)

    monkeypatch.setattr(importlib.util, "find_spec", fake)


def test_registered_op_and_schema() -> None:
    assert op.name == "difficulty_filter"
    assert op.config_schema is df_mod.CONFIG_SCHEMA
    schema = df_mod.CONFIG_SCHEMA
    assert schema["additionalProperties"] is False
    assert schema["required"] == ["model"]
    assert schema["properties"]["k"]["minimum"] == 2
    assert schema["properties"]["max_new_tokens"]["default"] == 64
    assert schema["properties"]["dtype"]["enum"] == ["bfloat16", "float16", "float32"]


def test_band_keeps_mixed_groups_only(monkeypatch) -> None:
    records = [
        rec("easy", meta={"source": "unit", "lang": "en"}),  # all-correct group: zero reward variance
        rec("hard", meta={"source": "unit"}),  # all-wrong group: zero reward variance
        rec("m1", meta={"source": "unit"}),
        rec("m2"),
    ]
    fake_rollouts(
        monkeypatch,
        {
            "easy": ["C"] * 8,
            "hard": ["A"] * 8,
            "m1": ["C"] * 4 + ["A"] * 4,
            "m2": ["A", "C"] * 4,
        },
    )
    out, stats = run_op(records)

    report = stats.to_dict()
    assert report["name"] == "difficulty_filter"
    assert report["records_in"] == 4 and report["records_out"] == 2
    assert report["dropped"] == {"difficulty_too_easy": 1, "difficulty_too_hard": 1}
    assert [r["id"] for r in out] == ["m1", "m2"]  # input order preserved over the kept records
    assert out[0]["meta"] == {"source": "unit", "pass_rate": 0.5, "pass_scored": 8, "pass_k": 8}
    assert out[1]["meta"] == {"pass_rate": 0.5, "pass_scored": 8, "pass_k": 8}  # meta created when absent
    assert out[0]["conversations"] == records[2]["conversations"]  # records otherwise unchanged
    # copies: the input records are never mutated
    assert out[0] is not records[2]
    assert records[2]["meta"] == {"source": "unit"}
    assert stats.extra["kept_fraction"] == 0.5


def test_custom_band_boundaries(monkeypatch) -> None:
    records = [rec("a1"), rec("b2"), rec("c3"), rec("d4")]
    fake_rollouts(
        monkeypatch,
        {
            "a1": ["A"] * 5,  # 0/5 -> at/below keep_above
            "b2": ["C"] + ["A"] * 4,  # 1/5 == keep_above -> too hard
            "c3": ["C"] * 2 + ["A"] * 3,  # 2/5 inside (0.2, 0.8) -> kept
            "d4": ["C"] * 4 + ["A"],  # 4/5 == keep_below -> too easy
        },
    )
    out, stats = run_op(records, {**DEFAULT_CFG, "k": 5, "keep_above": 0.2, "keep_below": 0.8})

    assert [r["id"] for r in out] == ["c3"]
    assert dict(stats.dropped) == {"difficulty_too_hard": 2, "difficulty_too_easy": 1}
    assert out[0]["meta"]["pass_rate"] == 0.4
    assert out[0]["meta"]["pass_scored"] == 5
    assert out[0]["meta"]["pass_k"] == 5
    assert stats.extra["keep_above"] == 0.2 and stats.extra["keep_below"] == 0.8


def test_abstentions_leave_the_denominator(monkeypatch) -> None:
    records = [rec("all-abstain"), rec("one-abstain")]
    fake_rollouts(
        monkeypatch,
        {
            "all-abstain": [ABSTAINING] * 8,
            "one-abstain": ["C"] * 4 + ["A"] * 3 + [ABSTAINING],
        },
    )
    out, stats = run_op(records)

    assert [r["id"] for r in out] == ["one-abstain"]
    assert dict(stats.dropped) == {"difficulty_unscorable": 1}  # all-abstain: scored < 2
    assert out[0]["meta"]["pass_rate"] == round(4 / 7, 4)  # the abstention left the denominator
    assert out[0]["meta"]["pass_scored"] == 7
    assert stats.extra["abstentions"] == 8 + 1
    assert stats.extra["rollouts"] == 16
    assert stats.extra["pass_rate_hist"] == {"0.571": 1}


def test_no_gold_and_unparseable_records_drop(monkeypatch) -> None:
    calls = fake_rollouts(monkeypatch, {})
    records = [
        rec("no-answer-key", answer=None),
        rec("multi-gold", answer="AC"),  # a declared gold is used verbatim or not at all
        rec("word-gold", answer="GDP"),
        {"id": "no-conversations"},
        "not-a-dict",
    ]
    out, stats = run_op(records)

    assert out == []
    assert stats.records_in == 5 and stats.records_out == 0
    assert dict(stats.dropped) == {"difficulty_no_gold": 3, "difficulty_unparseable_record": 2}
    assert stats.extra["rollouts"] == 0 and stats.extra["prompts_scored"] == 0
    assert stats.extra["abstentions"] == 0
    assert stats.extra["kept_fraction"] is None
    assert calls["samples"] == []  # nothing measurable: the rollout is never asked for


def test_stats_extra_histogram_rollouts_and_fraction(monkeypatch) -> None:
    records = [rec("r1"), rec("r2"), rec("r3"), rec("r4")]
    fake_rollouts(
        monkeypatch,
        {
            "r1": ["C"] * 8,  # 1.0 -> too easy
            "r2": ["C"] * 4 + ["A"] * 4,  # 0.5 -> kept
            "r3": ["A"] * 4 + ["C"] * 4,  # 0.5 -> kept
            "r4": ["A"] * 8,  # 0.0 -> too hard
        },
    )
    out, stats = run_op(records)
    extra = stats.extra

    assert extra["model"] == MODEL
    assert extra["model_revision"] == _resolve_revision(MODEL)
    assert (extra["k"], extra["temperature"], extra["top_p"], extra["max_new_tokens"]) == (8, 1.0, 1.0, 64)
    assert (extra["keep_above"], extra["keep_below"], extra["seed"]) == (0.0, 1.0, 0)
    assert extra["rollouts"] == 32  # k * (prompts scored + prompts unscorable)
    assert extra["abstentions"] == 0
    assert extra["prompts_scored"] == 4
    assert list(extra["pass_rate_hist"]) == ["0.000", "0.500", "1.000"]  # sorted by key
    assert extra["pass_rate_hist"] == {"0.000": 1, "0.500": 2, "1.000": 1}  # every scored prompt, kept or dropped
    assert sum(extra["pass_rate_hist"].values()) == extra["prompts_scored"]
    assert extra["kept_fraction"] == 2 / 4
    assert isinstance(extra["wall_s"], float)
    assert stats.backend == f"hf-generate:{MODEL}@{_resolve_revision(MODEL)}"


def test_backend_called_once_with_fs_samples(monkeypatch) -> None:
    records = [rec("p0"), rec("p1"), rec("p2")]
    cfg = {**DEFAULT_CFG, "k": 2, "temperature": 0.7, "seed": 3}
    calls = fake_rollouts(monkeypatch, {"p0": ["C", "C"], "p1": ["A", "A"], "p2": ["C", "A"]})
    out, stats = run_op(records, cfg)

    assert len(calls["factory"]) == 1  # the backend is loaded ONCE per invocation, never per record
    assert calls["factory"][0] == cfg  # the factory receives the cfg
    assert len(calls["samples"]) == 1
    samples = calls["samples"][0]
    assert [s.sample_id for s in samples] == ["p0", "p1", "p2"]
    assert all(isinstance(s, Sample) for s in samples)
    for index, sample in enumerate(samples):
        expected = _parse_record(records[index], index, gold_key="answer")  # the RL-trainer parse, verbatim
        assert sample == expected
        assert sample.prompt_turns == expected.prompt_turns  # and the same prompt surface, byte for byte
    assert [r["id"] for r in out] == ["p2"]
    assert dict(stats.dropped) == {"difficulty_too_easy": 1, "difficulty_too_hard": 1}


def test_answer_pattern_narrows_the_answer_surface(monkeypatch) -> None:
    cfg = {**DEFAULT_CFG, "k": 4, "answer_pattern": r"Answer: ([A-Z])"}
    records = [rec("declared"), rec("bare")]
    fake_rollouts(
        monkeypatch,
        {
            "declared": ["Answer: C", "Answer: A", "Answer: C", ABSTAINING],
            "bare": ["C", "A", "Answer: C", "Answer: C"],
        },
    )
    out, stats = run_op(records, cfg)

    assert [r["id"] for r in out] == ["declared"]
    assert out[0]["meta"]["pass_rate"] == round(2 / 3, 4)
    assert dict(stats.dropped) == {"difficulty_too_easy": 1}  # bare prompt read under this surface: 2/2 scored
    assert stats.extra["abstentions"] == 1 + 2  # ABSTAINING plus the two bare letters this surface cannot read


def test_empty_band_raises_op_unavailable(monkeypatch) -> None:
    calls = fake_rollouts(monkeypatch, {"p": ["C"] * 8})
    with pytest.raises(OpUnavailable) as excinfo:
        run_op([rec("p")], {**DEFAULT_CFG, "keep_above": 0.8, "keep_below": 0.2})
    message = str(excinfo.value)
    assert "0.8" in message and "0.2" in message  # the refusal names both values
    assert calls["factory"] == []  # refused before any backend work


def test_preflight_names_missing_model_dir() -> None:
    problems = df_mod.preflight({"model": "UnitTests/absent-model-1234"})
    assert any("UnitTests/absent-model-1234" in p for p in problems)


def test_preflight_schema_band_and_import_problems(tmp_path, monkeypatch) -> None:
    specs_available(monkeypatch)
    model = str(tmp_path)
    assert any("k" in p and "minimum" in p for p in df_mod.preflight({"model": model, "k": 1}))  # schema problem
    assert any("model" in p for p in df_mod.preflight({"k": 4}))  # required key missing
    assert any("bogus" in p for p in df_mod.preflight({"model": model, "bogus": 1}))  # additionalProperties false
    assert any("dtype" in p for p in df_mod.preflight({"model": model, "dtype": "float64"}))
    assert any("keep_above" in p for p in df_mod.preflight({"model": model, "keep_above": 0.5, "keep_below": 0.5}))


def test_preflight_reports_missing_imports(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(importlib.util, "find_spec", lambda name, package=None: None)
    problems = df_mod.preflight({"model": str(tmp_path)})
    for name in ("foundationscale", "torch", "transformers"):
        assert any(name in p for p in problems)


def test_preflight_clean_with_local_model(tmp_path, monkeypatch) -> None:
    specs_available(monkeypatch)
    assert df_mod.preflight({"model": str(tmp_path)}) == []


def test_a_short_backend_answer_is_refused_not_dropped(monkeypatch) -> None:
    monkeypatch.setattr(df_mod, "_ROLLOUT_FACTORY", lambda cfg: (lambda samples: [["A", "B"]]))
    with pytest.raises(OpUnavailable, match="returned 1 completion group"):
        run_op([rec("one"), rec("two")], DEFAULT_CFG)


def test_readiness_carries_the_measured_pass_rate_hist() -> None:
    from foundationskills.skills.data_engine.report import build_readiness

    op_stats = {"name": "difficulty_filter", "records_in": 4, "records_out": 1, "dropped": {"difficulty_too_easy": 3},
                "extra": {"pass_rate_hist": {"0.500": 1, "1.000": 3}, "kept_fraction": 0.25}}
    report = build_readiness({"format": "rl", "num_records": 1}, [op_stats], {"target_format": "rl"})
    assert report["stats"]["rl_pass_rate_hist"] == {"0.500": 1, "1.000": 3}
    assert report["stats"]["rl_kept_fraction"] == 0.25
    plain = build_readiness({"format": "rl", "num_records": 1}, [], {"target_format": "rl"})
    assert "rl_pass_rate_hist" not in plain["stats"]


def test_non_object_meta_is_dropped_before_rollouts(monkeypatch) -> None:
    calls = fake_rollouts(monkeypatch, {"ok": ["C", "C", "A", "A"]})
    bad = rec("listy")
    bad["meta"] = ["curated"]
    out, stats = run_op([bad, rec("ok", meta={"src": "x"})])
    assert [r["id"] for r in out] == ["ok"]
    assert out[0]["meta"]["src"] == "x" and out[0]["meta"]["pass_rate"] == 0.5
    assert dict(stats.dropped) == {"difficulty_meta_not_object": 1}
    assert [s.sample_id for s in calls["samples"][0]] == ["ok"]  # never spent rollouts on it
